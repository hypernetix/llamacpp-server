package llmservice

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/hypernetix/llamacpp_server/internal/logging"
)

type httpClient struct {
	once          sync.Once
	serverProcess Process
	baseURL       string
	client        *http.Client
	logger        logging.SprintfLogger
}

func newHTTPClient(host string, port int, serverProcess Process, logger logging.SprintfLogger) (LLMService, error) {
	baseURL := fmt.Sprintf("http://%s:%d", host, port)
	logger.Debugf("HTTP client targeting %s", baseURL)

	return &httpClient{
		serverProcess: serverProcess,
		baseURL:       baseURL,
		client:        &http.Client{},
		logger:        logger,
	}, nil
}

func (c *httpClient) Shutdown() {
	c.once.Do(func() {
		if c.serverProcess != nil {
			c.serverProcess.Stop()
			c.serverProcess = nil
		}
	})
}

func (c *httpClient) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		c.logger.Errorf("Ping: HTTP request failed: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("health check failed: %s %s", resp.Status, string(body))
	}
	c.logger.Debugf("Ping done (HTTP)")
	return nil
}

func (c *httpClient) LoadModel(ctx context.Context, name string, progress chan<- float32) error {
	c.logger.Infof("LoadModel (HTTP): %s", name)

	body, _ := json.Marshal(map[string]string{"path": name})
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/models/load", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		c.logger.Errorf("LoadModel: HTTP request failed: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("load model failed: %s %s", resp.Status, string(respBody))
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			if strings.HasPrefix(line, "event: error") {
				if scanner.Scan() {
					errData := strings.TrimPrefix(scanner.Text(), "data: ")
					return fmt.Errorf("server error: %s", errData)
				}
			}
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var evt struct {
			Progress float32 `json:"progress"`
		}
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			c.logger.Warnf("LoadModel: failed to parse SSE event: %v", err)
			continue
		}
		if progress != nil {
			progress <- evt.Progress
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("SSE stream read error: %v", err)
	}

	c.logger.Infof("LoadModel (HTTP): loaded %s", name)
	return nil
}

type httpCompletionRequest struct {
	Model       string                 `json:"model"`
	Prompt      string                 `json:"prompt"`
	Stream      bool                   `json:"stream"`
	MaxTokens   int                    `json:"max_tokens"`
	Temperature float64                `json:"temperature"`
	TopP        float64                `json:"top_p"`
	TopK        int                    `json:"top_k"`
	Options     *httpCompletionOptions `json:"options,omitempty"`
}

type httpCompletionOptions struct {
	MinP              *float32 `json:"min_p,omitempty"`
	RepetitionPenalty *float32 `json:"repetition_penalty,omitempty"`
	RandomSeed        *int32   `json:"random_seed,omitempty"`
}

type httpCompletionResponse struct {
	Message string `json:"message"`
	Token   int    `json:"token"`
	Tokens  int    `json:"tokens"`
}

func (c *httpClient) Predict(ctx context.Context, req PredictRequest, resp chan<- PredictResponse) error {
	c.logger.Infof("Predict (HTTP): model=%s, max_tokens=%d, temp=%.3f",
		req.ModelName, req.MaxTokens, req.Temperature)

	if !req.Stream {
		return fmt.Errorf("non-streaming predict not supported in test client")
	}

	httpReq := buildHTTPRequest(req)

	body, err := json.Marshal(httpReq)
	if err != nil {
		return err
	}

	httpRequest, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpRequest.Header.Set("Content-Type", "application/json")

	httpResp, err := c.client.Do(httpRequest)
	if err != nil {
		c.logger.Errorf("Predict: HTTP request failed: %v", err)
		return err
	}

	if httpResp.StatusCode != http.StatusOK {
		defer httpResp.Body.Close()
		respBody, _ := io.ReadAll(httpResp.Body)
		return fmt.Errorf("predict failed: %s %s", httpResp.Status, string(respBody))
	}

	go func() {
		defer httpResp.Body.Close()

		scanner := bufio.NewScanner(httpResp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				if strings.HasPrefix(line, "event: error") {
					if scanner.Scan() {
						errData := strings.TrimPrefix(scanner.Text(), "data: ")
						resp <- PredictResponse{Error: fmt.Errorf("server error: %s", errData), Done: true}
					}
					return
				}
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				resp <- PredictResponse{Done: true}
				return
			}
			var evt httpCompletionResponse
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				c.logger.Warnf("Predict: failed to parse SSE event: %v", err)
				continue
			}
			resp <- PredictResponse{
				Message: evt.Message,
				Token:   int32(evt.Token),
				Tokens:  int32(evt.Tokens),
			}
		}

		if err := scanner.Err(); err != nil {
			resp <- PredictResponse{Error: fmt.Errorf("SSE stream read error: %v", err), Done: true}
			return
		}

		resp <- PredictResponse{Done: true}
	}()

	return nil
}

func buildHTTPRequest(req PredictRequest) *httpCompletionRequest {
	topP := 1.0
	if req.TopP != nil {
		topP = *req.TopP
	}
	topK := 0
	if req.TopK != nil {
		topK = *req.TopK
	}

	httpReq := &httpCompletionRequest{
		Model:       req.ModelName,
		Prompt:      req.Message,
		Stream:      true,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        topP,
		TopK:        topK,
	}

	if req.MinP != nil || req.RepetitionPenalty != nil || req.RandomSeed != nil {
		opts := &httpCompletionOptions{}
		if req.MinP != nil {
			v := float32(*req.MinP)
			opts.MinP = &v
		}
		if req.RepetitionPenalty != nil {
			v := float32(*req.RepetitionPenalty)
			opts.RepetitionPenalty = &v
		}
		if req.RandomSeed != nil {
			v := int32(*req.RandomSeed)
			opts.RandomSeed = &v
		}
		httpReq.Options = opts
	}

	return httpReq
}
