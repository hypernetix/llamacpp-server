package httpserver

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/hypernetix/llamacpp_server/internal/inferenceengine"
)

// --- ID generation ---

func generateID(prefix string) string {
	b := make([]byte, 12)
	rand.Read(b)
	return prefix + hex.EncodeToString(b)
}

// --- OpenAI request/response types ---

type oaiModelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type oaiModelList struct {
	Object string           `json:"object"`
	Data   []oaiModelObject `json:"data"`
}

type oaiCompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`
	TopP        *float32 `json:"top_p,omitempty"`
	Stream      bool     `json:"stream"`
	Stop        any      `json:"stop,omitempty"`
}

type oaiCompletionChoice struct {
	Text         string  `json:"text"`
	Index        int     `json:"index"`
	FinishReason *string `json:"finish_reason"`
}

type oaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type oaiCompletionResponse struct {
	ID      string                `json:"id"`
	Object  string                `json:"object"`
	Created int64                 `json:"created"`
	Model   string                `json:"model"`
	Choices []oaiCompletionChoice `json:"choices"`
	Usage   *oaiUsage             `json:"usage,omitempty"`
}

type oaiChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type oaiChatCompletionRequest struct {
	Model       string           `json:"model"`
	Messages    []oaiChatMessage `json:"messages"`
	MaxTokens   *int             `json:"max_tokens,omitempty"`
	Temperature *float32         `json:"temperature,omitempty"`
	TopP        *float32         `json:"top_p,omitempty"`
	Stream      bool             `json:"stream"`
	Stop        any              `json:"stop,omitempty"`
}

type oaiChatChoiceMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type oaiChatChoiceDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type oaiChatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      *oaiChatChoiceMessage `json:"message,omitempty"`
	Delta        *oaiChatChoiceDelta   `json:"delta,omitempty"`
	FinishReason *string               `json:"finish_reason"`
}

type oaiChatCompletionResponse struct {
	ID      string                    `json:"id"`
	Object  string                    `json:"object"`
	Created int64                     `json:"created"`
	Model   string                    `json:"model"`
	Choices []oaiChatCompletionChoice `json:"choices"`
	Usage   *oaiUsage                 `json:"usage,omitempty"`
}

type oaiErrorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

type oaiErrorResponse struct {
	Error oaiErrorDetail `json:"error"`
}

// --- Chat template ---
// ChatML is the most common format for instruction-tuned models (used by
// SmolLM2-Instruct, Qwen, Mistral-Instruct, many others).

func applyChatMLTemplate(messages []oaiChatMessage) string {
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(msg.Role)
		sb.WriteString("\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

// --- Handlers ---

func (s *Server) handleV1Models(w http.ResponseWriter, r *http.Request) {
	paths := s.service.ListModels()
	now := time.Now().Unix()
	models := make([]oaiModelObject, 0, len(paths))
	for _, p := range paths {
		models = append(models, oaiModelObject{
			ID:      p,
			Object:  "model",
			Created: now,
			OwnedBy: "local",
		})
	}
	writeJSON(w, http.StatusOK, oaiModelList{
		Object: "list",
		Data:   models,
	})
}

// --- /v1/completions ---

func (s *Server) handleV1Completions(w http.ResponseWriter, r *http.Request) {
	var req oaiCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}

	maxTokens := 16
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	s.logger.Infof("v1/completions: model=%s, max_tokens=%d, stream=%v", req.Model, maxTokens, req.Stream)

	args := buildOAIPredictArgs(maxTokens, req.Temperature, req.TopP)

	if req.Stream {
		s.handleV1CompletionsStream(w, r, &req, args)
	} else {
		s.handleV1CompletionsNonStream(w, r, &req, args)
	}
}

func (s *Server) handleV1CompletionsNonStream(w http.ResponseWriter, r *http.Request, req *oaiCompletionRequest, args inferenceengine.PredictArgs) {
	text, err := s.service.Predict(req.Model, req.Prompt, args, nil)
	if err != nil {
		s.logger.Errorf("v1/completions failed: %v", err)
		writeOAIError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	finishReason := "stop"
	writeJSON(w, http.StatusOK, oaiCompletionResponse{
		ID:      generateID("cmpl-"),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []oaiCompletionChoice{{
			Text:         text,
			Index:        0,
			FinishReason: &finishReason,
		}},
		Usage: &oaiUsage{},
	})
}

func (s *Server) handleV1CompletionsStream(w http.ResponseWriter, r *http.Request, req *oaiCompletionRequest, args inferenceengine.PredictArgs) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOAIError(w, http.StatusInternalServerError, "server_error", "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := generateID("cmpl-")
	created := time.Now().Unix()
	ctx := r.Context()

	streamFunc := func(token, tokens int, message string) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		chunk := oaiCompletionResponse{
			ID:      id,
			Object:  "text_completion",
			Created: created,
			Model:   req.Model,
			Choices: []oaiCompletionChoice{{
				Text:  message,
				Index: 0,
			}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return nil
	}

	_, err := s.service.Predict(req.Model, req.Prompt, args, streamFunc)
	if err != nil {
		s.logger.Errorf("v1/completions streaming failed: %v", err)
		return
	}

	finishReason := "stop"
	final := oaiCompletionResponse{
		ID:      id,
		Object:  "text_completion",
		Created: created,
		Model:   req.Model,
		Choices: []oaiCompletionChoice{{
			Index:        0,
			FinishReason: &finishReason,
		}},
	}
	data, _ := json.Marshal(final)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// --- /v1/chat/completions ---

func (s *Server) handleV1ChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req oaiChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "messages is required and must be non-empty")
		return
	}

	maxTokens := 256
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	s.logger.Infof("v1/chat/completions: model=%s, messages=%d, max_tokens=%d, stream=%v",
		req.Model, len(req.Messages), maxTokens, req.Stream)

	prompt := applyChatMLTemplate(req.Messages)
	args := buildOAIPredictArgs(maxTokens, req.Temperature, req.TopP)

	if req.Stream {
		s.handleV1ChatCompletionsStream(w, r, &req, prompt, args)
	} else {
		s.handleV1ChatCompletionsNonStream(w, r, &req, prompt, args)
	}
}

func (s *Server) handleV1ChatCompletionsNonStream(w http.ResponseWriter, r *http.Request, req *oaiChatCompletionRequest, prompt string, args inferenceengine.PredictArgs) {
	text, err := s.service.Predict(req.Model, prompt, args, nil)
	if err != nil {
		s.logger.Errorf("v1/chat/completions failed: %v", err)
		writeOAIError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	finishReason := "stop"
	writeJSON(w, http.StatusOK, oaiChatCompletionResponse{
		ID:      generateID("chatcmpl-"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []oaiChatCompletionChoice{{
			Index: 0,
			Message: &oaiChatChoiceMessage{
				Role:    "assistant",
				Content: text,
			},
			FinishReason: &finishReason,
		}},
		Usage: &oaiUsage{},
	})
}

func (s *Server) handleV1ChatCompletionsStream(w http.ResponseWriter, r *http.Request, req *oaiChatCompletionRequest, prompt string, args inferenceengine.PredictArgs) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOAIError(w, http.StatusInternalServerError, "server_error", "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := generateID("chatcmpl-")
	created := time.Now().Unix()
	ctx := r.Context()

	// First chunk: assistant role announcement
	roleChunk := oaiChatCompletionResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   req.Model,
		Choices: []oaiChatCompletionChoice{{
			Index: 0,
			Delta: &oaiChatChoiceDelta{Role: "assistant"},
		}},
	}
	data, _ := json.Marshal(roleChunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()

	streamFunc := func(token, tokens int, message string) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		chunk := oaiChatCompletionResponse{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []oaiChatCompletionChoice{{
				Index: 0,
				Delta: &oaiChatChoiceDelta{Content: message},
			}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return nil
	}

	_, err := s.service.Predict(req.Model, prompt, args, streamFunc)
	if err != nil {
		s.logger.Errorf("v1/chat/completions streaming failed: %v", err)
		return
	}

	// Final chunk: finish_reason with empty delta
	finishReason := "stop"
	final := oaiChatCompletionResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   req.Model,
		Choices: []oaiChatCompletionChoice{{
			Index:        0,
			Delta:        &oaiChatChoiceDelta{},
			FinishReason: &finishReason,
		}},
	}
	data, _ = json.Marshal(final)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// --- Shared helpers ---

func buildOAIPredictArgs(maxTokens int, temperature, topP *float32) inferenceengine.PredictArgs {
	args := inferenceengine.PredictArgs{
		NPredict:          maxTokens,
		Temp:              0,
		TopP:              0,
		MinP:              0.05,
		MinTokensToKeep:   1,
		RepetitionPenalty: 1.0,
		LengthPenalty:     1.0,
		RandomSeed:        -1,
	}
	if temperature != nil {
		args.Temp = *temperature
	}
	if topP != nil {
		args.TopP = *topP
	}
	return args
}

func writeOAIError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(oaiErrorResponse{
		Error: oaiErrorDetail{
			Message: message,
			Type:    errType,
		},
	})
}
