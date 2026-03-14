package httpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"

	"github.com/hypernetix/llamacpp_server/internal/inferenceengine"
	"github.com/hypernetix/llamacpp_server/internal/llmservice"
	"github.com/hypernetix/llamacpp_server/internal/logging"
)

type Server struct {
	service    *llmservice.Service
	logger     logging.SprintfLogger
	httpServer *http.Server
}

func NewServer(service *llmservice.Service, addr string, logger logging.SprintfLogger) *Server {
	s := &Server{
		service: service,
		logger:  logger.With("module", "httpserver"),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("POST /models/load", s.handleLoadModel)
	mux.HandleFunc("POST /completions", s.handleCompletions)

	s.httpServer = &http.Server{
		Addr:    addr,
		Handler: mux,
	}
	return s
}

func (s *Server) Start(listener net.Listener) error {
	s.logger.Infof("HTTP server listening at %s", listener.Addr().String())
	return s.httpServer.Serve(listener)
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

// --- Health ---

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// --- Load Model ---

type loadModelRequest struct {
	Path string `json:"path"`
}

type loadModelEvent struct {
	Progress float32 `json:"progress"`
}

func (s *Server) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	var req loadModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: %v", err)
		return
	}
	if req.Path == "" {
		writeError(w, http.StatusBadRequest, "path is required")
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	onProgress := func(progress float32) {
		data, _ := json.Marshal(loadModelEvent{Progress: progress})
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	err := s.service.LoadModel(req.Path, onProgress)
	if err != nil {
		s.logger.Errorf("LoadModel failed: %v", err)
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", jsonString(err.Error()))
		flusher.Flush()
		return
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// --- Completions ---

type completionRequest struct {
	Model       string             `json:"model"`
	Prompt      string             `json:"prompt"`
	Stream      bool               `json:"stream"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float32            `json:"temperature"`
	TopP        float32            `json:"top_p"`
	TopK        int32              `json:"top_k"`
	Options     *completionOptions `json:"options,omitempty"`
}

type completionOptions struct {
	MinP              *float32 `json:"min_p,omitempty"`
	MinTokensToKeep   *int32   `json:"min_tokens_to_keep,omitempty"`
	MaxKvSize         *int32   `json:"max_kv_size,omitempty"`
	PrefillStepSize   *int32   `json:"prefill_step_size,omitempty"`
	KvBits            *int32   `json:"kv_bits,omitempty"`
	KvGroupSize       *int32   `json:"kv_group_size,omitempty"`
	QuantizedKvStart  *int32   `json:"quantized_kv_start,omitempty"`
	RepetitionPenalty *float32 `json:"repetition_penalty,omitempty"`
	LengthPenalty     *float32 `json:"length_penalty,omitempty"`
	DiversityPenalty  *float32 `json:"diversity_penalty,omitempty"`
	NoRepeatNgramSize *int32   `json:"no_repeat_ngram_size,omitempty"`
	RandomSeed        *int32   `json:"random_seed,omitempty"`
}

type completionResponse struct {
	Message string `json:"message"`
	Token   int    `json:"token"`
	Tokens  int    `json:"tokens"`
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: %v", err)
		return
	}

	s.logger.Infof("Completions: model=%s, max_tokens=%d, stream=%v, temp=%.3f",
		req.Model, req.MaxTokens, req.Stream, req.Temperature)

	if req.MaxTokens == 0 {
		writeJSON(w, http.StatusOK, completionResponse{})
		return
	}

	args := buildPredictArgs(&req)

	if req.Stream {
		s.handleStreamingCompletion(w, r, &req, args)
	} else {
		s.handleNonStreamingCompletion(w, r, &req, args)
	}
}

func (s *Server) handleStreamingCompletion(w http.ResponseWriter, r *http.Request, req *completionRequest, args inferenceengine.PredictArgs) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ctx := r.Context()

	streamFunc := func(token, tokens int, message string) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		data, _ := json.Marshal(completionResponse{
			Message: message,
			Token:   token,
			Tokens:  tokens,
		})
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return nil
	}

	_, err := s.service.Predict(req.Model, req.Prompt, args, streamFunc)
	if err != nil {
		s.logger.Errorf("Completions streaming failed: %v", err)
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", jsonString(err.Error()))
		flusher.Flush()
		return
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) handleNonStreamingCompletion(w http.ResponseWriter, r *http.Request, req *completionRequest, args inferenceengine.PredictArgs) {
	response, err := s.service.Predict(req.Model, req.Prompt, args, nil)
	if err != nil {
		s.logger.Errorf("Completions failed: %v", err)
		writeError(w, http.StatusInternalServerError, "prediction failed: %v", err)
		return
	}

	writeJSON(w, http.StatusOK, completionResponse{Message: response})
}

func buildPredictArgs(req *completionRequest) inferenceengine.PredictArgs {
	temp := req.Temperature
	if temp < 0 {
		temp = 0
	}
	topP := req.TopP
	if topP < 0 {
		topP = 0
	}
	topK := req.TopK
	if topK < 0 {
		topK = 0
	}
	maxTokens := req.MaxTokens
	if maxTokens < 0 {
		maxTokens = 0
	}

	args := inferenceengine.PredictArgs{
		NPredict:          maxTokens,
		Temp:              temp,
		TopP:              topP,
		TopK:              topK,
		MinP:              0.05,
		MinTokensToKeep:   1,
		RepetitionPenalty: 1.0,
		LengthPenalty:     1.0,
		RandomSeed:        -1,
	}

	if req.Options == nil {
		return args
	}
	opts := req.Options

	if opts.MinP != nil {
		v := *opts.MinP
		if v < 0 {
			v = 0
		}
		args.MinP = v
	}
	if opts.MinTokensToKeep != nil {
		v := *opts.MinTokensToKeep
		if v < 0 {
			v = 0
		}
		args.MinTokensToKeep = int(v)
	}
	if opts.MaxKvSize != nil {
		v := *opts.MaxKvSize
		if v < 0 {
			v = 0
		}
		args.MaxKvSize = int(v)
	}
	if opts.PrefillStepSize != nil {
		v := *opts.PrefillStepSize
		if v < 0 {
			v = 0
		}
		args.PrefillStepSize = int(v)
	}
	if opts.KvBits != nil {
		v := *opts.KvBits
		if v < 0 {
			v = 0
		}
		args.KvBits = int(v)
	}
	if opts.KvGroupSize != nil {
		v := *opts.KvGroupSize
		if v < 0 {
			v = 0
		}
		args.KvGroupSize = int(v)
	}
	if opts.QuantizedKvStart != nil {
		v := *opts.QuantizedKvStart
		if v < 0 {
			v = 0
		}
		args.QuantizedKvStart = int(v)
	}
	if opts.RepetitionPenalty != nil {
		args.RepetitionPenalty = *opts.RepetitionPenalty
	}
	if opts.LengthPenalty != nil {
		args.LengthPenalty = *opts.LengthPenalty
	}
	if opts.DiversityPenalty != nil {
		args.DiversityPenalty = *opts.DiversityPenalty
	}
	if opts.NoRepeatNgramSize != nil {
		args.NoRepeatNgramSize = int(*opts.NoRepeatNgramSize)
	}
	if opts.RandomSeed != nil {
		args.RandomSeed = int(*opts.RandomSeed)
	}

	return args
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
