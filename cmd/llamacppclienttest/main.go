package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	llmservice "github.com/hypernetix/llamacpp_server/cmd/llamacppclienttest/llmservice"
	"github.com/hypernetix/llamacpp_server/internal/logging"

	flags "github.com/jessevdk/go-flags"
)

type flagOptions struct {
	ModelPath  string  `long:"model" description:"path to the model file"`
	ServerPath string  `long:"server" description:"path to the server executable"`
	AttachHost string  `long:"host" description:"host address to attach to (default: 127.0.0.1)" default:"127.0.0.1"`
	AttachPort int     `long:"port" description:"port to attach to the server (0 = spawn server)"`
	Transport  string  `long:"transport" description:"transport protocol: grpc or http" default:"grpc"`
	Temperature    float64 `long:"temperature" description:"sampling temperature" default:"0.7"`
	TopP           float64 `long:"top-p" description:"top-p sampling" default:"1.0"`
	TopK           int     `long:"top-k" description:"top-k sampling" default:"0"`
	RepeatPenalty  float64 `long:"repeat-penalty" description:"repetition penalty" default:"1.0"`
	MinP           float64 `long:"min-p" description:"min-p sampling" default:"0.05"`
	RandomSeed     int     `long:"seed" description:"random seed for reproducible results (-1 for random)" default:"-1"`
	MaxTokens      int     `long:"max-tokens" description:"maximum tokens to generate" default:"100"`
	TestMode           string `long:"test-mode" description:"test mode: baseline, greedy, seeded, stress, parallel, or backpressure" default:"baseline"`
	ParallelN          int    `long:"parallel-n" description:"number of concurrent requests for parallel test mode" default:"4"`
}

// Helper functions for pointer creation
func Float64Ptr(f float64) *float64 { return &f }
func IntPtr(i int) *int             { return &i }

var testPrompts = []string{
	"<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
	"<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
	"<|im_start|>user\nName a color of the sky.<|im_end|>\n<|im_start|>assistant\n",
	"<|im_start|>user\nWhat is the largest planet in our solar system?<|im_end|>\n<|im_start|>assistant\n",
	"<|im_start|>user\nWhat year did World War II end?<|im_end|>\n<|im_start|>assistant\n",
	"<|im_start|>user\nHow many continents are there?<|im_end|>\n<|im_start|>assistant\n",
	"<|im_start|>user\nWhat is the chemical symbol for water?<|im_end|>\n<|im_start|>assistant\n",
	"<|im_start|>user\nWho wrote Romeo and Juliet?<|im_end|>\n<|im_start|>assistant\n",
}

func main() {
	var opts flagOptions
	var argv []string = os.Args[1:]
	var parser = flags.NewParser(&opts, flags.HelpFlag)
	var err error
	_, err = parser.ParseArgs(argv)
	if err != nil {
		fmt.Printf("Command line flags parsing failed: %v", err)
		os.Exit(1)
	}

	logger := logging.NewSprintfLogger()

	logger.Infof("opts: %+v", opts)

	if opts.Transport != "grpc" && opts.Transport != "http" {
		fmt.Printf("Invalid transport %q: must be 'grpc' or 'http'\n", opts.Transport)
		os.Exit(1)
	}

	if opts.ServerPath == "" && opts.AttachPort == 0 {
		fmt.Println("Server path (--server) or attach port (--port) is required")
		os.Exit(1)
	}

	if opts.ModelPath == "" {
		fmt.Println("Model path is required")
		os.Exit(1)
	}

	// Convert model path to absolute path (needed when server runs in different working directory)
	modelPath := opts.ModelPath
	if !filepath.IsAbs(modelPath) {
		absPath, err := filepath.Abs(modelPath)
		if err != nil {
			fmt.Printf("Failed to get absolute path for model: %v\n", err)
			os.Exit(1)
		}
		modelPath = absPath
	}

	// Backpressure test mode manages its own server lifecycle
	if opts.TestMode == "backpressure" {
		if opts.ServerPath == "" {
			fmt.Println("backpressure test mode requires --server (server binary path)")
			os.Exit(1)
		}
		runBackpressureTest(modelPath, opts, logger)
		logger.Infof("Done")
		return
	}

	logger.Infof("Starting LLM service...")

	llmServiceOptions := llmservice.LLMServiceOptions{
		ServerPath: opts.ServerPath,
		AttachHost: opts.AttachHost,
		AttachPort: opts.AttachPort,
		Transport:  opts.Transport,
		NParallel:  opts.ParallelN,
	}

	llmService, err := llmservice.NewLlamacppLLMService(llmServiceOptions, logger)
	if err != nil {
		logger.Errorf("Failed to create LLM service: %v", err)
		os.Exit(1)
	}

	shutdown := func() {
		logger.Infof("Stopping LLM service...")
		llmService.Shutdown()
		logger.Infof("LLM service stopped")
	}

	defer shutdown()

	ctx := context.Background()

	logger.Infof("Pinging LLM server...")

	retryAttempts := 10
	retryInterval := 1 * time.Second
	for retryAttempts > 0 {
		err = llmService.Ping(ctx)
		if err == nil {
			break
		}

		refusedError1 := "connectex: No connection could be made because the target machine actively refused it."
		refusedError2 := "connect: connection refused"
		if !strings.Contains(err.Error(), refusedError1) && !strings.Contains(err.Error(), refusedError2) {
			break
		}

		logger.Infof("LLM server is not running, retrying in %s", retryInterval)

		time.Sleep(retryInterval)

		retryInterval = retryInterval * 2
		retryAttempts--

		logger.Infof("Retrying Ping...")
	}

	if err != nil {
		logger.Errorf("Failed to ping LLM server: %v", err)
		os.Exit(1)
	}

	{
		logger.Infof("Loading model...")

		progressChan := make(chan float32)
		defer close(progressChan)

		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			defer wg.Done()

			for progress := range progressChan {
				fmt.Printf("Model progress: %f\n", progress)

				if progress >= 1.0 {
					break
				}
			}
		}()

		logger.Infof("Loading model from '%s'...", modelPath)

		err = llmService.LoadModel(ctx, modelPath, progressChan)
		if err != nil {
			logger.Errorf("Failed to load model: %v", err)
			os.Exit(1)
		}

		wg.Wait()

		logger.Infof("Model loaded")
	}

	if opts.TestMode == "parallel" {
		runParallelTest(ctx, llmService, modelPath, opts, logger)
	} else {
		runSingleTest(ctx, llmService, modelPath, opts, logger)
	}

	logger.Infof("Done")
}

func runSingleTest(ctx context.Context, llmService llmservice.LLMService, modelPath string, opts flagOptions, logger logging.SprintfLogger) {
	prompt := "<|im_start|>user\nWhat is the capital of USA?<|im_end|>\n<|im_start|>assistant\n"

	var predictRequest llmservice.PredictRequest

	switch opts.TestMode {
	case "greedy":
		logger.Infof("Running GREEDY test mode (deterministic)")
		predictRequest = llmservice.PredictRequest{
			ModelName:         modelPath,
			Message:           prompt,
			MaxTokens:         opts.MaxTokens,
			Temperature:       0.0,
			Stream:            true,
			TopP:              Float64Ptr(1.0),
			TopK:              IntPtr(0),
			RepetitionPenalty: Float64Ptr(opts.RepeatPenalty),
		}
		if opts.MinP != 0.05 {
			predictRequest.MinP = Float64Ptr(opts.MinP)
		}
		if opts.RandomSeed >= 0 {
			predictRequest.RandomSeed = IntPtr(opts.RandomSeed)
		}

	case "seeded":
		logger.Infof("Running SEEDED test mode (reproducible randomized)")
		predictRequest = llmservice.PredictRequest{
			ModelName:         modelPath,
			Message:           prompt,
			MaxTokens:         opts.MaxTokens,
			Temperature:       opts.Temperature,
			Stream:            true,
			TopP:              Float64Ptr(opts.TopP),
			TopK:              IntPtr(opts.TopK),
			MinP:              Float64Ptr(opts.MinP),
			RepetitionPenalty: Float64Ptr(opts.RepeatPenalty),
			RandomSeed:        IntPtr(12345),
		}

	case "stress":
		logger.Infof("Running STRESS test mode (performance testing)")
		predictRequest = llmservice.PredictRequest{
			ModelName:         modelPath,
			Message:           "Write a detailed explanation of machine learning concepts, including supervised learning, unsupervised learning, and neural networks. Include examples and applications.",
			MaxTokens:         500,
			Temperature:       opts.Temperature,
			Stream:            true,
			TopP:              Float64Ptr(opts.TopP),
			TopK:              IntPtr(opts.TopK),
			MinP:              Float64Ptr(opts.MinP),
			RepetitionPenalty: Float64Ptr(opts.RepeatPenalty),
		}
		if opts.RandomSeed >= 0 {
			predictRequest.RandomSeed = IntPtr(opts.RandomSeed)
		}

	default: // "baseline"
		logger.Infof("Running BASELINE test mode (configurable parameters)")
		predictRequest = llmservice.PredictRequest{
			ModelName:         modelPath,
			Message:           prompt,
			MaxTokens:         opts.MaxTokens,
			Temperature:       opts.Temperature,
			Stream:            true,
			TopP:              Float64Ptr(opts.TopP),
			TopK:              IntPtr(opts.TopK),
			MinP:              Float64Ptr(opts.MinP),
			RepetitionPenalty: Float64Ptr(opts.RepeatPenalty),
		}
		if opts.RandomSeed >= 0 {
			predictRequest.RandomSeed = IntPtr(opts.RandomSeed)
		}
	}

	logger.Infof("=== TEST CONFIGURATION ===")
	logger.Infof("Test Mode: %s", opts.TestMode)
	logger.Infof("Model: %s", modelPath)
	logger.Infof("Prompt: %s", prompt)
	logger.Infof("Parameters:")
	logger.Infof("  - temperature: %.3f", predictRequest.Temperature)
	logger.Infof("  - max_tokens: %d", predictRequest.MaxTokens)
	if predictRequest.TopP != nil {
		logger.Infof("  - top_p: %.3f", *predictRequest.TopP)
	}
	if predictRequest.TopK != nil {
		logger.Infof("  - top_k: %d", *predictRequest.TopK)
	}
	if predictRequest.MinP != nil {
		logger.Infof("  - min_p: %.3f", *predictRequest.MinP)
	}
	if predictRequest.RepetitionPenalty != nil {
		logger.Infof("  - repetition_penalty: %.3f", *predictRequest.RepetitionPenalty)
	}
	if predictRequest.RandomSeed != nil {
		logger.Infof("  - random_seed: %d", *predictRequest.RandomSeed)
	} else {
		logger.Infof("  - random_seed: random")
	}
	logger.Infof("==========================")

	logger.Infof("Predicting...")

	wg := sync.WaitGroup{}
	wg.Add(1)

	predictResponseChan := make(chan llmservice.PredictResponse)
	defer close(predictResponseChan)

	startTime := time.Now()

	fullResponse := ""
	var tokenCount int

	go func() {
		defer wg.Done()

		for predictResponse := range predictResponseChan {
			fmt.Printf("Predict response: %+v\n", predictResponse)

			fullResponse += predictResponse.Message
			if predictResponse.Tokens > 0 {
				tokenCount = int(predictResponse.Tokens)
			}

			if predictResponse.Done {
				break
			}
		}
	}()

	err := llmService.Predict(ctx, predictRequest, predictResponseChan)
	if err != nil {
		logger.Errorf("Failed to predict: %v", err)
		os.Exit(1)
	}

	wg.Wait()

	generationTime := time.Since(startTime)
	throughput := float64(tokenCount) / generationTime.Seconds()

	logger.Infof("=== PERFORMANCE RESULTS ===")
	logger.Infof("Total tokens: %d", tokenCount)
	logger.Infof("Generation time: %.2fs", generationTime.Seconds())
	logger.Infof("Throughput: %.2f tokens/second", throughput)
	logger.Infof("Full response: %s", fullResponse)
	logger.Infof("================================")

	logger.Infof("Predict done")
}

type parallelResult struct {
	index    int
	prompt   string
	response string
	tokens   int
	err      error
	duration time.Duration
}

func runParallelTest(ctx context.Context, llmService llmservice.LLMService, modelPath string, opts flagOptions, logger logging.SprintfLogger) {
	nParallel := opts.ParallelN
	if nParallel < 2 {
		nParallel = 2
	}

	prompts := testPrompts

	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 50
	}

	logger.Infof("=== PARALLEL TEST CONFIGURATION ===")
	logger.Infof("Concurrent requests: %d", nParallel)
	logger.Infof("Max tokens per request: %d", maxTokens)
	logger.Infof("Model: %s", modelPath)
	logger.Infof("===================================")

	results := make([]parallelResult, nParallel)
	var wg sync.WaitGroup

	logger.Infof("Launching %d concurrent prediction requests...", nParallel)
	startTime := time.Now()

	for i := 0; i < nParallel; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			prompt := prompts[idx%len(prompts)]
			reqStart := time.Now()

			req := llmservice.PredictRequest{
				ModelName:   modelPath,
				Message:     prompt,
				MaxTokens:   maxTokens,
				Temperature: 0.0,
				Stream:      true,
			}

			respChan := make(chan llmservice.PredictResponse, 128)

			err := llmService.Predict(ctx, req, respChan)
			if err != nil {
				results[idx] = parallelResult{index: idx, prompt: prompt, err: err, duration: time.Since(reqStart)}
				return
			}

			var fullResponse string
			var tokenCount int

			for resp := range respChan {
				if resp.Error != nil {
					results[idx] = parallelResult{index: idx, prompt: prompt, err: resp.Error, duration: time.Since(reqStart)}
					return
				}
				fullResponse += resp.Message
				if resp.Tokens > 0 {
					tokenCount = int(resp.Tokens)
				}
				if resp.Done {
					break
				}
			}

			results[idx] = parallelResult{
				index:    idx,
				prompt:   prompt,
				response: fullResponse,
				tokens:   tokenCount,
				duration: time.Since(reqStart),
			}
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	logger.Infof("=== PARALLEL TEST RESULTS ===")
	allPassed := true
	totalTokens := 0
	for _, r := range results {
		if r.err != nil {
			logger.Errorf("  Request %d: FAILED after %.2fs - %v", r.index, r.duration.Seconds(), r.err)
			allPassed = false
		} else {
			throughput := float64(0)
			if r.duration.Seconds() > 0 {
				throughput = float64(r.tokens) / r.duration.Seconds()
			}
			logger.Infof("  Request %d: OK - %d tokens in %.2fs (%.2f t/s)",
				r.index, r.tokens, r.duration.Seconds(), throughput)
			logger.Infof("    Prompt:   %q", r.prompt[:min(60, len(r.prompt))]+"...")
			logger.Infof("    Response: %q", r.response[:min(80, len(r.response))])
			totalTokens += r.tokens
		}
	}

	logger.Infof("")
	logger.Infof("Total wall-clock time: %.2fs", totalTime.Seconds())
	logger.Infof("Total tokens generated: %d", totalTokens)
	if totalTime.Seconds() > 0 {
		logger.Infof("Aggregate throughput: %.2f tokens/second", float64(totalTokens)/totalTime.Seconds())
	}

	if allPassed {
		logger.Infof("RESULT: ALL %d PARALLEL REQUESTS COMPLETED SUCCESSFULLY", nParallel)
	} else {
		logger.Errorf("RESULT: SOME PARALLEL REQUESTS FAILED")
		os.Exit(1)
	}
}

// =============================================================================
// Helpers for compare / backpressure test modes
// =============================================================================

func pingWithRetry(ctx context.Context, svc llmservice.LLMService, logger logging.SprintfLogger) error {
	retryAttempts := 10
	retryInterval := 1 * time.Second
	var err error
	for retryAttempts > 0 {
		err = svc.Ping(ctx)
		if err == nil {
			return nil
		}
		refusedError1 := "connectex: No connection could be made because the target machine actively refused it."
		refusedError2 := "connect: connection refused"
		if !strings.Contains(err.Error(), refusedError1) && !strings.Contains(err.Error(), refusedError2) {
			return err
		}
		logger.Infof("Server not ready, retrying in %s", retryInterval)
		time.Sleep(retryInterval)
		retryInterval *= 2
		retryAttempts--
	}
	return err
}

func loadModelHelper(ctx context.Context, svc llmservice.LLMService, modelPath string, logger logging.SprintfLogger) error {
	progressChan := make(chan float32)
	defer close(progressChan)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for progress := range progressChan {
			fmt.Printf("Model progress: %f\n", progress)
			if progress >= 1.0 {
				break
			}
		}
	}()
	logger.Infof("Loading model from '%s'...", modelPath)
	err := svc.LoadModel(ctx, modelPath, progressChan)
	wg.Wait()
	return err
}

func createAndPrepareService(ctx context.Context, opts flagOptions, modelPath string, nParallel int, logger logging.SprintfLogger) (llmservice.LLMService, error) {
	svcOpts := llmservice.LLMServiceOptions{
		ServerPath: opts.ServerPath,
		AttachHost: opts.AttachHost,
		Transport:  opts.Transport,
		NParallel:  nParallel,
	}
	svc, err := llmservice.NewLlamacppLLMService(svcOpts, logger)
	if err != nil {
		return nil, fmt.Errorf("create service: %w", err)
	}
	if err := pingWithRetry(ctx, svc, logger); err != nil {
		svc.Shutdown()
		return nil, fmt.Errorf("ping: %w", err)
	}
	if err := loadModelHelper(ctx, svc, modelPath, logger); err != nil {
		svc.Shutdown()
		return nil, fmt.Errorf("load model: %w", err)
	}
	return svc, nil
}

func runGreedyRequest(ctx context.Context, svc llmservice.LLMService, modelPath string, prompt string, maxTokens int) parallelResult {
	start := time.Now()
	req := llmservice.PredictRequest{
		ModelName:   modelPath,
		Message:     prompt,
		MaxTokens:   maxTokens,
		Temperature: 0.0,
		Stream:      true,
	}
	respChan := make(chan llmservice.PredictResponse, 128)
	err := svc.Predict(ctx, req, respChan)
	if err != nil {
		return parallelResult{prompt: prompt, err: err, duration: time.Since(start)}
	}
	var fullResponse string
	var tokenCount int
	for resp := range respChan {
		if resp.Error != nil {
			return parallelResult{prompt: prompt, err: resp.Error, duration: time.Since(start)}
		}
		fullResponse += resp.Message
		if resp.Tokens > 0 {
			tokenCount = int(resp.Tokens)
		}
		if resp.Done {
			break
		}
	}
	return parallelResult{
		prompt:   prompt,
		response: fullResponse,
		tokens:   tokenCount,
		duration: time.Since(start),
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}

// =============================================================================
// Backpressure test: 2N requests with N slots, all must complete
// =============================================================================

func runBackpressureTest(modelPath string, opts flagOptions, logger logging.SprintfLogger) {
	ctx := context.Background()
	nSlots := opts.ParallelN
	if nSlots < 2 {
		nSlots = 2
	}
	nRequests := 2 * nSlots
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 50
	}

	logger.Infof("=== BACKPRESSURE TEST CONFIGURATION ===")
	logger.Infof("Server slots (n_parallel): %d", nSlots)
	logger.Infof("Concurrent requests: %d (2x slots)", nRequests)
	logger.Infof("Max tokens per request: %d", maxTokens)
	logger.Infof("=======================================")

	svc, err := createAndPrepareService(ctx, opts, modelPath, nSlots, logger)
	if err != nil {
		logger.Errorf("Failed to create service: %v", err)
		os.Exit(1)
	}
	defer func() {
		logger.Infof("Shutting down server...")
		svc.Shutdown()
	}()

	prompts := make([]string, nRequests)
	for i := range prompts {
		prompts[i] = testPrompts[i%len(testPrompts)]
	}

	results := make([]parallelResult, nRequests)
	var wg sync.WaitGroup
	startTime := time.Now()

	logger.Infof("Launching %d concurrent requests against %d slots...", nRequests, nSlots)
	for i := 0; i < nRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = runGreedyRequest(ctx, svc, modelPath, prompts[idx], maxTokens)
			results[idx].index = idx
		}(i)
	}
	wg.Wait()
	totalTime := time.Since(startTime)

	logger.Infof("=== BACKPRESSURE TEST RESULTS ===")
	allPassed := true
	totalTokens := 0
	for _, r := range results {
		if r.err != nil {
			logger.Errorf("  Request %d: FAILED - %v", r.index, r.err)
			allPassed = false
		} else {
			throughput := float64(0)
			if r.duration.Seconds() > 0 {
				throughput = float64(r.tokens) / r.duration.Seconds()
			}
			logger.Infof("  Request %d: OK - %d tokens in %.2fs (%.2f t/s)",
				r.index, r.tokens, r.duration.Seconds(), throughput)
			logger.Infof("    Response: %q", truncate(r.response, 80))
			totalTokens += r.tokens
		}
	}

	logger.Infof("")
	logger.Infof("Total wall-clock time: %.2fs", totalTime.Seconds())
	logger.Infof("Total tokens generated: %d", totalTokens)
	if totalTime.Seconds() > 0 {
		logger.Infof("Aggregate throughput: %.2f tokens/second", float64(totalTokens)/totalTime.Seconds())
	}

	if allPassed {
		logger.Infof("RESULT: ALL %d REQUESTS COMPLETED SUCCESSFULLY (%d slots, %dx backpressure)",
			nRequests, nSlots, nRequests/nSlots)
	} else {
		logger.Errorf("RESULT: SOME REQUESTS FAILED")
		os.Exit(1)
	}
}
