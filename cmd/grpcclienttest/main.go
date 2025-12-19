package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	llmservice "github.com/hypernetix/llamacpp_server/cmd/grpcclienttest/llmservice"
	"github.com/hypernetix/llamacpp_server/internal/logging"

	flags "github.com/jessevdk/go-flags"
)

type flagOptions struct {
	ModelPath     string  `long:"model" description:"path to the model file"`
	ServerPath    string  `long:"server" description:"path to the server executable"`
	AttachHost    string  `long:"host" description:"host address to attach to (default: 127.0.0.1)" default:"127.0.0.1"`
	AttachPort    int     `long:"port" description:"port to attach to the server"`
	Temperature   float64 `long:"temperature" description:"sampling temperature" default:"0.7"`
	TopP          float64 `long:"top-p" description:"top-p sampling" default:"1.0"`
	TopK          int     `long:"top-k" description:"top-k sampling" default:"0"`
	RepeatPenalty float64 `long:"repeat-penalty" description:"repetition penalty" default:"1.0"`
	MinP          float64 `long:"min-p" description:"min-p sampling" default:"0.05"`
	RandomSeed    int     `long:"seed" description:"random seed for reproducible results (-1 for random)" default:"-1"`
	MaxTokens     int     `long:"max-tokens" description:"maximum tokens to generate" default:"100"`
	TestMode      string  `long:"test-mode" description:"test mode: baseline, greedy, seeded, or stress" default:"baseline"`
}

// Helper functions for pointer creation
func Float64Ptr(f float64) *float64 { return &f }
func IntPtr(i int) *int             { return &i }

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

	if opts.ServerPath == "" && opts.AttachPort == 0 {
		fmt.Println("Server path or attach port is required")
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

	logger.Infof("Starting LLM service...")

	llmServiceOptions := llmservice.LLMServiceOptions{
		ServerPath: opts.ServerPath,
		AttachHost: opts.AttachHost,
		AttachPort: opts.AttachPort,
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

	// Use SmolLM2/ChatML format for compatibility with chat models
	// This format works with SmolLM2, and other ChatML-based models
	// Raw prompt would work for base models but chat models need this format
	prompt := "<|im_start|>user\nWhat is the capital of USA?<|im_end|>\n<|im_start|>assistant\n"

	// Configure test parameters based on command line options and test mode
	var predictRequest llmservice.PredictRequest

	switch opts.TestMode {
	case "greedy":
		// Deterministic greedy sampling
		logger.Infof("Running GREEDY test mode (deterministic)")
		predictRequest = llmservice.PredictRequest{
			ModelName:         modelPath,
			Message:           prompt,
			MaxTokens:         opts.MaxTokens,
			Temperature:       0.0, // Greedy
			Stream:            true,
			TopP:              Float64Ptr(1.0),
			TopK:              IntPtr(0),
			RepetitionPenalty: Float64Ptr(opts.RepeatPenalty),
		}
		// Only add min_p if explicitly specified and not default
		if opts.MinP != 0.05 {
			predictRequest.MinP = Float64Ptr(opts.MinP)
		}
		if opts.RandomSeed >= 0 {
			predictRequest.RandomSeed = IntPtr(opts.RandomSeed)
		}

	case "seeded":
		// Randomized but reproducible sampling
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
			RandomSeed:        IntPtr(12345), // Fixed seed for reproducibility
		}

	case "stress":
		// Stress test with longer generation
		logger.Infof("Running STRESS test mode (performance testing)")
		predictRequest = llmservice.PredictRequest{
			ModelName:         modelPath,
			Message:           "Write a detailed explanation of machine learning concepts, including supervised learning, unsupervised learning, and neural networks. Include examples and applications.",
			MaxTokens:         500, // Longer generation
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
		// Use command line parameters as-is
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

	// Log the final configuration
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

	{
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

	logger.Infof("Done")
}
