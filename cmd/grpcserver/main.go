package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/hypernetix/llamacpp_server/api/proto"
	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
	llamacppgrpcserver "github.com/hypernetix/llamacpp_server/internal/grpcserver"
	"github.com/hypernetix/llamacpp_server/internal/logging"

	flags "github.com/jessevdk/go-flags"
	"google.golang.org/grpc"
)

type flagOptions struct {
	Host         string `long:"host" default:"127.0.0.1" description:"host address to bind (use 0.0.0.0 for Docker)"`
	Port         string `long:"port" default:"50051" description:"port to listen for gRPC server"`
	NGpuLayers   int    `long:"ngpu" default:"99" description:"number of GPU layers"`
	UseMmap      bool   `long:"mmap" description:"use mmap"`
	SplitMode    string `long:"split-mode" default:"layer" description:"how to split model across GPUs: none, layer, row (row=tensor parallelism)"`
	MainGpu      int    `long:"main-gpu" default:"0" description:"main GPU index when split-mode=none"`
	TensorSplit  string `long:"tensor-split" default:"" description:"GPU split proportions, comma-separated (e.g. '0.5,0.5' for even 2-GPU split)"`
	FlashAttn    bool   `long:"flash-attn" description:"enable flash attention for faster inference"`
	NParallel    int    `long:"n-parallel" default:"0" description:"max concurrent inference requests (0=unlimited)"`
	Threads      int    `long:"threads" default:"0" description:"number of threads for generation (0=auto-detect)"`
	ThreadsBatch int    `long:"threads-batch" default:"0" description:"number of threads for batch/prompt processing (0=auto-detect)"`
	CtxSize      int    `long:"ctx-size" default:"4096" description:"context window size per inference slot"`
	BatchSize    int    `long:"batch-size" default:"2048" description:"batch size for prompt processing"`
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

	port, err := strconv.Atoi(opts.Port)
	if err != nil {
		fmt.Printf("Command line flags parsing failed: %v", err)
		os.Exit(1)
	}

	listener, err := net.Listen("tcp", fmt.Sprintf("%s:%d", opts.Host, port))
	if err != nil {
		fmt.Printf("Failed to listen at %s:%d: %v", opts.Host, port, err)
		os.Exit(1)
	}

	logger := logging.NewSprintfLogger()

	logger.Infof("Listening at %s", listener.Addr().String())

	logger.Infof("Initializing llama.cpp...")

	llamacppbindings.Initialize(logger.With("module", "llama.cpp"))

	logger.Infof("Starting LLM gRPC server...")

	grpcServer := grpc.NewServer(
		grpc.WriteBufferSize(1*1024*1024),
		grpc.InitialWindowSize(1*1024*1024),
		grpc.InitialConnWindowSize(1*1024*1024),
	)

	splitMode := llamacppbindings.SplitModeLayer
	switch strings.ToLower(opts.SplitMode) {
	case "none":
		splitMode = llamacppbindings.SplitModeNone
	case "layer":
		splitMode = llamacppbindings.SplitModeLayer
	case "row":
		splitMode = llamacppbindings.SplitModeRow
	default:
		logger.Errorf("Unknown split-mode %q, using 'layer'", opts.SplitMode)
	}

	var tensorSplit []float32
	if opts.TensorSplit != "" {
		for _, s := range strings.Split(opts.TensorSplit, ",") {
			s = strings.TrimSpace(s)
			v, err := strconv.ParseFloat(s, 32)
			if err != nil {
				fmt.Printf("Invalid tensor-split value %q: %v\n", s, err)
				os.Exit(1)
			}
			tensorSplit = append(tensorSplit, float32(v))
		}
	}

	llmServerOptions := llamacppgrpcserver.GlobalOptions{
		Model: llamacppgrpcserver.LoadModelOptions{
			NGpuLayers:  opts.NGpuLayers,
			UseMmap:     opts.UseMmap,
			SplitMode:   splitMode,
			MainGpu:     opts.MainGpu,
			TensorSplit: tensorSplit,
		},
		Predict: llamacppgrpcserver.PredictOptions{
			FlashAttn:     opts.FlashAttn,
			NParallel:     opts.NParallel,
			NThreads:      opts.Threads,
			NThreadsBatch: opts.ThreadsBatch,
			CtxSize:       opts.CtxSize,
			BatchSize:     opts.BatchSize,
		},
	}

	logger.Infof("Split mode: %s", opts.SplitMode)
	if len(tensorSplit) > 0 {
		logger.Infof("Tensor split: %v", tensorSplit)
	}
	if opts.NParallel > 0 {
		logger.Infof("Max concurrent predictions: %d", opts.NParallel)
	} else {
		logger.Infof("Max concurrent predictions: unlimited")
	}
	if opts.FlashAttn {
		logger.Infof("Flash attention: enabled")
	}
	llmServer := llamacppgrpcserver.NewLLMServer(llmServerOptions, logger)

	proto.RegisterLLMServerServer(grpcServer, llmServer)
	go func() {
		err := grpcServer.Serve(listener)
		if err != nil {
			logger.Errorf("LLM gRPC server Serve failed: %v", err)
			return
		}
	}()

	sig := make(chan os.Signal, 1)
	if runtime.GOOS == "windows" {
		signal.Notify(sig) // Unix signals not implemented on Windows
	} else {
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	}

	receivedSignal := <-sig

	if runtime.GOOS == "windows" {
		if receivedSignal != os.Interrupt {
			logger.Errorf("Wrong signal received: got %q, want %q\n", receivedSignal, os.Interrupt)
			os.Exit(42)
		}
	}

	logger.Infof("Received signal: %v", receivedSignal)

	logger.Infof("Stopping...")

	// Create a timeout context for graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	// Stop components in reverse order of creation
	logger.Infof("Stopping LLM server...")
	llmServer.Stop()
	logger.Infof("LLM server stopped")

	logger.Infof("Stopping gRPC server...")
	// Use GracefulStop instead of Stop for graceful shutdown
	stopped := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(stopped)
	}()

	// Wait for graceful shutdown or timeout
	select {
	case <-stopped:
		logger.Infof("gRPC server stopped gracefully")
	case <-ctx.Done():
		logger.Infof("Shutdown timed out, forcing stop")
		grpcServer.Stop()
	}

	logger.Infof("Stopped")
}
