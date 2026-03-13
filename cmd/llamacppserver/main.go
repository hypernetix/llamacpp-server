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
	"github.com/hypernetix/llamacpp_server/internal/grpcserver"
	"github.com/hypernetix/llamacpp_server/internal/httpserver"
	"github.com/hypernetix/llamacpp_server/internal/llmservice"
	"github.com/hypernetix/llamacpp_server/internal/logging"

	flags "github.com/jessevdk/go-flags"
	"google.golang.org/grpc"
)

type flagOptions struct {
	Host         string `long:"host" default:"127.0.0.1" description:"host address to bind (use 0.0.0.0 for Docker)"`
	GRPCPort     string `long:"grpc-port" default:"50052" description:"port for gRPC server (disabled if empty)"`
	HTTPPort     string `long:"http-port" default:"50051" description:"port for HTTP+SSE server (disabled if empty)"`
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

	if opts.GRPCPort == "" && opts.HTTPPort == "" {
		fmt.Printf("gRPC port or HTTP port is required")
		os.Exit(1)
	}

	logger := logging.NewSprintfLogger()

	// --- Parse model options ---

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

	// --- Initialize llama.cpp and create shared service ---

	logger.Infof("Initializing llama.cpp...")
	llamacppbindings.Initialize(logger.With("module", "llama.cpp"))

	serviceOpts := llmservice.Options{
		Model: llmservice.LoadModelOptions{
			NGpuLayers:  opts.NGpuLayers,
			UseMmap:     opts.UseMmap,
			SplitMode:   splitMode,
			MainGpu:     opts.MainGpu,
			TensorSplit: tensorSplit,
		},
		Predict: llmservice.PredictOptions{
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

	service := llmservice.NewService(serviceOpts, logger)

	// --- Start gRPC server (if configured) ---

	var grpcServer *grpc.Server
	if opts.GRPCPort != "" {
		grpcPort, err := strconv.Atoi(opts.GRPCPort)
		if err != nil {
			fmt.Printf("Invalid gRPC port: %v", err)
			os.Exit(1)
		}

		grpcListener, err := net.Listen("tcp", fmt.Sprintf("%s:%d", opts.Host, grpcPort))
		if err != nil {
			fmt.Printf("Failed to listen for gRPC at %s:%d: %v", opts.Host, grpcPort, err)
			os.Exit(1)
		}

		grpcServer = grpc.NewServer(
			grpc.WriteBufferSize(1*1024*1024),
			grpc.InitialWindowSize(1*1024*1024),
			grpc.InitialConnWindowSize(1*1024*1024),
		)

		proto.RegisterLLMServerServer(grpcServer, grpcserver.NewServer(service, logger))

		logger.Infof("gRPC server listening at %s", grpcListener.Addr().String())
		go func() {
			if err := grpcServer.Serve(grpcListener); err != nil {
				logger.Errorf("gRPC server Serve failed: %v", err)
			}
		}()
	}

	// --- Start HTTP server (if configured) ---

	var httpSrv *httpserver.Server
	if opts.HTTPPort != "" {
		httpPort, err := strconv.Atoi(opts.HTTPPort)
		if err != nil {
			fmt.Printf("Invalid HTTP port: %v\n", err)
			os.Exit(1)
		}

		httpListener, err := net.Listen("tcp", fmt.Sprintf("%s:%d", opts.Host, httpPort))
		if err != nil {
			fmt.Printf("Failed to listen for HTTP at %s:%d: %v", opts.Host, httpPort, err)
			os.Exit(1)
		}

		httpSrv = httpserver.NewServer(service, httpListener.Addr().String(), logger)
		go func() {
			if err := httpSrv.Start(httpListener); err != nil && err.Error() != "http: Server closed" {
				logger.Errorf("HTTP server failed: %v", err)
			}
		}()
	}

	// --- Wait for shutdown signal ---

	sig := make(chan os.Signal, 1)
	if runtime.GOOS == "windows" {
		signal.Notify(sig)
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

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	// Stop inference service first (drains in-flight predictions)
	logger.Infof("Stopping inference service...")
	service.Stop()
	logger.Infof("Inference service stopped")

	// Stop HTTP server
	if httpSrv != nil {
		logger.Infof("Stopping HTTP server...")
		if err := httpSrv.Shutdown(ctx); err != nil {
			logger.Errorf("HTTP server shutdown error: %v", err)
		}
		logger.Infof("HTTP server stopped")
	}

	// Stop gRPC server
	if grpcServer != nil {
		logger.Infof("Stopping gRPC server...")
		stopped := make(chan struct{})
		go func() {
			grpcServer.GracefulStop()
			close(stopped)
		}()

		select {
		case <-stopped:
			logger.Infof("gRPC server stopped gracefully")
		case <-ctx.Done():
			logger.Infof("Shutdown timed out, forcing gRPC server stop")
			grpcServer.Stop()
		}
	}

	logger.Infof("Stopped")
}
