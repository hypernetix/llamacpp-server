package llmservice

import (
	"context"
	"fmt"

	"github.com/hypernetix/llamacpp_server/internal/logging"

	"github.com/phayes/freeport"
)

type PredictRequest struct {
	ModelName         string
	Message           string
	Temperature       float64
	MaxTokens         int
	Stream            bool
	TopP              *float64
	TopK              *int
	MinP              *float64
	RepetitionPenalty *float64
	RandomSeed        *int
}

type PredictResponse struct {
	Message string
	Token   int32
	Tokens  int32
	Error   error
	Done    bool
}

type LLMService interface {
	Shutdown()
	Ping(ctx context.Context) error
	LoadModel(ctx context.Context, name string, progress chan<- float32) error
	Predict(ctx context.Context, req PredictRequest, resp chan<- PredictResponse) error
}

type LLMServiceOptions struct {
	ServerPath         string
	AttachHost         string
	AttachPort         int
	Transport          string // "grpc" or "http"
	ContinuousBatching bool
	NParallel          int
}

func NewLlamacppLLMService(options LLMServiceOptions, logger logging.SprintfLogger) (LLMService, error) {
	initialLogger := logger
	logger = initialLogger.With("module", "LLMService")

	host := options.AttachHost
	if host == "" {
		host = "127.0.0.1"
	}

	transport := options.Transport
	if transport == "" {
		transport = "grpc"
	}

	useHTTP := transport == "http"

	// Attach to an existing server
	if options.AttachPort > 0 {
		if useHTTP {
			logger.Infof("Attaching to existing HTTP server at %s:%d", host, options.AttachPort)
			return newHTTPClient(host, options.AttachPort, nil, logger)
		}
		logger.Infof("Attaching to existing gRPC server at %s:%d", host, options.AttachPort)
		return newGRPCClient(host, options.AttachPort, nil, logger)
	}

	// Spawn a new server process
	port, err := freeport.GetFreePort()
	if err != nil {
		return nil, fmt.Errorf("failed allocating free port: %v", err)
	}

	var portFlag string
	if useHTTP {
		portFlag = "--http-port"
	} else {
		portFlag = "--grpc-port"
	}

	args := []string{portFlag, fmt.Sprintf("%d", port)}
	if options.ContinuousBatching {
		args = append(args, "--continuous-batching")
		nPar := options.NParallel
		if nPar <= 0 {
			nPar = 4
		}
		args = append(args, "--n-parallel", fmt.Sprintf("%d", nPar))
	}
	logger.Infof("Spawning server with %s transport on port %d (continuousBatching=%v)", transport, port, options.ContinuousBatching)

	serverProcess, err := NewProcess(initialLogger, options.ServerPath, args)
	if err != nil {
		return nil, fmt.Errorf("failed to create server process: %v", err)
	}

	if useHTTP {
		return newHTTPClient(host, port, serverProcess, logger)
	}
	return newGRPCClient(host, port, serverProcess, logger)
}
