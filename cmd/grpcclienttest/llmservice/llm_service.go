package llmservice

import (
	"context"
	"fmt"
	"io"
	"sync"

	"github.com/hypernetix/llamacpp_server/internal/logging"
	"github.com/hypernetix/llamacpp_server/internal/proto"

	"github.com/phayes/freeport"
	"google.golang.org/grpc"
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
	ServerPath string
	AttachPort int
}

func NewLlamacppLLMService(options LLMServiceOptions, logger logging.SprintfLogger) (LLMService, error) {
	initialLogger := logger
	logger = initialLogger.With("module", "LLMService")

	var serverProcess Process

	port := options.AttachPort
	if port == 0 {
		logger.Debugf("Allocating free port")

		var err error
		port, err = freeport.GetFreePort()
		if err != nil {
			return nil, fmt.Errorf("failed allocating free ports: %v", err)
		}

		logger.Debugf("Allocated free port: %d", port)

		serverPath := options.ServerPath
		args := []string{"--port", fmt.Sprintf("%d", port)}

		logger.Debugf("Running server process '%s' with args: %v", serverPath, args)

		serverProcess, err = NewProcess(initialLogger, serverPath, args)
		if err != nil {
			return nil, fmt.Errorf("failed to create server process: %v", err)
		}
	}

	address := fmt.Sprintf("127.0.0.1:%d", port)

	logger.Debugf("Dialing server process at %s", address)

	dialOpts := []grpc.DialOption{
		grpc.WithInsecure(),
		grpc.WithReadBufferSize(1 * 1024 * 1024),
		grpc.WithInitialWindowSize(1 * 1024 * 1024),
		grpc.WithInitialConnWindowSize(1 * 1024 * 1024),
	}

	grpcServerConnection, err := grpc.NewClient(address, dialOpts...)
	if err != nil {
		if serverProcess != nil {
			serverProcess.Stop()
		}
		return nil, err
	}

	grcpClient := proto.NewLLMServerClient(grpcServerConnection)

	logger.Debugf("Connected to server process at %s", address)

	llmService := &llamaService{
		serverProcess:        serverProcess,
		grpcServerConnection: grpcServerConnection,
		grcpClient:           grcpClient,
		logger:               logger,
	}

	return llmService, nil
}

type llamaService struct {
	once                 sync.Once
	serverProcess        Process
	grcpClient           proto.LLMServerClient
	grpcServerConnection *grpc.ClientConn
	logger               logging.SprintfLogger
}

func (s *llamaService) Shutdown() {
	s.once.Do(func() {
		if s.grpcServerConnection != nil {
			s.grpcServerConnection.Close()
			s.grpcServerConnection = nil
		}
		if s.serverProcess != nil {
			s.serverProcess.Stop()
			s.serverProcess = nil
		}
	})
}

func (s *llamaService) Ping(ctx context.Context) error {
	_, err := s.grcpClient.Ping(ctx, &proto.PingRequest{})
	if err != nil {
		s.logger.Errorf("Ping: gRPC client call failed: %v", err)
		return err
	}
	s.logger.Debugf("Ping done")
	return nil
}

func (s *llamaService) LoadModel(ctx context.Context, name string, progress chan<- float32) error {
	s.logger.Errorf("LoadModel, name: %v", name)

	request := proto.LoadModelRequest{Path: name}
	messages, err := s.grcpClient.LoadModel(ctx, &request)
	if err != nil {
		s.logger.Errorf("LoadModel: gRPC client call failed: %v", err)
		return err
	}

	for {
		msg, err := messages.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			s.logger.Errorf("LoadModel: gRPC client stream recv failed: %v", err)
			return err
		}
		if progress != nil {
			progress <- msg.Progress
		}
	}

	s.logger.Errorf("LoadModel, model loaded: %v", name)
	return nil
}

func (s *llamaService) Predict(ctx context.Context, req PredictRequest, resp chan<- PredictResponse) error {
	s.logger.Errorf("ChatCompletionsStreamRaw, req: %+v", req)

	if !req.Stream {
		return fmt.Errorf("stream is not enabled")
	}

	modelName := req.ModelName
	message := req.Message
	mxTokens := req.MaxTokens
	temperature := req.Temperature

	var topP float64
	if req.TopP == nil {
		topP = 1.0
	}

	topK := int32(0)
	if req.TopK == nil {
		topK = 0
	} else {
		topK = int32(*req.TopK)
	}

	request := proto.PredictRequest{
		Model:       modelName,
		Prompt:      message,
		Stream:      true,
		MaxTokens:   int32(mxTokens),
		Temperature: float32(temperature),
		TopP:        float32(topP),
		TopK:        topK,
	}

	// Add optional parameters if present
	if req.MinP != nil {
		minP := float32(*req.MinP)
		request.Options = &proto.PredictRequest_Options{
			MinP: &minP,
		}
	}

	if req.RepetitionPenalty != nil {
		repPenalty := float32(*req.RepetitionPenalty)
		if request.Options == nil {
			request.Options = &proto.PredictRequest_Options{}
		}
		request.Options.RepetitionPenalty = &repPenalty
	}

	if req.RandomSeed != nil {
		seed := int32(*req.RandomSeed)
		if request.Options == nil {
			request.Options = &proto.PredictRequest_Options{}
		}
		request.Options.RandomSeed = &seed
	}

	s.logger.Infof("=== FINAL REQUEST PARAMETERS ===")
	s.logger.Infof("Model: %s", request.Model)
	s.logger.Infof("Core parameters:")
	s.logger.Infof("  - temperature: %.3f", request.Temperature)
	s.logger.Infof("  - top_p: %.3f", request.TopP)
	s.logger.Infof("  - top_k: %d", request.TopK)
	s.logger.Infof("  - max_tokens: %d", request.MaxTokens)
	if request.Options != nil {
		s.logger.Infof("Advanced parameters:")
		if request.Options.MinP != nil {
			s.logger.Infof("  - min_p: %.3f", *request.Options.MinP)
		}
		if request.Options.RepetitionPenalty != nil {
			s.logger.Infof("  - repetition_penalty: %.3f", *request.Options.RepetitionPenalty)
		}
		if request.Options.RandomSeed != nil {
			s.logger.Infof("  - random_seed: %d", *request.Options.RandomSeed)
		}
	} else {
		s.logger.Infof("Advanced parameters: None")
	}
	s.logger.Infof("================================")

	predictStream, err := s.grcpClient.Predict(ctx, &request)
	if err != nil {
		s.logger.Errorf("Predict: gRPC client call failed: %v", err)
		return err
	}

	s.logger.Errorf("Starting predict stream: %+v", req)

	// Use context to handle client disconnects
	streamCtx, cancelStream := context.WithCancel(ctx)

	go func() {
		defer cancelStream()

		for {
			select {
			case <-streamCtx.Done():
				s.logger.Debugf("Stream context canceled, stopping prediction stream")
				resp <- PredictResponse{
					Error: streamCtx.Err(),
					Done:  true,
				}
				return

			default:
				predictResponse, err := predictStream.Recv()
				if err == io.EOF {
					// Stream completed successfully
					resp <- PredictResponse{
						Error: nil,
						Done:  true,
					}
					return
				}
				if err != nil {
					s.logger.Errorf("Predict: gRPC client stream Recv failed: %v", err)
					resp <- PredictResponse{
						Error: err,
						Done:  true,
					}
					return
				}

				resp <- PredictResponse{
					Message: string(predictResponse.Message),
					Token:   predictResponse.Token,
					Tokens:  predictResponse.Tokens,
					Error:   nil,
					Done:    false,
				}
			}
		}
	}()

	return nil
}
