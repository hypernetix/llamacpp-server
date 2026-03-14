package llmservice

import (
	"context"
	"fmt"
	"io"
	"sync"

	"github.com/hypernetix/llamacpp_server/api/proto"
	"github.com/hypernetix/llamacpp_server/internal/logging"

	"google.golang.org/grpc"
)

type grpcClient struct {
	once          sync.Once
	serverProcess Process
	conn          *grpc.ClientConn
	client        proto.LLMServerClient
	logger        logging.SprintfLogger
}

func newGRPCClient(host string, port int, serverProcess Process, logger logging.SprintfLogger) (LLMService, error) {
	address := fmt.Sprintf("%s:%d", host, port)
	logger.Debugf("Dialing gRPC server at %s", address)

	dialOpts := []grpc.DialOption{
		grpc.WithInsecure(),
		grpc.WithReadBufferSize(1 * 1024 * 1024),
		grpc.WithInitialWindowSize(1 * 1024 * 1024),
		grpc.WithInitialConnWindowSize(1 * 1024 * 1024),
	}

	conn, err := grpc.NewClient(address, dialOpts...)
	if err != nil {
		if serverProcess != nil {
			serverProcess.Stop()
		}
		return nil, err
	}

	logger.Debugf("Connected to gRPC server at %s", address)

	return &grpcClient{
		serverProcess: serverProcess,
		conn:          conn,
		client:        proto.NewLLMServerClient(conn),
		logger:        logger,
	}, nil
}

func (c *grpcClient) Shutdown() {
	c.once.Do(func() {
		if c.conn != nil {
			c.conn.Close()
			c.conn = nil
		}
		if c.serverProcess != nil {
			c.serverProcess.Stop()
			c.serverProcess = nil
		}
	})
}

func (c *grpcClient) Ping(ctx context.Context) error {
	_, err := c.client.Ping(ctx, &proto.PingRequest{})
	if err != nil {
		c.logger.Errorf("Ping: gRPC call failed: %v", err)
		return err
	}
	c.logger.Debugf("Ping done")
	return nil
}

func (c *grpcClient) LoadModel(ctx context.Context, name string, progress chan<- float32) error {
	c.logger.Infof("LoadModel: %s", name)

	stream, err := c.client.LoadModel(ctx, &proto.LoadModelRequest{Path: name})
	if err != nil {
		c.logger.Errorf("LoadModel: gRPC call failed: %v", err)
		return err
	}

	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			c.logger.Errorf("LoadModel: stream recv failed: %v", err)
			return err
		}
		if progress != nil {
			progress <- msg.Progress
		}
	}

	c.logger.Infof("LoadModel: loaded %s", name)
	return nil
}

func (c *grpcClient) Predict(ctx context.Context, req PredictRequest, resp chan<- PredictResponse) error {
	c.logger.Infof("Predict: model=%s, max_tokens=%d, temp=%.3f",
		req.ModelName, req.MaxTokens, req.Temperature)

	if !req.Stream {
		return fmt.Errorf("non-streaming predict not supported in test client")
	}

	protoReq := buildProtoRequest(req)

	predictStream, err := c.client.Predict(ctx, protoReq)
	if err != nil {
		c.logger.Errorf("Predict: gRPC call failed: %v", err)
		return err
	}

	streamCtx, cancelStream := context.WithCancel(ctx)

	go func() {
		defer cancelStream()

		for {
			select {
			case <-streamCtx.Done():
				resp <- PredictResponse{Error: streamCtx.Err(), Done: true}
				return
			default:
				msg, err := predictStream.Recv()
				if err == io.EOF {
					resp <- PredictResponse{Done: true}
					return
				}
				if err != nil {
					c.logger.Errorf("Predict: stream recv failed: %v", err)
					resp <- PredictResponse{Error: err, Done: true}
					return
				}
				resp <- PredictResponse{
					Message: string(msg.Message),
					Token:   msg.Token,
					Tokens:  msg.Tokens,
				}
			}
		}
	}()

	return nil
}

func buildProtoRequest(req PredictRequest) *proto.PredictRequest {
	topP := float32(1.0)
	if req.TopP != nil {
		topP = float32(*req.TopP)
	}
	topK := int32(0)
	if req.TopK != nil {
		topK = int32(*req.TopK)
	}

	protoReq := &proto.PredictRequest{
		Model:       req.ModelName,
		Prompt:      req.Message,
		Stream:      true,
		MaxTokens:   int32(req.MaxTokens),
		Temperature: float32(req.Temperature),
		TopP:        topP,
		TopK:        topK,
	}

	if req.MinP != nil {
		minP := float32(*req.MinP)
		if protoReq.Options == nil {
			protoReq.Options = &proto.PredictRequest_Options{}
		}
		protoReq.Options.MinP = &minP
	}
	if req.RepetitionPenalty != nil {
		rp := float32(*req.RepetitionPenalty)
		if protoReq.Options == nil {
			protoReq.Options = &proto.PredictRequest_Options{}
		}
		protoReq.Options.RepetitionPenalty = &rp
	}
	if req.RandomSeed != nil {
		seed := int32(*req.RandomSeed)
		if protoReq.Options == nil {
			protoReq.Options = &proto.PredictRequest_Options{}
		}
		protoReq.Options.RandomSeed = &seed
	}

	return protoReq
}
