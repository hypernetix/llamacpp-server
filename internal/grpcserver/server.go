package grpcserver

import (
	"context"

	"github.com/hypernetix/llamacpp_server/api/proto"
	"github.com/hypernetix/llamacpp_server/internal/inferenceengine"
	"github.com/hypernetix/llamacpp_server/internal/llmservice"
	"github.com/hypernetix/llamacpp_server/internal/logging"
)

type Server struct {
	logger  logging.SprintfLogger
	service *llmservice.Service
	proto.UnimplementedLLMServerServer
}

func NewServer(service *llmservice.Service, logger logging.SprintfLogger) *Server {
	return &Server{
		service: service,
		logger:  logger.With("module", "llamagrpcserver"),
	}
}

func (server *Server) Stop() {
	server.service.Stop()
}

func (server *Server) Ping(ctx context.Context, pingRequest *proto.PingRequest) (*proto.PingResponse, error) {
	server.logger.Debugf("Ping done")
	return &proto.PingResponse{}, nil
}

func (server *Server) LoadModel(loadModelRequest *proto.LoadModelRequest, stream proto.LLMServer_LoadModelServer) error {
	server.logger.Debugf("LoadModel: %s", loadModelRequest.Path)

	progressFunc := func(progress float32) {
		msg := proto.LoadModelResponse{Progress: progress}
		if err := stream.Send(&msg); err != nil {
			server.logger.Errorf("LoadModel: stream Send failed: %v", err)
		}
	}

	return server.service.LoadModel(loadModelRequest.Path, progressFunc)
}

func (server *Server) Predict(predictRequest *proto.PredictRequest, stream proto.LLMServer_PredictServer) error {
	modelPath := predictRequest.Model
	prompt := predictRequest.Prompt
	maxTokens := int(predictRequest.MaxTokens)
	streamMode := predictRequest.Stream

	server.logger.Infof("Predict: model=%s, max_tokens=%d, stream=%v, temp=%.3f, top_p=%.3f, top_k=%d",
		modelPath, maxTokens, streamMode,
		predictRequest.Temperature, predictRequest.TopP, predictRequest.TopK)
	server.logger.Debugf("Predict: prompt: %s", prompt)

	if predictRequest.Options != nil {
		server.logPredictOptions(predictRequest.Options)
	}

	if maxTokens == 0 {
		server.logger.Infof("Predict: maxTokens=0, skipping generation")
		return nil
	}

	args := buildPredictArgs(predictRequest)
	server.logSamplingBehavior(args)

	var streamFunc inferenceengine.StreamFunc
	if streamMode {
		streamFunc = func(token, tokens int, message string) error {
			msg := proto.PredictResponse{
				Message: []byte(message),
				Token:   int32(token),
				Tokens:  int32(tokens),
			}
			if err := stream.Send(&msg); err != nil {
				server.logger.Errorf("Predict: stream Send failed: %v", err)
				return err
			}
			return nil
		}
	}

	response, err := server.service.Predict(modelPath, prompt, args, streamFunc)
	if err != nil {
		server.logger.Errorf("Predict: failed: %v", err)
		return err
	}

	if !streamMode {
		msg := proto.PredictResponse{Message: []byte(response)}
		if err := stream.Send(&msg); err != nil {
			server.logger.Errorf("Predict: stream Send failed (non-streaming): %v", err)
			return err
		}
	}

	server.logger.Debugf("Predict: done")
	return nil
}

func buildPredictArgs(req *proto.PredictRequest) inferenceengine.PredictArgs {
	nPredict := int(req.MaxTokens)
	if nPredict < 0 {
		nPredict = 0
	}
	temp := float32(req.Temperature)
	if temp < 0 {
		temp = 0
	}
	topP := float32(req.TopP)
	if topP < 0 {
		topP = 0
	}
	topK := int32(req.TopK)
	if topK < 0 {
		topK = 0
	}

	args := inferenceengine.PredictArgs{
		NPredict:          nPredict,
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

func (server *Server) logPredictOptions(opts *proto.PredictRequest_Options) {
	if opts.MinP != nil {
		server.logger.Infof("  option min_p: %.3f", *opts.MinP)
	}
	if opts.MinTokensToKeep != nil {
		server.logger.Infof("  option min_tokens_to_keep: %d", *opts.MinTokensToKeep)
	}
	if opts.MaxKvSize != nil {
		server.logger.Infof("  option max_kv_size: %d", *opts.MaxKvSize)
	}
	if opts.PrefillStepSize != nil {
		server.logger.Infof("  option prefill_step_size: %d", *opts.PrefillStepSize)
	}
	if opts.KvBits != nil {
		server.logger.Infof("  option kv_bits: %d", *opts.KvBits)
	}
	if opts.KvGroupSize != nil {
		server.logger.Infof("  option kv_group_size: %d", *opts.KvGroupSize)
	}
	if opts.QuantizedKvStart != nil {
		server.logger.Infof("  option quantized_kv_start: %d", *opts.QuantizedKvStart)
	}
	if opts.RepetitionPenalty != nil {
		server.logger.Infof("  option repetition_penalty: %.3f", *opts.RepetitionPenalty)
	}
	if opts.LengthPenalty != nil {
		server.logger.Infof("  option length_penalty: %.3f", *opts.LengthPenalty)
	}
	if opts.DiversityPenalty != nil {
		server.logger.Infof("  option diversity_penalty: %.3f", *opts.DiversityPenalty)
	}
	if opts.NoRepeatNgramSize != nil {
		server.logger.Infof("  option no_repeat_ngram_size: %d", *opts.NoRepeatNgramSize)
	}
	if opts.RandomSeed != nil {
		server.logger.Infof("  option random_seed: %d", *opts.RandomSeed)
	}
}

func (server *Server) logSamplingBehavior(args inferenceengine.PredictArgs) {
	if args.Temp == 0.0 || args.TopK == 1 {
		server.logger.Infof("Predict: GREEDY SAMPLING (deterministic)")
	} else {
		server.logger.Infof("Predict: RANDOMIZED SAMPLING (temp=%.3f)", args.Temp)
	}
	if args.RepetitionPenalty != 1.0 {
		server.logger.Infof("Predict: repetition_penalty=%.3f", args.RepetitionPenalty)
	}
	if args.RandomSeed >= 0 {
		server.logger.Infof("Predict: seed=%d (reproducible)", args.RandomSeed)
	}
}
