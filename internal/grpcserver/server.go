package llamacppgrpcserver

import (
	"context"
	"fmt"

	"github.com/hypernetix/llamacpp_server/api/proto"
	"github.com/hypernetix/llamacpp_server/internal/logging"
	"github.com/hypernetix/llamacpp_server/internal/modelmanagement"
)

type GlobalOptions struct {
	Model LoadModelOptions
}

type LLMServer struct {
	logger logging.SprintfLogger
	proto.UnimplementedLLMServerServer
	modelManager      modelmanagement.ModelManager
	predictionManager PredictionsManager
}

func NewLLMServer(opts GlobalOptions, logger logging.SprintfLogger) *LLMServer {
	loadModelFunc := NewLoadModelFunc(opts.Model, logger)
	modelManager := modelmanagement.NewModelManager(loadModelFunc, logger)
	predictFunc := NewPredictFunc(logger)
	predictionManager := NewPredictionsManager(predictFunc)
	return &LLMServer{
		modelManager:      modelManager,
		predictionManager: predictionManager,
		logger:            logger.With("module", "llamagrpcserver"),
	}
}

func (server *LLMServer) Stop() {
	server.predictionManager.Stop()
	server.modelManager.Stop()
}

func (server *LLMServer) Ping(ctx context.Context, pingRequest *proto.PingRequest) (*proto.PingResponse, error) {
	msg := proto.PingResponse{}
	server.logger.Debugf("Ping done")
	return &msg, nil
}

func (server *LLMServer) LoadModel(loadModelRequest *proto.LoadModelRequest, stream proto.LLMServer_LoadModelServer) error {
	server.logger.Debugf("LoadModel: %s", loadModelRequest.Path)

	progressFunc := func(progress float32) {
		msg := proto.LoadModelResponse{
			Progress: progress,
		}
		if err := stream.Send(&msg); err != nil {
			server.logger.Errorf("LoadModel: stream Send failed: %v", err)
			return
		}
	}

	model, err := server.modelManager.LoadModel(loadModelRequest.Path, progressFunc)
	if err != nil {
		server.logger.Errorf("LoadModel: failed to load model: %v", err)
		return err
	}

	modelData, ok := model.(*ModelData)
	if !ok {
		server.logger.Errorf("LoadModel: invalid model type")
		return fmt.Errorf("invalid model type")
	}

	server.logger.Debugf("LoadModel: model loaded, params: %+v, info: %+v",
		modelData.ModelParams, modelData.Model.Info())
	return nil
}

func BuildPredictCommandArgs(predictRequest *proto.PredictRequest) PredictCommandArgs {
	nPredict := int(predictRequest.MaxTokens)
	if nPredict < 0 {
		nPredict = 0
	}
	temp := float32(predictRequest.Temperature)
	if temp < 0 {
		temp = 0
	}
	topP := float32(predictRequest.TopP)
	if topP < 0 {
		topP = 0
	}
	topK := int32(predictRequest.TopK)
	if topK < 0 {
		topK = 0
	}

	args := PredictCommandArgs{
		NPredict: nPredict,
		Temp:     temp,
		TopP:     topP,
		TopK:     topK,
	}

	// Set default values for advanced options
	defaultMinP := float32(0.05)
	defaultMinTokensToKeep := int32(1)
	defaultMaxKvSize := int32(0)
	defaultPrefillStepSize := int32(0)
	defaultKvBits := int32(0)
	defaultKvGroupSize := int32(0)
	defaultQuantizedKvStart := int32(0)
	defaultRepetitionPenalty := float32(1.0)
	defaultLengthPenalty := float32(1.0)
	defaultDiversityPenalty := float32(0.0)
	defaultNoRepeatNgramSize := int32(0)
	defaultRandomSeed := int32(-1) // -1 means random

	if predictRequest.Options != nil {
		minP := defaultMinP
		if predictRequest.Options.MinP != nil {
			minP = *predictRequest.Options.MinP
			if minP < 0 {
				minP = 0
			}
		}
		minTokensToKeep := defaultMinTokensToKeep
		if predictRequest.Options.MinTokensToKeep != nil {
			minTokensToKeep = *predictRequest.Options.MinTokensToKeep
			if minTokensToKeep < 0 {
				minTokensToKeep = 0
			}
		}
		maxKvSize := defaultMaxKvSize
		if predictRequest.Options.MaxKvSize != nil {
			maxKvSize = *predictRequest.Options.MaxKvSize
			if maxKvSize < 0 {
				maxKvSize = 0
			}
		}
		prefillStepSize := defaultPrefillStepSize
		if predictRequest.Options.PrefillStepSize != nil {
			prefillStepSize = *predictRequest.Options.PrefillStepSize
			if prefillStepSize < 0 {
				prefillStepSize = 0
			}
		}
		kvBits := defaultKvBits
		if predictRequest.Options.KvBits != nil {
			kvBits = *predictRequest.Options.KvBits
			if kvBits < 0 {
				kvBits = 0
			}
		}
		kvGroupSize := defaultKvGroupSize
		if predictRequest.Options.KvGroupSize != nil {
			kvGroupSize = *predictRequest.Options.KvGroupSize
			if kvGroupSize < 0 {
				kvGroupSize = 0
			}
		}
		quantizedKvStart := defaultQuantizedKvStart
		if predictRequest.Options.QuantizedKvStart != nil {
			quantizedKvStart = *predictRequest.Options.QuantizedKvStart
			if quantizedKvStart < 0 {
				quantizedKvStart = 0
			}
		}
		repetitionPenalty := defaultRepetitionPenalty
		if predictRequest.Options.RepetitionPenalty != nil {
			repetitionPenalty = *predictRequest.Options.RepetitionPenalty
		}
		lengthPenalty := defaultLengthPenalty
		if predictRequest.Options.LengthPenalty != nil {
			lengthPenalty = *predictRequest.Options.LengthPenalty
		}
		diversityPenalty := defaultDiversityPenalty
		if predictRequest.Options.DiversityPenalty != nil {
			diversityPenalty = *predictRequest.Options.DiversityPenalty
		}
		noRepeatNgramSize := defaultNoRepeatNgramSize
		if predictRequest.Options.NoRepeatNgramSize != nil {
			noRepeatNgramSize = *predictRequest.Options.NoRepeatNgramSize
		}
		randomSeed := defaultRandomSeed
		if predictRequest.Options.RandomSeed != nil {
			randomSeed = *predictRequest.Options.RandomSeed
		}

		args.MinP = minP
		args.MinTokensToKeep = int(minTokensToKeep)
		args.MaxKvSize = int(maxKvSize)
		args.PrefillStepSize = int(prefillStepSize)
		args.KvBits = int(kvBits)
		args.KvGroupSize = int(kvGroupSize)
		args.QuantizedKvStart = int(quantizedKvStart)
		args.RepetitionPenalty = repetitionPenalty
		args.LengthPenalty = lengthPenalty
		args.DiversityPenalty = diversityPenalty
		args.NoRepeatNgramSize = int(noRepeatNgramSize)
		args.RandomSeed = int(randomSeed)
	} else {
		args.MinP = defaultMinP
		args.MinTokensToKeep = int(defaultMinTokensToKeep)
		args.MaxKvSize = int(defaultMaxKvSize)
		args.PrefillStepSize = int(defaultPrefillStepSize)
		args.KvBits = int(defaultKvBits)
		args.KvGroupSize = int(defaultKvGroupSize)
		args.QuantizedKvStart = int(defaultQuantizedKvStart)
		args.RepetitionPenalty = defaultRepetitionPenalty
		args.LengthPenalty = defaultLengthPenalty
		args.DiversityPenalty = defaultDiversityPenalty
		args.NoRepeatNgramSize = int(defaultNoRepeatNgramSize)
		args.RandomSeed = int(defaultRandomSeed)
	}

	return args
}

func (server *LLMServer) Predict(predictRequest *proto.PredictRequest, stream proto.LLMServer_PredictServer) error {
	modelPath := predictRequest.Model
	prompt := predictRequest.Prompt
	maxTokens := int(predictRequest.MaxTokens)
	streamMode := predictRequest.Stream
	temp := predictRequest.Temperature
	topP := predictRequest.TopP
	topK := predictRequest.TopK

	server.logger.Infof("Predict: Starting generation for model=%s, max_tokens=%d, stream=%v",
		modelPath, maxTokens, streamMode)
	server.logger.Infof("Predict: Core parameters - temp=%.3f, top_p=%.3f, top_k=%d",
		temp, topP, topK)
	server.logger.Debugf("Predict: Using prompt: %s", prompt)

	// Log options if present
	if predictRequest.Options != nil {
		opts := predictRequest.Options
		server.logger.Infof("Predict: Advanced options provided:")

		if opts.MinP != nil {
			server.logger.Infof("  - min_p: %.3f", *opts.MinP)
		}
		if opts.MinTokensToKeep != nil {
			server.logger.Infof("  - min_tokens_to_keep: %d", *opts.MinTokensToKeep)
		}
		if opts.MaxKvSize != nil {
			server.logger.Infof("  - max_kv_size: %d", *opts.MaxKvSize)
		}
		if opts.PrefillStepSize != nil {
			server.logger.Infof("  - prefill_step_size: %d", *opts.PrefillStepSize)
		}
		if opts.KvBits != nil {
			server.logger.Infof("  - kv_bits: %d", *opts.KvBits)
		}
		if opts.KvGroupSize != nil {
			server.logger.Infof("  - kv_group_size: %d", *opts.KvGroupSize)
		}
		if opts.QuantizedKvStart != nil {
			server.logger.Infof("  - quantized_kv_start: %d", *opts.QuantizedKvStart)
		}
		if opts.RepetitionPenalty != nil {
			server.logger.Infof("  - repetition_penalty: %.3f", *opts.RepetitionPenalty)
		}
		if opts.LengthPenalty != nil {
			server.logger.Infof("  - length_penalty: %.3f", *opts.LengthPenalty)
		}
		if opts.DiversityPenalty != nil {
			server.logger.Infof("  - diversity_penalty: %.3f", *opts.DiversityPenalty)
		}
		if opts.NoRepeatNgramSize != nil {
			server.logger.Infof("  - no_repeat_ngram_size: %d", *opts.NoRepeatNgramSize)
		}
		if opts.RandomSeed != nil {
			server.logger.Infof("  - random_seed: %d", *opts.RandomSeed)
		}
	} else {
		server.logger.Infof("Predict: No advanced options provided")
	}

	if maxTokens == 0 {
		server.logger.Infof("Predict: Request has maxTokens=0, skipping generation")
		return nil
	}

	model, err := server.modelManager.GetModel(predictRequest.Model)
	if err != nil {
		server.logger.Errorf("Predict: failed to get model: %v", err)
		return err
	}

	modelData, ok := model.(*ModelData)
	if !ok {
		server.logger.Errorf("Predict: invalid model type")
		return fmt.Errorf("invalid model type")
	}

	server.logger.Debugf("Predict: model got, params: %+v, info: %+v",
		modelData.ModelParams, modelData.Model.Info())

	var streamFunc PredictCommandStreamFunc
	if predictRequest.Stream {
		streamFunc = func(token, tokens int, message string) error {
			msg := proto.PredictResponse{
				Message: []byte(message),
				Token:   int32(token),
				Tokens:  int32(tokens),
			}
			if err := stream.Send(&msg); err != nil {
				server.logger.Errorf("Predict: stream Send failed (streaming mode): %v", err)
				return err
			}
			return nil
		}
	}

	prompt = predictRequest.Prompt
	args := BuildPredictCommandArgs(predictRequest)

	// Log final parameter summary
	server.logger.Infof("Predict: ===== FINAL PARAMETER SUMMARY =====")
	server.logger.Infof("Predict: Model: %s", modelPath)
	server.logger.Infof("Predict: Core parameters:")
	server.logger.Infof("  - temperature: %.3f", args.Temp)
	server.logger.Infof("  - top_p: %.3f", args.TopP)
	server.logger.Infof("  - top_k: %d", args.TopK)
	server.logger.Infof("  - max_tokens: %d", args.NPredict)
	server.logger.Infof("  - min_p: %.3f", args.MinP)
	server.logger.Infof("  - min_tokens_to_keep: %d", args.MinTokensToKeep)

	server.logger.Infof("Predict: Advanced parameters:")
	server.logger.Infof("  - repetition_penalty: %.3f", args.RepetitionPenalty)
	server.logger.Infof("  - length_penalty: %.3f", args.LengthPenalty)
	server.logger.Infof("  - diversity_penalty: %.3f", args.DiversityPenalty)
	server.logger.Infof("  - no_repeat_ngram_size: %d", args.NoRepeatNgramSize)
	server.logger.Infof("  - random_seed: %d", args.RandomSeed)

	// Add parameter interpretation
	server.logger.Infof("Predict: ===== SAMPLING BEHAVIOR INTERPRETATION =====")

	// Determine sampling mode based on parameters
	if args.Temp == 0.0 || args.TopK == 1 {
		server.logger.Infof("Predict: 🎯 GREEDY SAMPLING (deterministic)")
		if args.Temp == 0.0 {
			server.logger.Infof("  - Reason: temperature=0.0")
		}
		if args.TopK == 1 {
			server.logger.Infof("  - Reason: top_k=1")
		}
	} else {
		server.logger.Infof("Predict: 🎲 RANDOMIZED SAMPLING")

		if args.TopK > 1 && args.TopP > 0.0 && args.TopP < 1.0 {
			server.logger.Infof("  - Mode: Combined Top-K (k=%d) + Top-P (p=%.3f)", args.TopK, args.TopP)
		} else if args.TopK > 1 {
			server.logger.Infof("  - Mode: Top-K only (k=%d)", args.TopK)
		} else if args.TopP > 0.0 && args.TopP < 1.0 {
			server.logger.Infof("  - Mode: Top-P/Nucleus only (p=%.3f)", args.TopP)
		} else if args.TopK == 0 && (args.TopP >= 1.0 || args.TopP == 0.0) {
			server.logger.Infof("  - Mode: Temperature-only sampling (no top-k/top-p filtering)")
		}

		server.logger.Infof("  - Temperature: %.3f (%.1f%% randomness)", args.Temp, args.Temp*100)
	}

	// Repetition handling
	if args.RepetitionPenalty == 1.0 {
		server.logger.Infof("  - Repetition: No penalty (penalty=1.0)")
	} else if args.RepetitionPenalty > 1.0 {
		server.logger.Infof("  - Repetition: Discouraged (penalty=%.3f)", args.RepetitionPenalty)
	} else {
		server.logger.Infof("  - Repetition: Encouraged (penalty=%.3f)", args.RepetitionPenalty)
	}

	// Determinism
	if args.RandomSeed >= 0 {
		server.logger.Infof("  - Determinism: Reproducible (seed=%d)", args.RandomSeed)
	} else {
		server.logger.Infof("  - Determinism: Non-reproducible (no seed set)")
	}

	server.logger.Infof("Predict: ==========================================")

	response, err := server.predictionManager.Predict(modelData.Model, prompt, args, streamFunc)
	if err != nil {
		server.logger.Errorf("Predict: failed to predict: %v", err)
		return err
	}

	if !predictRequest.Stream {
		msg := proto.PredictResponse{
			Message: []byte(response),
		}
		if err := stream.Send(&msg); err != nil {
			server.logger.Errorf("Predict: stream Send failed (non-streaming mode): %v", err)
			return err
		}
	}

	server.logger.Debugf("Predict: done")
	return nil
}
