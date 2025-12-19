package llamacppgrpcserver

import (
	"fmt"
	"time"

	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
	"github.com/hypernetix/llamacpp_server/internal/logging"
)

type PredictCommandStreamFunc func(token, tokens int, message string) error

type PredictCommandArgs struct {
	NPredict          int
	Temp              float32
	TopP              float32
	TopK              int32
	MinP              float32
	MinTokensToKeep   int
	MaxKvSize         int
	PrefillStepSize   int
	KvBits            int
	KvGroupSize       int
	QuantizedKvStart  int
	RepetitionPenalty float32
	LengthPenalty     float32
	DiversityPenalty  float32
	NoRepeatNgramSize int
	RandomSeed        int
}

func NewPredictFunc(logger logging.SprintfLogger) PredictFunc {
	cmd := &predictCmd{
		logger: logger.With("module", "predictCmd"),
	}
	return cmd.Do
}

type PredictFunc func(model *llamacppbindings.Model, prompt string, args PredictCommandArgs, stream PredictCommandStreamFunc) (string, error)

type predictCmd struct {
	logger logging.SprintfLogger
}

func (cmd *predictCmd) Do(model *llamacppbindings.Model, prompt string, args PredictCommandArgs, stream PredictCommandStreamFunc) (string, error) {
	startTime := time.Now()
	cmd.logger.Infof("Do: Starting text generation")
	cmd.logger.Infof("Do: Model info: %+v", model.Info())
	cmd.logger.Debugf("Do: Prompt: %q", prompt)
	cmd.logger.Infof("Do: Generation parameters: %+v", args)

	// Get vocabulary information
	vocab := model.Vocab()
	vocabInfo := vocab.Info()
	nVocab := vocab.NTokens()
	cmd.logger.Infof("Do: Vocabulary - type: %d, tokens: %d, add_bos: %v",
		vocabInfo.Type, nVocab, vocab.AddBOS())

	// Set up context with improved parameters
	contextParams := llamacppbindings.NewContextDefaultParams()
	contextParams.SetNCtx(4096)       // Larger context window
	contextParams.SetNBatch(2048)     // Larger batch size for better performance
	contextParams.SetNThreads(0)      // Auto-detect threads
	contextParams.SetNThreadsBatch(0) // Auto-detect batch threads

	cmd.logger.Infof("Do: Creating context - ctx_size: %d, batch_size: %d",
		contextParams.NCtx(), contextParams.NBatch())

	context, err := llamacppbindings.NewContext(model, contextParams)
	if err != nil {
		cmd.logger.Errorf("Do: Failed to create context: %v", err)
		return "", fmt.Errorf("failed to create context: %w", err)
	}
	defer context.Free()

	cmd.logger.Infof("Do: Context created successfully - available: %d, used: %d",
		context.NCells(), context.NCellsUsed())

	// Build sophisticated sampler chain
	cmd.logger.Infof("Do: ===== BUILDING SAMPLER CHAIN =====")

	samplerChainParams := llamacppbindings.NewSamplerChainDefaultParams()
	samplerChain, err := llamacppbindings.NewSamplerChain(samplerChainParams)
	if err != nil {
		cmd.logger.Errorf("Do: Failed to create sampler chain: %v", err)
		return "", fmt.Errorf("failed to create sampler chain: %w", err)
	}
	defer samplerChain.Free()

	// Add samplers in the correct order (this order matters!)
	// 1. Penalties (repetition control) - should be first
	if args.RepetitionPenalty != 1.0 {
		cmd.logger.Infof("Do: ✓ Attempting to add penalties sampler (repetition=%.3f)", args.RepetitionPenalty)
		// Based on llama.cpp common parameters:
		// penalty_last_n: number of last tokens to consider (typically 64 or -1 for context size)
		// penalty_repeat: the repetition penalty (args.RepetitionPenalty)
		// penalty_freq: frequency penalty (0.0 = disabled, we don't have this parameter)
		// penalty_present: presence penalty (0.0 = disabled, we don't have this parameter)
		penaltyLastN := 64 // Standard value, can be adjusted
		if penaltyLastN > context.NCells() {
			penaltyLastN = context.NCells() // Use full context if smaller
		}

		penaltiesSampler, err := llamacppbindings.NewPenaltiesSampler(
			penaltyLastN,           // penalty_last_n
			args.RepetitionPenalty, // penalty_repeat
			0.0,                    // penalty_freq (not supported in our args)
			0.0,                    // penalty_present (not supported in our args)
		)
		if err != nil {
			cmd.logger.Errorf("Do: Failed to create penalties sampler: %v", err)
			cmd.logger.Warnf("Do: ⚪ Penalties sampler creation failed, skipping")
		} else {
			samplerChain.AddSampler(penaltiesSampler)
			cmd.logger.Infof("Do: ✓ Added penalties sampler (last_n=%d, repeat=%.3f)", penaltyLastN, args.RepetitionPenalty)
		}
	} else {
		cmd.logger.Infof("Do: ⚪ Skipping penalties sampler (penalty=1.0)")
	}

	// 2. Top-K filtering
	if args.TopK > 0 {
		cmd.logger.Infof("Do: ✓ Attempting to add top-k sampler (k=%d)", args.TopK)
		topKSampler, err := llamacppbindings.NewTopKSampler(int(args.TopK))
		if err != nil {
			cmd.logger.Errorf("Do: Failed to create top-k sampler: %v", err)
			cmd.logger.Warnf("Do: ⚪ Top-K sampler creation failed, skipping")
		} else {
			samplerChain.AddSampler(topKSampler)
			cmd.logger.Infof("Do: ✓ Added top-k sampler (k=%d)", args.TopK)
		}
	} else {
		cmd.logger.Infof("Do: ⚪ Skipping top-k sampler (k=%d)", args.TopK)
	}

	// 3. Top-P (nucleus) filtering
	if args.TopP > 0.0 && args.TopP < 1.0 {
		cmd.logger.Infof("Do: ✓ Attempting to add top-p sampler (p=%.3f, min_keep=%d)",
			args.TopP, args.MinTokensToKeep)
		topPSampler, err := llamacppbindings.NewTopPSampler(args.TopP, args.MinTokensToKeep)
		if err != nil {
			cmd.logger.Errorf("Do: Failed to create top-p sampler: %v", err)
			cmd.logger.Warnf("Do: ⚪ Top-P sampler creation failed, skipping")
		} else {
			samplerChain.AddSampler(topPSampler)
			cmd.logger.Infof("Do: ✓ Added top-p sampler (p=%.3f, min_keep=%d)", args.TopP, args.MinTokensToKeep)
		}
	} else {
		cmd.logger.Infof("Do: ⚪ Skipping top-p sampler (p=%.3f)", args.TopP)
	}

	// 4. Min-P filtering
	if args.MinP > 0.0 {
		cmd.logger.Infof("Do: ✓ Adding min-p sampler (p=%.3f, min_keep=%d)",
			args.MinP, args.MinTokensToKeep)
		minPSampler, err := llamacppbindings.NewMinPSampler(args.MinP, args.MinTokensToKeep)
		if err != nil {
			cmd.logger.Errorf("Do: Failed to create min-p sampler: %v", err)
			return "", fmt.Errorf("failed to create min-p sampler: %w", err)
		}
		samplerChain.AddSampler(minPSampler)
		cmd.logger.Infof("Do: ✓ Added min-p sampler")
	} else {
		cmd.logger.Infof("Do: ⚪ Skipping min-p sampler (p=%.3f)", args.MinP)
	}

	// 5. Temperature scaling
	if args.Temp > 0.0 {
		cmd.logger.Infof("Do: ✓ Adding temperature sampler (temp=%.3f)", args.Temp)
		tempSampler, err := llamacppbindings.NewTempSampler(args.Temp)
		if err != nil {
			cmd.logger.Errorf("Do: Failed to create temperature sampler: %v", err)
			return "", fmt.Errorf("failed to create temperature sampler: %w", err)
		}
		samplerChain.AddSampler(tempSampler)
		cmd.logger.Infof("Do: ✓ Added temperature sampler")
	} else {
		cmd.logger.Infof("Do: ⚪ Using greedy sampling (temp=%.3f)", args.Temp)
	}

	// 6. Final distribution sampler (for randomness/seed control)
	var seed uint32 = 0xFFFFFFFF // Default random seed
	if args.RandomSeed >= 0 {
		seed = uint32(args.RandomSeed)
		cmd.logger.Infof("Do: ✓ Adding distribution sampler (seed=%d)", seed)
	} else {
		cmd.logger.Infof("Do: ✓ Adding distribution sampler (random seed)")
	}

	distSampler, err := llamacppbindings.NewDistSampler(seed)
	if err != nil {
		cmd.logger.Errorf("Do: Failed to create distribution sampler: %v", err)
		return "", fmt.Errorf("failed to create distribution sampler: %w", err)
	}
	samplerChain.AddSampler(distSampler)
	cmd.logger.Infof("Do: ✓ Added distribution sampler")

	cmd.logger.Infof("Do: ===== SAMPLER CHAIN BUILT SUCCESSFULLY =====")

	sampler := samplerChain.Sampler()

	// Tokenize the prompt
	cmd.logger.Infof("Do: Tokenizing prompt...")
	isFirst := context.NCellsUsed() == 0
	tokens, err := vocab.Tokenize(prompt, isFirst, true)
	if err != nil {
		cmd.logger.Errorf("Do: Failed to tokenize prompt: %v", err)
		return "", fmt.Errorf("failed to tokenize prompt: %w", err)
	}

	inputTokensCount := len(tokens)
	cmd.logger.Infof("Do: Tokenized prompt into %d tokens: %v", inputTokensCount, tokens)

	// Validate context size
	maxContextSize := context.NCells()
	if inputTokensCount+args.NPredict > maxContextSize {
		cmd.logger.Warnf("Do: Prompt + generation (%d + %d = %d) exceeds context size (%d), adjusting...",
			inputTokensCount, args.NPredict, inputTokensCount+args.NPredict, maxContextSize)
		args.NPredict = maxContextSize - inputTokensCount - 10 // Leave some buffer
		if args.NPredict < 0 {
			return "", fmt.Errorf("prompt too long for context size")
		}
	}

	batch := llamacppbindings.NotOwnedOneItemBatch(tokens)
	defer batch.Free()

	var response string
	pos := 0
	generatedTokens := 0

	cmd.logger.Infof("Do: Starting generation loop (max_tokens=%d)...", args.NPredict)
	generationStartTime := time.Now()

	for {
		nCells := context.NCells()
		nUsed := context.NCellsUsed()
		nBatch := batch.NTokens()

		cmd.logger.Debugf("Do: Loop iteration - pos=%d, batch_size=%d, ctx_used=%d/%d, generated=%d/%d",
			pos, nBatch, nUsed, nCells, generatedTokens, args.NPredict)

		// Check if we should stop generation
		if args.NPredict > 0 && generatedTokens >= args.NPredict {
			cmd.logger.Infof("Do: Reached max tokens limit (%d)", args.NPredict)
			break
		}

		// Check if context is full
		if nUsed+nBatch > nCells {
			cmd.logger.Errorf("Do: Context size exceeded (%d + %d > %d)", nUsed, nBatch, nCells)
			return "", fmt.Errorf("context size exceeded")
		}

		// Decode the batch
		err := context.Decode(batch)
		if err != nil {
			if err == llamacppbindings.ErrKvCacheFull {
				cmd.logger.Warnf("Do: KV cache full, stopping generation")
				break
			}
			cmd.logger.Errorf("Do: Failed to decode batch: %v", err)
			return "", fmt.Errorf("failed to decode batch: %w", err)
		}

		pos += nBatch

		// Sample next token
		nextToken := sampler.Sample(context, -1)

		// Check for end-of-generation
		if vocab.IsEog(nextToken) {
			cmd.logger.Infof("Do: End-of-generation token encountered, stopping")
			break
		}

		cmd.logger.Debugf("Do: Sampled token: %d", nextToken)

		// Convert token to text
		piece, err := vocab.TokenToPiece(nextToken)
		if err != nil {
			cmd.logger.Errorf("Do: Failed to convert token to text: %v", err)
			return "", fmt.Errorf("failed to convert token to text: %w", err)
		}

		cmd.logger.Debugf("Do: Token piece: %q", piece)

		// Send streaming response or accumulate
		if stream != nil {
			err := stream(nextToken, len(tokens)+generatedTokens, piece)
			if err != nil {
				cmd.logger.Errorf("Do: Streaming callback failed: %v", err)
				return "", fmt.Errorf("streaming callback failed: %w", err)
			}
		}
		response += piece

		generatedTokens++

		// Create batch for next token
		batch = llamacppbindings.NotOwnedOneItemBatch([]int{nextToken})
	}

	// Calculate performance metrics
	generationTime := time.Since(generationStartTime)
	totalTime := time.Since(startTime)
	throughput := float64(generatedTokens) / generationTime.Seconds()

	cmd.logger.Infof("Do: ===== GENERATION COMPLETED =====")
	cmd.logger.Infof("Do: Input tokens: %d", inputTokensCount)
	cmd.logger.Infof("Do: Generated tokens: %d", generatedTokens)
	cmd.logger.Infof("Do: Total tokens: %d", inputTokensCount+generatedTokens)
	cmd.logger.Infof("Do: Generation time: %.2fs", generationTime.Seconds())
	cmd.logger.Infof("Do: Total time: %.2fs", totalTime.Seconds())
	cmd.logger.Infof("Do: Throughput: %.2f tokens/second", throughput)
	cmd.logger.Infof("Do: Context usage: %d/%d (%.1f%%)",
		context.NCellsUsed(), context.NCells(),
		float64(context.NCellsUsed())/float64(context.NCells())*100)
	cmd.logger.Debugf("Do: Final response: %q", response)

	return response, nil
}
