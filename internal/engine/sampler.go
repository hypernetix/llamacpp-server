package engine

import (
	"fmt"

	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
	"github.com/hypernetix/llamacpp_server/internal/inference"
	"github.com/hypernetix/llamacpp_server/internal/logging"
)

// buildSamplerChain constructs a sampler chain matching the given PredictArgs.
// The caller owns the returned chain and must Free it.
func buildSamplerChain(args inference.PredictArgs, logger logging.SprintfLogger) (
	*llamacppbindings.SamplerChain, *llamacppbindings.Sampler, error) {

	chainParams := llamacppbindings.NewSamplerChainDefaultParams()
	chain, err := llamacppbindings.NewSamplerChain(chainParams)
	if err != nil {
		return nil, nil, fmt.Errorf("sampler chain: %w", err)
	}

	if args.RepetitionPenalty != 0 && args.RepetitionPenalty != 1.0 {
		s, err := llamacppbindings.NewPenaltiesSampler(64, args.RepetitionPenalty, 0.0, 0.0)
		if err != nil {
			logger.Warnf("penalties sampler unavailable: %v", err)
		} else {
			chain.AddSampler(s)
		}
	}

	if args.TopK > 0 {
		s, err := llamacppbindings.NewTopKSampler(int(args.TopK))
		if err != nil {
			logger.Warnf("top-k sampler unavailable: %v", err)
		} else {
			chain.AddSampler(s)
		}
	}

	if args.TopP > 0.0 && args.TopP < 1.0 {
		s, err := llamacppbindings.NewTopPSampler(args.TopP, args.MinTokensToKeep)
		if err != nil {
			logger.Warnf("top-p sampler unavailable: %v", err)
		} else {
			chain.AddSampler(s)
		}
	}

	if args.MinP > 0.0 {
		s, err := llamacppbindings.NewMinPSampler(args.MinP, args.MinTokensToKeep)
		if err != nil {
			chain.Free()
			return nil, nil, fmt.Errorf("min-p sampler: %w", err)
		}
		chain.AddSampler(s)
	}

	if args.Temp > 0.0 {
		s, err := llamacppbindings.NewTempSampler(args.Temp)
		if err != nil {
			chain.Free()
			return nil, nil, fmt.Errorf("temp sampler: %w", err)
		}
		chain.AddSampler(s)
	}

	var seed uint32 = 0xFFFFFFFF
	if args.RandomSeed >= 0 {
		seed = uint32(args.RandomSeed)
	}
	dist, err := llamacppbindings.NewDistSampler(seed)
	if err != nil {
		chain.Free()
		return nil, nil, fmt.Errorf("dist sampler: %w", err)
	}
	chain.AddSampler(dist)

	return chain, chain.Sampler(), nil
}
