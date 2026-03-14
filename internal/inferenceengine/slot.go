package inferenceengine

import (
	"strings"
	"time"

	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
)

type slotState int

const (
	slotIdle       slotState = iota
	slotPrefilling           // prompt tokens being fed
	slotGenerating           // auto-regressive token generation
)

func (s slotState) String() string {
	switch s {
	case slotIdle:
		return "idle"
	case slotPrefilling:
		return "prefilling"
	case slotGenerating:
		return "generating"
	default:
		return "unknown"
	}
}

// slot tracks one inference sequence inside the shared context.
type slot struct {
	id    int
	seqId int
	state slotState
	pos   int

	// prefill
	promptTokens []int
	prefillIdx   int
	inputCount   int

	// generation
	nextToken int
	generated int
	maxTokens int

	// sampler (per-slot, owns lifecycle)
	samplerChain *llamacppbindings.SamplerChain
	sampler      *llamacppbindings.Sampler

	// request data
	stream   StreamFunc
	resultCh chan requestResult
	response strings.Builder

	startTime time.Time
}

// assign initialises a slot for a new request.
func (s *slot) assign(tokens []int, maxTokens int,
	chain *llamacppbindings.SamplerChain, sampler *llamacppbindings.Sampler,
	req *request) {

	s.state = slotPrefilling
	s.pos = 0
	s.promptTokens = tokens
	s.prefillIdx = 0
	s.inputCount = len(tokens)
	s.nextToken = 0
	s.generated = 0
	s.maxTokens = maxTokens
	s.samplerChain = chain
	s.sampler = sampler
	s.stream = req.stream
	s.resultCh = req.done
	s.response.Reset()
	s.startTime = time.Now()
}

// finish frees per-request resources and sends the result.
func (s *slot) finish(err error) {
	if s.samplerChain != nil {
		s.samplerChain.Free()
		s.samplerChain = nil
		s.sampler = nil
	}
	if err != nil {
		s.resultCh <- requestResult{err: err}
	} else {
		s.resultCh <- requestResult{text: s.response.String()}
	}
	s.state = slotIdle
	s.stream = nil
	s.resultCh = nil
	s.promptTokens = nil
}

// request is a pending inference request waiting for a slot.
type request struct {
	model  *llamacppbindings.Model
	prompt string
	args   PredictArgs
	stream StreamFunc
	done   chan requestResult
}

type requestResult struct {
	text string
	err  error
}
