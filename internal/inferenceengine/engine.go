package inferenceengine

import (
	"fmt"
	"time"

	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
	"github.com/hypernetix/llamacpp_server/internal/logging"
)

// StreamFunc is a function type for streaming a prediction
type StreamFunc func(token, tokens int, message string) error

// PredictArgs are the arguments for a prediction
type PredictArgs struct {
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

// PredictionsManager interface defines the operations for managing predictions
type PredictionsManager interface {
	Predict(model *llamacppbindings.Model, prompt string, args PredictArgs, stream StreamFunc) (string, error)
	Stop()
}

// Options configures the continuous batching engine.
type Options struct {
	NParallel     int
	CtxSize       int
	BatchSize     int
	NThreads      int
	NThreadsBatch int
	FlashAttn     bool
}

// Engine implements continuous batching inference with a single shared
// context and N concurrent slots. It satisfies PredictionsManager.
type Engine struct {
	opts   Options
	logger logging.SprintfLogger

	// llama.cpp state — owned by the run goroutine, never accessed concurrently
	model   *llamacppbindings.Model
	vocab   *llamacppbindings.Vocab
	context *llamacppbindings.Context
	memory  *llamacppbindings.Memory
	batch   *llamacppbindings.Batch
	slots   []*slot

	requests chan *request
	quit     chan struct{}
	done     chan struct{}
}

var _ PredictionsManager = (*Engine)(nil)

// New creates and starts a continuous batching engine.
// The engine goroutine runs until Stop is called.
func New(opts Options, logger logging.SprintfLogger) *Engine {
	if opts.NParallel <= 0 {
		opts.NParallel = 1
	}
	if opts.CtxSize <= 0 {
		opts.CtxSize = 4096
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = 2048
	}

	e := &Engine{
		opts:     opts,
		logger:   logger.With("module", "engine"),
		requests: make(chan *request, opts.NParallel*2),
		quit:     make(chan struct{}),
		done:     make(chan struct{}),
	}

	go e.run()
	return e
}

// Predict submits a request to the engine and blocks until completion.
func (e *Engine) Predict(
	model *llamacppbindings.Model,
	prompt string,
	args PredictArgs,
	stream StreamFunc,
) (string, error) {
	req := &request{
		model:  model,
		prompt: prompt,
		args:   args,
		stream: stream,
		done:   make(chan requestResult, 1),
	}

	select {
	case e.requests <- req:
	case <-e.quit:
		return "", fmt.Errorf("engine stopped")
	}

	res := <-req.done
	return res.text, res.err
}

// Stop shuts down the engine and waits for the run goroutine to finish.
func (e *Engine) Stop() {
	select {
	case <-e.quit:
		return
	default:
		close(e.quit)
	}
	<-e.done
}

// ---------------------------------------------------------------------------
// run loop
// ---------------------------------------------------------------------------

func (e *Engine) run() {
	defer close(e.done)
	e.logger.Infof("started (nParallel=%d, ctxSize=%d, batchSize=%d)",
		e.opts.NParallel, e.opts.CtxSize, e.opts.BatchSize)

	for {
		// When idle, block waiting for a request or shutdown signal.
		if !e.hasActiveSlots() {
			select {
			case req := <-e.requests:
				e.handleRequest(req)
			case <-e.quit:
				e.shutdown()
				return
			}
		}

		// Non-blocking: assign any additional queued requests to idle slots.
		e.drainPendingRequests()

		select {
		case <-e.quit:
			e.abortAll(fmt.Errorf("engine stopped"))
			e.shutdown()
			return
		default:
		}

		if e.hasActiveSlots() {
			if err := e.tick(); err != nil {
				e.logger.Errorf("fatal tick error: %v", err)
				e.abortAll(err)
			}
		}
	}
}

func (e *Engine) handleRequest(req *request) {
	if err := e.ensureContext(req.model); err != nil {
		req.done <- requestResult{err: err}
		return
	}
	idle := e.findIdleSlot()
	if idle == nil {
		req.done <- requestResult{err: fmt.Errorf("no idle slots available")}
		return
	}
	if err := e.assignSlot(idle, req); err != nil {
		req.done <- requestResult{err: err}
	}
}

func (e *Engine) drainPendingRequests() {
	for {
		idle := e.findIdleSlot()
		if idle == nil {
			return
		}
		select {
		case req := <-e.requests:
			if err := e.ensureContext(req.model); err != nil {
				req.done <- requestResult{err: err}
				continue
			}
			if err := e.assignSlot(idle, req); err != nil {
				req.done <- requestResult{err: err}
			}
		default:
			return
		}
	}
}

// ---------------------------------------------------------------------------
// context lifecycle
// ---------------------------------------------------------------------------

func (e *Engine) ensureContext(model *llamacppbindings.Model) error {
	if e.context != nil && e.model == model {
		return nil
	}
	if e.context != nil {
		if e.hasActiveSlots() {
			return fmt.Errorf("cannot switch model while requests are active")
		}
		e.teardown()
	}
	return e.initContext(model)
}

func (e *Engine) initContext(model *llamacppbindings.Model) error {
	params := llamacppbindings.NewContextDefaultParams()
	params.SetNCtx(e.opts.CtxSize)
	params.SetNBatch(e.opts.BatchSize)
	params.SetNSeqMax(e.opts.NParallel)
	params.SetNThreads(e.opts.NThreads)
	params.SetNThreadsBatch(e.opts.NThreadsBatch)
	if e.opts.FlashAttn {
		params.SetFlashAttention(true)
	}

	ctx, err := llamacppbindings.NewContext(model, params)
	if err != nil {
		return fmt.Errorf("create shared context: %w", err)
	}

	mem := ctx.Memory()
	if mem == nil {
		ctx.Free()
		return fmt.Errorf("context has no memory")
	}

	e.model = model
	e.vocab = model.Vocab()
	e.context = ctx
	e.memory = mem
	e.batch = llamacppbindings.BatchInit(e.opts.BatchSize, 0, e.opts.NParallel)

	e.slots = make([]*slot, e.opts.NParallel)
	for i := range e.slots {
		e.slots[i] = &slot{id: i, seqId: i, state: slotIdle}
	}

	e.logger.Infof("shared context ready (nCtx=%d, nBatch=%d, slots=%d)",
		e.opts.CtxSize, e.opts.BatchSize, e.opts.NParallel)
	return nil
}

func (e *Engine) teardown() {
	if e.batch != nil {
		e.batch.Free()
		e.batch = nil
	}
	if e.context != nil {
		e.context.Free()
		e.context = nil
	}
	e.memory = nil
	e.model = nil
	e.vocab = nil
	e.slots = nil
}

func (e *Engine) shutdown() {
	e.teardown()
	for {
		select {
		case req := <-e.requests:
			req.done <- requestResult{err: fmt.Errorf("engine stopped")}
		default:
			return
		}
	}
}

// ---------------------------------------------------------------------------
// slot management
// ---------------------------------------------------------------------------

func (e *Engine) hasActiveSlots() bool {
	for _, s := range e.slots {
		if s.state != slotIdle {
			return true
		}
	}
	return false
}

func (e *Engine) findIdleSlot() *slot {
	for _, s := range e.slots {
		if s.state == slotIdle {
			return s
		}
	}
	return nil
}

func (e *Engine) assignSlot(s *slot, req *request) error {
	tokens, err := e.vocab.Tokenize(req.prompt, true, true)
	if err != nil {
		return fmt.Errorf("tokenize: %w", err)
	}

	perSlotCtx := e.opts.CtxSize / e.opts.NParallel
	maxTokens := req.args.NPredict
	if len(tokens)+maxTokens > perSlotCtx {
		maxTokens = perSlotCtx - len(tokens)
		if maxTokens <= 0 {
			return fmt.Errorf("prompt (%d tokens) exceeds slot budget (%d)", len(tokens), perSlotCtx)
		}
	}

	chain, sampler, err := buildSamplerChain(req.args, e.logger)
	if err != nil {
		return err
	}

	e.memory.SeqRm(s.seqId, -1, -1)
	s.assign(tokens, maxTokens, chain, sampler, req)

	e.logger.Infof("slot %d: assigned (prompt=%d, maxGen=%d, seqId=%d)",
		s.id, len(tokens), maxTokens, s.seqId)
	return nil
}

func (e *Engine) finishSlot(s *slot, err error) {
	e.memory.SeqRm(s.seqId, -1, -1)

	dur := time.Since(s.startTime)
	if err != nil {
		e.logger.Infof("slot %d: error after %s: %v", s.id, dur, err)
	} else if dur.Seconds() > 0 {
		tps := float64(s.generated) / dur.Seconds()
		e.logger.Infof("slot %d: done (%d tokens, %s, %.1f tok/s)",
			s.id, s.generated, dur, tps)
	}

	s.finish(err)
}

func (e *Engine) abortAll(err error) {
	for _, s := range e.slots {
		if s.state != slotIdle {
			e.finishSlot(s, err)
		}
	}
}

// ---------------------------------------------------------------------------
// tick — one batch cycle: build → decode → sample → dispatch
// ---------------------------------------------------------------------------

type sampleTarget struct {
	slotIdx  int
	batchIdx int // position in the batch array — passed to llama_sampler_sample
}

func (e *Engine) tick() error {
	e.batch.Clear()
	var targets []sampleTarget

	// Phase 1: decode tokens from generating slots (one token each, highest
	// priority because they are blocking streaming output).
	for i, s := range e.slots {
		if s.state != slotGenerating {
			continue
		}
		batchIdx := e.batch.NTokens()
		e.batch.Add(s.nextToken, s.pos, s.seqId, true)
		s.pos++
		targets = append(targets, sampleTarget{slotIdx: i, batchIdx: batchIdx})
	}

	// Phase 2: fill remaining capacity with prefill chunks. Long prompts
	// are split across ticks so generating slots aren't starved.
	remaining := e.batch.Cap() - e.batch.NTokens()
	for i, s := range e.slots {
		if s.state != slotPrefilling || remaining <= 0 {
			continue
		}

		left := len(s.promptTokens) - s.prefillIdx
		chunk := left
		if chunk > remaining {
			chunk = remaining
		}

		for j := 0; j < chunk; j++ {
			last := s.prefillIdx+j+1 == len(s.promptTokens)
			batchIdx := e.batch.NTokens()
			e.batch.Add(s.promptTokens[s.prefillIdx+j], s.pos, s.seqId, last)
			s.pos++

			if last {
				targets = append(targets, sampleTarget{slotIdx: i, batchIdx: batchIdx})
			}
		}

		s.prefillIdx += chunk
		remaining -= chunk

		if s.prefillIdx >= len(s.promptTokens) {
			s.state = slotGenerating
			e.logger.Debugf("slot %d: prefill complete (%d tokens)", s.id, len(s.promptTokens))
		}
	}

	if e.batch.NTokens() == 0 {
		return nil
	}

	// Phase 3: single decode call for the entire batch.
	if err := e.context.Decode(e.batch); err != nil {
		return fmt.Errorf("decode: %w", err)
	}

	// Phase 4: sample at each target's batch position and dispatch results.
	// llama_sampler_sample takes the batch index (not a contiguous output index).
	for _, t := range targets {
		s := e.slots[t.slotIdx]
		token := s.sampler.Sample(e.context, t.batchIdx)

		if e.vocab.IsEog(token) {
			e.logger.Debugf("slot %d: EoG", s.id)
			e.finishSlot(s, nil)
			continue
		}

		if s.generated >= s.maxTokens {
			e.logger.Debugf("slot %d: max tokens reached (%d)", s.id, s.maxTokens)
			e.finishSlot(s, nil)
			continue
		}

		piece, err := e.vocab.TokenToPiece(token)
		if err != nil {
			e.finishSlot(s, fmt.Errorf("token to piece: %w", err))
			continue
		}

		if s.stream != nil {
			if err := s.stream(token, s.inputCount+s.generated, piece); err != nil {
				e.finishSlot(s, err)
				continue
			}
		}

		s.response.WriteString(piece)
		s.generated++
		s.nextToken = token
	}

	return nil
}
