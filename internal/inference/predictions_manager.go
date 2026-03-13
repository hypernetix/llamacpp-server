package inference

import (
	"errors"
	"sync"

	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
)

// ErrPredictionsManagerClosed is returned when the predictions manager is closed
var ErrPredictionsManagerClosed = errors.New("predictions manager closed")

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

// PredictFunc is a function type for making a prediction
type PredictFunc func(model *llamacppbindings.Model, prompt string, args PredictArgs, stream StreamFunc) (string, error)

// PredictionsManager interface defines the operations for managing predictions
type PredictionsManager interface {
	Predict(model *llamacppbindings.Model, prompt string, args PredictArgs, stream StreamFunc) (string, error)
	Stop()
}

func NewPredictionsManager(pf PredictFunc, nParallel int) PredictionsManager {
	var sem chan struct{}
	if nParallel > 0 {
		sem = make(chan struct{}, nParallel)
	}
	return &predictionsManager{
		pf:  pf,
		sem: sem,
	}
}

type predictionsManager struct {
	pf     PredictFunc
	closed bool
	mx     sync.Mutex
	wg     sync.WaitGroup
	sem    chan struct{}
}

func (m *predictionsManager) initiatePrediction() error {
	m.mx.Lock()
	defer m.mx.Unlock()
	if m.closed {
		return ErrPredictionsManagerClosed
	}
	m.wg.Add(1)
	return nil
}

func (m *predictionsManager) cancelPredictions() {
	m.mx.Lock()
	defer m.mx.Unlock()
	if m.closed {
		return
	}
	m.closed = true
}

func (m *predictionsManager) Predict(model *llamacppbindings.Model, prompt string, args PredictArgs, stream StreamFunc) (string, error) {
	err := m.initiatePrediction()
	if err != nil {
		return "", err
	}
	defer m.wg.Done()

	if m.sem != nil {
		m.sem <- struct{}{}
		defer func() { <-m.sem }()
	}

	return m.pf(model, prompt, args, stream)
}

func (m *predictionsManager) Stop() {
	m.cancelPredictions()
	m.wg.Wait()
}
