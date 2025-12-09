package llamacppgrpcserver

import (
	"errors"
	"sync"

	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
)

var ErrPredictionsManagerClosed = errors.New("predictions manager closed")

type PredictionsManager interface {
	Predict(model *llamacppbindings.Model, prompt string, options PredictCommandArgs, stream PredictCommandStreamFunc) (string, error)
	Stop()
}

type predictionsManager struct {
	predictFunc PredictFunc
	closed      bool
	mx          sync.Mutex
	wg          sync.WaitGroup
}

func NewPredictionsManager(predictFunc PredictFunc) PredictionsManager {
	return &predictionsManager{
		predictFunc: predictFunc,
	}
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

func (m *predictionsManager) Predict(model *llamacppbindings.Model, prompt string, options PredictCommandArgs, stream PredictCommandStreamFunc) (string, error) {
	err := m.initiatePrediction()
	if err != nil {
		return "", err
	}
	defer m.wg.Done()
	return m.predictFunc(model, prompt, options, stream)
}

func (m *predictionsManager) cancelPredictions() {
	m.mx.Lock()
	defer m.mx.Unlock()
	if m.closed {
		return
	}
	m.closed = true
}

func (m *predictionsManager) Stop() {
	m.cancelPredictions()
	m.wg.Wait()
}
