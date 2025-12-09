package modelmanagement

import (
	"fmt"
	"sync"

	"github.com/hypernetix/llamacpp_server/internal/logging"
)

// DestroyableModel defines an interface for models that require explicit resource cleanup.
type DestroyableModel interface {
	Destroy() error
}

// ModelState represents the state of a model being loaded
type ModelState struct {
	Model      interface{}
	Progresses []func(float32)
	Err        error
	Wg         sync.WaitGroup
	Mx         sync.Mutex
	logger     logging.SprintfLogger
}

// ModelManager interface defines the operations for managing model loading
type ModelManager interface {
	LoadModel(path string, progress LoadModelProgressFunc) (interface{}, error)
	GetModel(path string) (interface{}, error)
	Stop()
}

// LoadModelProgressFunc is a function type for reporting loading progress
type LoadModelProgressFunc func(float32)

// LoadModelFunc is a function type for loading a model
type LoadModelFunc func(path string, progress LoadModelProgressFunc) (interface{}, error)

// ErrModelManagerClosed is returned when the model manager is closed
var ErrModelManagerClosed = fmt.Errorf("model manager is closed")

// ErrModelNotFound is returned when a model is not found
var ErrModelNotFound = fmt.Errorf("model not found")

type modelManager struct {
	ModelStates   map[string]*ModelState
	LoadModelFunc LoadModelFunc
	Closed        bool
	Mx            sync.Mutex
	logger        logging.SprintfLogger
}

// NewModelManager creates a new model manager instance
func NewModelManager(loadModelFunc LoadModelFunc, logger logging.SprintfLogger) ModelManager {
	return &modelManager{
		ModelStates:   make(map[string]*ModelState),
		LoadModelFunc: loadModelFunc,
		Closed:        false,
		logger:        logger,
	}
}

func (state *ModelState) addProgress(progress func(float32)) {
	state.Mx.Lock()
	defer state.Mx.Unlock()
	state.Progresses = append(state.Progresses, progress)
}

func (state *ModelState) getProgresses() []func(float32) {
	state.Mx.Lock()
	defer state.Mx.Unlock()
	progresses := make([]func(float32), len(state.Progresses))
	copy(progresses, state.Progresses)
	return progresses
}

func (state *ModelState) broadcastingProgressFunc() func(float32) {
	var lastIntProgress int = 0
	return func(progress float32) {
		if progress > 0.0 && progress < 1.0 {
			intProgress := int(progress * 100.0)
			if intProgress < lastIntProgress+1 {
				return
			}
			lastIntProgress = intProgress
		}
		progresses := state.getProgresses()
		for _, p := range progresses {
			p(progress)
		}
	}
}

func (state *ModelState) getModel() (interface{}, error) {
	state.Mx.Lock()
	defer state.Mx.Unlock()
	if state.Err != nil {
		return nil, state.Err
	}
	return state.Model, nil
}

func (state *ModelState) saveLoaded(model interface{}, err error) {
	state.Mx.Lock()
	defer state.Mx.Unlock()
	state.Model = model
	state.Err = err
	state.Progresses = nil
}

func (state *ModelState) free() {
	state.Mx.Lock()
	defer state.Mx.Unlock()

	if state.Model != nil { // Check if model exists
		if dm, ok := state.Model.(DestroyableModel); ok {
			if state.logger != nil {
				state.logger.Debugf("Destroying model")
			}
			err := dm.Destroy()
			if err != nil {
				if state.logger != nil {
					state.logger.Errorf("Failed to destroy model: %v", err)
				}
			}
		}
		state.Model = nil // Nil it out regardless of whether it was Destroyable or if Destroy failed
	}
	state.Err = nil
	state.Progresses = nil
}

func (m *modelManager) initiateLoad(path string, progress LoadModelProgressFunc) (*ModelState, bool, error) {
	m.Mx.Lock()
	defer m.Mx.Unlock()
	if m.Closed {
		return nil, false, ErrModelManagerClosed
	}
	state, ok := m.ModelStates[path]
	if !ok {
		var logger logging.SprintfLogger
		if m.logger != nil {
			logger = m.logger.With("path", path)
		}
		state = &ModelState{
			logger: logger,
		}
		m.ModelStates[path] = state
		state.Wg.Add(1)
	}
	state.addProgress(progress)
	return state, ok, nil
}

func (m *modelManager) removeState(path string, state *ModelState) {
	m.Mx.Lock()
	defer m.Mx.Unlock()
	currentState := m.ModelStates[path]
	if currentState == state {
		delete(m.ModelStates, path)
	}
}

func (m *modelManager) LoadModel(path string, progress LoadModelProgressFunc) (interface{}, error) {
	state, ok, err := m.initiateLoad(path, progress)
	if err != nil {
		return nil, err
	}

	if ok {
		model, err := state.getModel()
		if model != nil || (err == nil) {
			return model, err
		}

		m.removeState(path, state)

		state.Wg.Wait()

		return m.LoadModel(path, progress)
	}

	defer state.Wg.Done()

	model, err := m.LoadModelFunc(path, state.broadcastingProgressFunc())

	state.saveLoaded(model, err)

	return model, err
}

func (m *modelManager) GetModel(path string) (interface{}, error) {
	m.Mx.Lock()
	defer m.Mx.Unlock()
	if m.Closed {
		return nil, ErrModelManagerClosed
	}
	state, ok := m.ModelStates[path]
	if !ok {
		return nil, ErrModelNotFound
	}
	return state.getModel()
}

func (m *modelManager) cancelLoads() []*ModelState {
	m.Mx.Lock()
	defer m.Mx.Unlock()
	if m.Closed {
		return nil
	}
	m.Closed = true
	modelStates := make([]*ModelState, 0, len(m.ModelStates))
	for _, state := range m.ModelStates {
		modelStates = append(modelStates, state)
	}
	return modelStates
}

func (m *modelManager) Stop() {
	modelStates := m.cancelLoads()
	for _, state := range modelStates {
		state.Wg.Wait()
		state.free()
	}
}
