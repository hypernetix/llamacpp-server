package modelmanagement

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// simpleMockLoadFunc creates a loadFunc that simulates loading a model
func simpleMockLoadFunc(delay time.Duration, returnErr error) LoadModelFunc {
	return func(path string, progress LoadModelProgressFunc) (interface{}, error) {
		// Simulate progress updates
		if progress != nil {
			progress(0.0)
			if delay > 0 {
				time.Sleep(delay / 2)
				progress(0.5)
				time.Sleep(delay / 2)
			}
			progress(1.0)
		}

		if returnErr != nil {
			return nil, returnErr
		}

		// Return nil pointers - this is safe for testing since we don't call Free()
		// and avoids crashes that can occur with unsafe pointer casting
		return nil, nil
	}
}

func TestImportLoadModel(t *testing.T) {
	// Create a model manager with a mock load function
	manager := NewModelManager(simpleMockLoadFunc(10*time.Millisecond, nil), nil)

	// Test progress tracking
	progressCalled := false
	progressFunc := func(p float32) {
		progressCalled = true
		require.GreaterOrEqual(t, p, float32(0.0))
		require.LessOrEqual(t, p, float32(1.0))
	}

	// Load a model
	_, err := manager.LoadModel("test_model.bin", progressFunc)

	// Verify results
	require.NoError(t, err)
	// We don't check the model value since our mock returns nil
	require.True(t, progressCalled)

	// Clean up
	manager.Stop()
}

func TestImportLoadModelError(t *testing.T) {
	// Create an error to return
	expectedErr := errors.New("mock error")

	// Create a model manager with a mock load function that returns an error
	manager := NewModelManager(simpleMockLoadFunc(0, expectedErr), nil)

	// Load a model (should fail)
	model, err := manager.LoadModel("test_model.bin", func(p float32) {})

	// Verify results
	require.Error(t, err)
	require.Equal(t, expectedErr, err)
	require.Nil(t, model)

	// Clean up
	manager.Stop()
}

func TestRetryAfterModelLoadFailure(t *testing.T) {
	// Create a mock loader that fails on first attempt but succeeds on second attempt
	loadCount := 0
	mockLoadFunc := func(path string, progress LoadModelProgressFunc) (interface{}, error) {
		loadCount++

		// Simulate progress
		if progress != nil {
			progress(0.0)
			progress(0.5)
			progress(1.0)
		}

		if loadCount == 1 {
			// First attempt - fail with error
			return nil, errors.New("first load attempt failed")
		} else {
			// Second attempt - succeed
			return nil, nil
		}
	}

	// Create model manager with the mock load function
	manager := NewModelManager(mockLoadFunc, nil)
	defer manager.Stop()

	// First load attempt should fail and be cached
	_, err := manager.LoadModel("test_model.bin", func(p float32) {})
	require.Error(t, err)
	require.Equal(t, "first load attempt failed", err.Error())
	require.Equal(t, 1, loadCount)

	// Second call to LoadModel should retry and succeed
	model, err := manager.LoadModel("test_model.bin", func(p float32) {})
	require.NoError(t, err)
	require.Nil(t, model) // Our mock returns nil model
	require.Equal(t, 2, loadCount)
}
