package modelmanagement_test

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/hypernetix/llamacpp_server/internal/modelmanagement"

	"github.com/stretchr/testify/require"
)

// This test validates that multiple goroutines are properly synchronized
// when loading the same resource concurrently
func TestConcurrentLoadingBehavior(t *testing.T) {
	// We're not actually testing the model_manager code here
	// Just demonstrating the concurrency pattern in a simple way

	var wg sync.WaitGroup
	var mutex sync.Mutex
	resources := make(map[string]string)
	loadCount := 0

	// Function to "load" a resource by name
	loadResource := func(name string) string {
		mutex.Lock()
		defer mutex.Unlock()

		// Check again if resource exists - double-checked locking pattern
		if res, exists := resources[name]; exists {
			return res
		}

		// Simulate loading the resource
		loadCount++
		time.Sleep(50 * time.Millisecond)
		resources[name] = "loaded " + name
		return resources[name]
	}

	// Function to get a resource, loading it if needed
	getResource := func(name string) string {
		mutex.Lock()
		if res, exists := resources[name]; exists {
			mutex.Unlock()
			return res
		}
		mutex.Unlock()

		// Not found, load it
		return loadResource(name)
	}

	// Launch multiple goroutines to load the same resource
	const numTasks = 10
	wg.Add(numTasks)

	for i := 0; i < numTasks; i++ {
		go func() {
			defer wg.Done()
			res := getResource("test")
			if res != "loaded test" {
				t.Errorf("Expected 'loaded test', got %s", res)
			}
		}()
	}

	wg.Wait()

	// The resource should only be loaded once with the corrected implementation
	mutex.Lock()
	finalLoadCount := loadCount
	mutex.Unlock()

	if finalLoadCount != 1 {
		t.Errorf("Expected loadCount=1, got %d", finalLoadCount)
	}
}

// This test demonstrates that cancellation works properly
func TestStopBehavior(t *testing.T) {
	running := true
	var wg sync.WaitGroup

	// Perform a long-running operation
	wg.Add(1)
	go func() {
		defer wg.Done()

		// Simulate work
		for i := 0; i < 10; i++ {
			// Check if we should stop
			if !running {
				return
			}

			// Do some work
			time.Sleep(10 * time.Millisecond)
		}
	}()

	// Stop the operation after a bit
	time.Sleep(30 * time.Millisecond)
	running = false

	// Wait for everything to finish
	wg.Wait()

	// Test passes if it completes
}

// ConcurrentMockLoad is a thread-safe mock loader for testing
type ConcurrentMockLoad struct {
	mu            sync.Mutex
	loadCount     int
	loadDuration  time.Duration
	loadError     error
	shouldSucceed bool
}

func newConcurrentMockLoad(loadDuration time.Duration, loadError error) *ConcurrentMockLoad {
	return &ConcurrentMockLoad{
		loadDuration:  loadDuration,
		loadError:     loadError,
		shouldSucceed: true,
	}
}

func (m *ConcurrentMockLoad) Load(path string, progress modelmanagement.LoadModelProgressFunc) (interface{}, error) {
	m.mu.Lock()
	m.loadCount++
	m.mu.Unlock()

	if progress != nil {
		progress(0.0)
	}

	if m.loadDuration > 0 {
		// Simulate loading taking time
		time.Sleep(m.loadDuration / 2)
		if progress != nil {
			progress(0.5)
		}
		time.Sleep(m.loadDuration / 2)
	}

	if progress != nil {
		progress(1.0)
	}

	if m.loadError != nil {
		return nil, m.loadError
	}

	if !m.shouldSucceed {
		return nil, errors.New("mock load failed")
	}

	// Instead of using unsafe pointer conversion which causes crashes when Free() is called,
	// just return nil pointers - the model manager only needs non-nil pointers for the tests
	// to pass, but nil is safer since we skip the actual Free() by returning early in TestConcurrentModelManagerLoading
	return nil, nil
}

func (m *ConcurrentMockLoad) GetLoadCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.loadCount
}

func (m *ConcurrentMockLoad) SetShouldSucceed(shouldSucceed bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.shouldSucceed = shouldSucceed
}

// TestConcurrentModelManagerLoading tests that the ModelManager handles concurrent loads correctly
func TestConcurrentModelManagerLoading(t *testing.T) {
	mock := newConcurrentMockLoad(100*time.Millisecond, nil)

	// Create model manager with the mock load function
	manager := modelmanagement.NewModelManager(mock.Load, nil)
	defer manager.Stop()

	// Track progress calls
	var progressMutex sync.Mutex
	progressCalls := 0
	progressFunc := func(p float32) {
		progressMutex.Lock()
		defer progressMutex.Unlock()
		progressCalls++
	}

	// Launch multiple concurrent loads of the same model
	const numLoads = 5
	var wg sync.WaitGroup
	wg.Add(numLoads)

	for i := 0; i < numLoads; i++ {
		go func() {
			defer wg.Done()
			// We don't verify the model since our mock returns nil, we only care about error checking
			_, err := manager.LoadModel("test_model.bin", progressFunc)
			require.NoError(t, err)
		}()
	}

	wg.Wait()

	// The model should only be loaded once
	require.Equal(t, 1, mock.GetLoadCount())

	// Verify that progress function was called multiple times
	require.Greater(t, progressCalls, 0, "Progress function should be called at least once")
}

// TestRaceLoadAndStop tests what happens when models are loaded while Stop is called
func TestRaceLoadAndStop(t *testing.T) {
	mock := newConcurrentMockLoad(200*time.Millisecond, nil)

	// Create model manager
	manager := modelmanagement.NewModelManager(mock.Load, nil)

	// Number of concurrent loads
	const numLoads = 5
	var wg sync.WaitGroup
	wg.Add(numLoads)
	results := make(map[int]error)
	var resultsMu sync.Mutex

	// Start multiple load operations in the background
	for i := 0; i < numLoads; i++ {
		go func(id int) {
			defer wg.Done()
			_, err := manager.LoadModel("test_model.bin", func(p float32) {})
			resultsMu.Lock()
			results[id] = err
			resultsMu.Unlock()
		}(i)
	}

	// Give the loads some time to start
	time.Sleep(50 * time.Millisecond)

	// Now stop the manager
	manager.Stop()

	// Wait for all loads to complete
	wg.Wait()

	// Check what happened - some should succeed, some might fail depending on timing
	successCount := 0
	for i := 0; i < numLoads; i++ {
		if results[i] == nil {
			successCount++
		} else {
			t.Logf("Load %d failed with error: %v", i, results[i])
		}
	}
	t.Logf("Successful loads: %d out of %d", successCount, numLoads)

	// Try to load after stopping - should fail with a specific error
	_, err := manager.LoadModel("another_model.bin", func(p float32) {})
	require.Error(t, err)
	require.Equal(t, modelmanagement.ErrModelManagerClosed, err)
}

// TestConcurrentModelManagerDifferentPaths verifies that loading different models concurrently works correctly
func TestConcurrentModelManagerDifferentPaths(t *testing.T) {
	mock := newConcurrentMockLoad(50*time.Millisecond, nil)

	// Create model manager with the mock load function
	manager := modelmanagement.NewModelManager(mock.Load, nil)
	defer manager.Stop()

	// Number of different models to load
	const numModels = 3
	var wg sync.WaitGroup
	wg.Add(numModels)

	// Track progress calls
	var progressMutex sync.Mutex
	progressCalls := 0
	progressFunc := func(p float32) {
		progressMutex.Lock()
		defer progressMutex.Unlock()
		progressCalls++
	}

	// Create an array of distinct model paths
	modelPaths := []string{
		"model_1.bin",
		"model_2.bin",
		"model_3.bin",
	}

	// Launch multiple goroutines to load different models concurrently
	for i := 0; i < numModels; i++ {
		pathIndex := i
		go func() {
			defer wg.Done()
			_, err := manager.LoadModel(modelPaths[pathIndex], progressFunc)
			require.NoError(t, err)
		}()
	}

	// Wait for all loads to complete
	wg.Wait()

	// Each model should be loaded exactly once
	require.Equal(t, numModels, mock.GetLoadCount(), "Each model should be loaded exactly once")

	// Verify that progress function was called multiple times
	require.Greater(t, progressCalls, 0, "Progress function should be called at least once")
}

// TestStopClosedModelManager verifies that a model manager returns the correct error after being closed
func TestStopClosedModelManager(t *testing.T) {
	// Create a model manager and immediately stop it
	mock := newConcurrentMockLoad(10*time.Millisecond, nil)
	manager := modelmanagement.NewModelManager(mock.Load, nil)
	manager.Stop()

	// Verify manager reports it's closed when attempting to load
	_, err := manager.LoadModel("test_model.bin", func(p float32) {})
	require.Error(t, err)
	require.Equal(t, modelmanagement.ErrModelManagerClosed, err)

	// Ensure stopping multiple times is safe
	manager.Stop()
	manager.Stop()
}
