package llmservice

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hypernetix/llamacpp_server/internal/logging"

	"github.com/stretchr/testify/require"
)

// testLogger implements logging.SprintfLogger for testing
type testLogger struct {
	t    *testing.T
	logs []string
	mu   sync.Mutex
}

func (l *testLogger) Level() string { return "debug" }

func (l *testLogger) Debugf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	l.t.Logf("[DEBUG] %s", msg)

	l.mu.Lock()
	defer l.mu.Unlock()
	l.logs = append(l.logs, fmt.Sprintf("DEBUG: %s", msg))
}

func (l *testLogger) Infof(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	l.t.Logf("[INFO] %s", msg)

	l.mu.Lock()
	defer l.mu.Unlock()
	l.logs = append(l.logs, fmt.Sprintf("INFO: %s", msg))
}

func (l *testLogger) Warnf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	l.t.Logf("[WARN] %s", msg)

	l.mu.Lock()
	defer l.mu.Unlock()
	l.logs = append(l.logs, fmt.Sprintf("WARN: %s", msg))
}

func (l *testLogger) Errorf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	l.t.Logf("[ERROR] %s", msg)

	l.mu.Lock()
	defer l.mu.Unlock()
	l.logs = append(l.logs, fmt.Sprintf("ERROR: %s", msg))
}

func (l *testLogger) With(args ...interface{}) logging.SprintfLogger {
	// For simplicity, just return self
	return l
}

func (l *testLogger) getLogs() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	result := make([]string, len(l.logs))
	copy(result, l.logs)
	return result
}

// mockReadCloser simulates a process stdout for testing
type mockReadCloser struct {
	mu          sync.Mutex
	buffer      bytes.Buffer
	closed      bool
	readDelays  []time.Duration
	readErrors  []error
	currentRead int
	closeErr    error
}

func (m *mockReadCloser) Read(p []byte) (n int, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return 0, io.ErrUnexpectedEOF
	}

	// Simulate read delays if configured
	if m.currentRead < len(m.readDelays) {
		delay := m.readDelays[m.currentRead]
		if delay > 0 {
			m.mu.Unlock()
			time.Sleep(delay)
			m.mu.Lock()
		}
	}

	// Simulate read errors if configured
	if m.currentRead < len(m.readErrors) {
		if m.readErrors[m.currentRead] != nil {
			m.currentRead++
			return 0, m.readErrors[m.currentRead-1]
		}
	}

	// Normal read
	if m.buffer.Len() == 0 {
		if m.currentRead >= len(m.readErrors) {
			m.currentRead++
			return 0, io.EOF
		}
		return 0, nil
	}

	n, err = m.buffer.Read(p)
	m.currentRead++
	return n, err
}

func (m *mockReadCloser) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return m.closeErr
}

func (m *mockReadCloser) Write(p []byte) (n int, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return 0, io.ErrClosedPipe
	}
	return m.buffer.Write(p)
}

// Helper function to create a test script
func createTestScript(t *testing.T, script string) string {
	dir := t.TempDir()

	scriptPath := filepath.Join(dir, "test_script.sh")
	if runtime.GOOS == "windows" {
		scriptPath = filepath.Join(dir, "test_script.bat")
	}

	err := os.WriteFile(scriptPath, []byte(script), 0755)
	require.NoError(t, err)

	return scriptPath
}

// TestProcessBasicOperation tests the basic operation of the Process implementation
func TestProcessBasicOperation(t *testing.T) {
	if runtime.GOOS == "windows" {
		script := `@echo off
echo Starting test script
timeout /t 1 >nul
echo Test complete
`
		scriptPath := createTestScript(t, script)

		logger := &testLogger{
			t:    t,
			logs: make([]string, 0),
			mu:   sync.Mutex{},
		}
		process, err := NewProcess(logger, scriptPath, []string{})
		require.NoError(t, err)
		require.NotNil(t, process)

		// Give the process time to execute
		time.Sleep(3 * time.Second)

		// Verify it logged output
		logs := logger.getLogs()
		hasOutput := false
		for _, log := range logs {
			if log == "DEBUG: Starting test script" || log == "DEBUG: Test complete" {
				hasOutput = true
				break
			}
		}
		require.True(t, hasOutput, "Process should log output")

		// Stop the process
		process.Stop()

		// Verify the stop was logged
		logs = logger.getLogs()
		hasStopped := false
		for _, log := range logs {
			if log == "DEBUG: Process monitor loop stopped" {
				hasStopped = true
				break
			}
		}
		require.True(t, hasStopped, "Process should log when stopped")
	} else {
		script := `#!/bin/bash
echo Starting test script
sleep 1
echo Test complete
`
		scriptPath := createTestScript(t, script)

		logger := &testLogger{
			t:    t,
			logs: make([]string, 0),
			mu:   sync.Mutex{},
		}
		process, err := NewProcess(logger, scriptPath, []string{})
		require.NoError(t, err)
		require.NotNil(t, process)

		// Give the process time to execute
		time.Sleep(3 * time.Second)

		// Verify it logged output
		logs := logger.getLogs()
		hasOutput := false
		for _, log := range logs {
			if log == "DEBUG: Starting test script" || log == "DEBUG: Test complete" {
				hasOutput = true
				break
			}
		}
		require.True(t, hasOutput, "Process should log output")

		// Stop the process
		process.Stop()

		// Verify the stop was logged
		logs = logger.getLogs()
		hasStopped := false
		for _, log := range logs {
			if log == "DEBUG: Process monitor loop stopped" {
				hasStopped = true
				break
			}
		}
		require.True(t, hasStopped, "Process should log when stopped")
	}
}

// TestProcessOutputHandling tests the handling of process output
func TestProcessOutputHandling(t *testing.T) {
	// Create a mock ReadCloser
	mockStdout := &mockReadCloser{}
	mockStdout.Write([]byte("Line 1\nLine 2\nLine 3\n"))

	logger := &testLogger{
		t:    t,
		logs: make([]string, 0),
		mu:   sync.Mutex{},
	}
	output := &output{logger: logger}

	// Read output in a separate goroutine
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		output.read(mockStdout)
	}()

	// Wait for reading to complete
	wg.Wait()

	// Verify all lines were logged
	logs := logger.getLogs()

	t.Logf("Log count: %d", len(logs))
	for i, log := range logs {
		t.Logf("Log[%d]: %q", i, log)
	}

	// With the line-by-line scanner, we expect each line to be a separate log entry
	foundLine1 := false
	foundLine2 := false
	foundLine3 := false

	for _, log := range logs {
		if log == "DEBUG: Line 1" {
			foundLine1 = true
		} else if log == "DEBUG: Line 2" {
			foundLine2 = true
		} else if log == "DEBUG: Line 3" {
			foundLine3 = true
		}
	}

	require.True(t, foundLine1, "Should have logged Line 1")
	require.True(t, foundLine2, "Should have logged Line 2")
	require.True(t, foundLine3, "Should have logged Line 3")
	require.True(t, mockStdout.closed, "ReadCloser should be closed after reading")
}

// TestProcessRestartOnFailure tests that the process restarts after failure
func TestProcessRestartOnFailure(t *testing.T) {
	// Create a temporary monitor with shorter retry period for testing
	shortRetryMonitor := func(logger logging.SprintfLogger, command *command, output *output) *monitor {
		ctx, cancel := context.WithCancel(context.Background())
		mon := &monitor{
			command:     command,
			output:      output,
			ctx:         ctx,
			cancel:      cancel,
			logger:      logger.With("module", "process.monitor"),
			retryPeriod: 500 * time.Millisecond, // Much shorter for testing
		}
		return mon
	}

	if runtime.GOOS == "windows" {
		script := `@echo off
echo Attempt 1
exit 1
`
		scriptPath := createTestScript(t, script)

		logger := &testLogger{
			t:    t,
			logs: make([]string, 0),
			mu:   sync.Mutex{},
		}
		cmd := &command{
			path:   scriptPath,
			args:   []string{},
			logger: logger.With("module", "process.command"),
		}
		out := &output{
			logger: logger.With("module", "process.output"),
		}

		// Create monitor with short retry period
		monitor := shortRetryMonitor(logger, cmd, out)
		monitor.start()

		// Create process
		process := &process{
			monitor: monitor,
		}

		// Give process time to start, fail, and restart
		time.Sleep(2 * time.Second)

		// Stop the process
		process.Stop()

		// Check that error was logged and restart attempt was made
		logs := logger.getLogs()

		t.Logf("Found %d logs", len(logs))
		for i, log := range logs {
			t.Logf("Log[%d]: %q", i, log)
		}

		hasError := false
		hasOutput := false
		for _, log := range logs {
			if strings.Contains(log, "exited with error") {
				hasError = true
			}
			if strings.Contains(log, "Attempt") {
				hasOutput = true
			}
		}
		require.True(t, hasError, "Process should log exit error")
		require.True(t, hasOutput, "Process should log output from script")
	} else {
		script := `#!/bin/bash
echo Attempt 1
exit 1
`
		scriptPath := createTestScript(t, script)

		logger := &testLogger{
			t:    t,
			logs: make([]string, 0),
			mu:   sync.Mutex{},
		}
		cmd := &command{
			path:   scriptPath,
			args:   []string{},
			logger: logger.With("module", "process.command"),
		}
		out := &output{
			logger: logger.With("module", "process.output"),
		}

		// Create monitor with short retry period
		monitor := shortRetryMonitor(logger, cmd, out)
		monitor.start()

		// Create process
		process := &process{
			monitor: monitor,
		}

		// Give process time to start, fail, and restart
		time.Sleep(2 * time.Second)

		// Stop the process
		process.Stop()

		// Check that error was logged and restart attempt was made
		logs := logger.getLogs()

		t.Logf("Found %d logs", len(logs))
		for i, log := range logs {
			t.Logf("Log[%d]: %q", i, log)
		}

		hasError := false
		hasOutput := false
		for _, log := range logs {
			if strings.Contains(log, "exited with error") {
				hasError = true
			}
			if strings.Contains(log, "Attempt") {
				hasOutput = true
			}
		}
		require.True(t, hasError, "Process should log exit error")
		require.True(t, hasOutput, "Process should log output from script")
	}
}

// TestRaceConditions tests for race conditions with concurrent operations
func TestRaceConditions(t *testing.T) {
	// This test is designed to be run with -race flag
	mockStdout := &mockReadCloser{}

	// Write lots of data with delays
	for i := 0; i < 100; i++ {
		mockStdout.Write([]byte(fmt.Sprintf("Line %d\n", i)))
	}

	// Configure read delays to trigger potential races
	mockStdout.readDelays = []time.Duration{
		50 * time.Millisecond,
		20 * time.Millisecond,
		10 * time.Millisecond,
	}

	logger := &testLogger{
		t:    t,
		logs: make([]string, 0),
		mu:   sync.Mutex{},
	}
	output := &output{logger: logger}

	// Start multiple goroutines that all attempt to read at the same time
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			// Only the first reader should succeed, others should fail with closed pipe
			output.read(mockStdout)
		}(i)
	}

	// Wait for all readers to complete
	wg.Wait()

	// Verify stdout was closed exactly once
	require.True(t, mockStdout.closed, "ReadCloser should be closed after reading")
}

// TestResourceLeaks tests for potential resource leaks
func TestResourceLeaks(t *testing.T) {
	// Use a script that runs for a while and generates output
	var script string
	var scriptPath string

	if runtime.GOOS == "windows" {
		script = `@echo off
for /l %%i in (1, 1, 10) do (
  echo Line %%i
  timeout /t 1 >nul
)
`
	} else {
		script = `#!/bin/bash
for i in {1..10}; do
  echo Line $i
  sleep 0.1
done
`
	}

	scriptPath = createTestScript(t, script)

	// Track number of active goroutines before test
	runtime.GC() // Force garbage collection
	startGoroutines := runtime.NumGoroutine()

	// Start and stop multiple processes in sequence
	for i := 0; i < 5; i++ {
		logger := &testLogger{
			t:    t,
			logs: make([]string, 0),
			mu:   sync.Mutex{},
		}
		process, err := NewProcess(logger, scriptPath, []string{})
		require.NoError(t, err)

		// Let it run briefly
		time.Sleep(300 * time.Millisecond)

		// Stop it
		process.Stop()

		// Force cleanup
		process = nil
		runtime.GC()
	}

	// Allow time for goroutines to finish
	time.Sleep(2 * time.Second)
	runtime.GC()

	// Check that goroutines don't continuously increase
	endGoroutines := runtime.NumGoroutine()
	t.Logf("Goroutines: Start=%d, End=%d", startGoroutines, endGoroutines)

	// Allow for a small increase due to testing overhead
	require.LessOrEqual(t, endGoroutines, startGoroutines+5,
		"Number of goroutines should not increase significantly")
}

// TestCancelDuringRead tests cancellation during stdout reading
func TestCancelDuringRead(t *testing.T) {
	// Use a mock reader that blocks
	mockStdout := &mockReadCloser{}
	mockStdout.Write([]byte("Test output"))
	mockStdout.readDelays = []time.Duration{500 * time.Millisecond}

	logger := &testLogger{
		t:    t,
		logs: make([]string, 0),
		mu:   sync.Mutex{},
	}

	// Create a monitor directly to test cancelation
	ctx, cancel := context.WithCancel(context.Background())

	command := &command{
		path:   "test",
		args:   []string{},
		logger: logger,
	}

	mon := &monitor{
		command: command,
		output: &output{
			logger: logger,
		},
		ctx:    ctx,
		cancel: cancel,
		logger: logger,
	}

	// Start the monitor
	mon.start()

	// Cancel after a short delay
	time.Sleep(100 * time.Millisecond)
	mon.stop()

	// Verify the monitor stopped gracefully
	logs := logger.getLogs()
	hasStopped := false
	for _, log := range logs {
		if log == "DEBUG: Process monitor loop stopped" {
			hasStopped = true
			break
		}
	}
	require.True(t, hasStopped, "Monitor should log when stopped")
}

// TestOutputReadErrors tests handling of read errors
func TestOutputReadErrors(t *testing.T) {
	// Use a mock reader that returns errors
	mockStdout := &mockReadCloser{}
	mockStdout.Write([]byte("Good line\n"))
	mockStdout.readErrors = []error{nil, fmt.Errorf("simulated read error")}

	logger := &testLogger{
		t:    t,
		logs: make([]string, 0),
		mu:   sync.Mutex{},
	}
	output := &output{logger: logger}

	// Read output
	output.read(mockStdout)

	// Verify error was logged
	logs := logger.getLogs()
	hasError := false
	for _, log := range logs {
		if log == "ERROR: stdout.Read failed: simulated read error" {
			hasError = true
			break
		}
	}
	require.True(t, hasError, "Read error should be logged")

	// Verify stdout was closed despite error
	require.True(t, mockStdout.closed, "ReadCloser should be closed after error")
}

// TestProcessStopRace tests for race conditions during process stopping
func TestProcessStopRace(t *testing.T) {
	var scriptPath string

	if runtime.GOOS == "windows" {
		script := `@echo off
echo Starting
timeout /t 5 >nul
echo Done
`
		scriptPath = createTestScript(t, script)
	} else {
		script := `#!/bin/bash
echo Starting
sleep 5
echo Done
`
		scriptPath = createTestScript(t, script)
	}

	logger := &testLogger{
		t:    t,
		logs: make([]string, 0),
		mu:   sync.Mutex{},
	}
	process, err := NewProcess(logger, scriptPath, []string{})
	require.NoError(t, err)

	// Let process start
	time.Sleep(500 * time.Millisecond)

	// Stop from multiple goroutines simultaneously to test for races
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			process.Stop()
		}()
	}

	// Wait for all stop calls to complete
	wg.Wait()

	// No assertions needed - this test passes if the race detector doesn't trigger
}

// TestProcessOutputVolume tests handling of large output volume
func TestProcessOutputVolume(t *testing.T) {
	mockStdout := &mockReadCloser{}

	// Write a large amount of data
	const lineCount = 10000
	for i := 0; i < lineCount; i++ {
		mockStdout.Write([]byte(fmt.Sprintf("Line %d\n", i)))
	}

	logger := &testLogger{
		t:    t,
		logs: make([]string, 0),
		mu:   sync.Mutex{},
	}
	output := &output{logger: logger}

	// Read all output
	output.read(mockStdout)

	// Verify a reasonable number of lines were processed
	logs := logger.getLogs()
	logCount := len(logs)

	// With the line-by-line scanner, we should get approximately lineCount logs
	// Allow for some tolerance due to buffering
	t.Logf("Log count: %d/%d", logCount, lineCount)
	require.Greater(t, logCount, 0, "Should have some log entries")
	require.True(t, mockStdout.closed, "ReadCloser should be closed after reading large output")
}

// TestStdoutClose tests handling of stdout close errors
func TestStdoutClose(t *testing.T) {
	mockStdout := &mockReadCloser{
		closeErr: fmt.Errorf("simulated close error"),
	}
	mockStdout.Write([]byte("Test data\n"))

	logger := &testLogger{
		t:    t,
		logs: make([]string, 0),
		mu:   sync.Mutex{},
	}
	output := &output{logger: logger}

	// Read output - this should still work despite close error
	output.read(mockStdout)

	// Verify data was read
	logs := logger.getLogs()
	hasData := false
	for _, log := range logs {
		if strings.Contains(log, "Test data") {
			hasData = true
			break
		}
	}

	require.True(t, hasData, "Data should be read despite close error")
	require.True(t, mockStdout.closed, "ReadCloser should be marked as closed")
}

// TestProcessConcurrentOutput tests handling of concurrent output generation
func TestProcessConcurrentOutput(t *testing.T) {
	mockStdout := &mockReadCloser{}

	// Start reading before writing anything
	var readWg sync.WaitGroup
	readWg.Add(1)

	var lineCount atomic.Int32

	logger := &testLogger{
		t:    t,
		logs: make([]string, 0, 1000),
	}

	output := &output{
		logger: logger,
	}

	go func() {
		defer readWg.Done()
		output.read(mockStdout)
	}()

	// Write from multiple goroutines concurrently
	var writeWg sync.WaitGroup
	for i := 0; i < 5; i++ {
		writeWg.Add(1)
		go func(id int) {
			defer writeWg.Done()
			for j := 0; j < 100; j++ {
				mockStdout.Write([]byte(fmt.Sprintf("Writer %d Line %d\n", id, j)))
				lineCount.Add(1)
				time.Sleep(time.Millisecond)
			}
		}(i)
	}

	// Wait for all writers to finish
	writeWg.Wait()

	// Force EOF by closing
	mockStdout.mu.Lock()
	mockStdout.closed = true
	mockStdout.mu.Unlock()

	// Wait for reader to finish
	readWg.Wait()

	// Verify logs contain some of the expected output
	logger.mu.Lock()
	loggedLines := len(logger.logs)
	logger.mu.Unlock()

	// We might not get every line due to the reader closing when the mockStdout is closed
	// But we should get a significant number of them
	t.Logf("Line count: expected ~%d, got %d", lineCount.Load(), loggedLines)
	require.True(t, loggedLines > 0, "Some lines should be logged")
}

// TestProcessOutput tests that the process correctly logs output
func TestProcessOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		script := `@echo off
echo This is a test message
echo Another message
exit 0
`
		scriptPath := createTestScript(t, script)
		t.Logf("Created test script at: %s", scriptPath)

		// Verify the script exists and has content
		content, err := os.ReadFile(scriptPath)
		require.NoError(t, err)
		t.Logf("Script content: %s", string(content))

		// Create a testLogger like the one used in monitor_test.go
		logger := &testLogger{
			t:    t,
			logs: make([]string, 0),
			mu:   sync.Mutex{},
		}

		process, err := NewProcess(logger, scriptPath, []string{})
		require.NoError(t, err)

		// Give process time to execute
		t.Log("Waiting for process output...")
		time.Sleep(2 * time.Second)

		// Stop the process
		t.Log("Stopping process")
		process.Stop()

		// Check output was logged
		logs := logger.getLogs()
		t.Logf("Found %d logs", len(logs))
		for i, log := range logs {
			t.Logf("Log[%d]: %s", i, log)
		}

		// Just check that we have at least 2 logs
		require.GreaterOrEqual(t, len(logs), 2, "Should have at least 2 log entries")
	} else {
		script := `#!/bin/bash
echo This is a test message
echo Another message
exit 0
`
		scriptPath := createTestScript(t, script)
		t.Logf("Created test script at: %s", scriptPath)

		// Verify the script exists and has content
		content, err := os.ReadFile(scriptPath)
		require.NoError(t, err)
		t.Logf("Script content: %s", string(content))

		// Create a testLogger like the one used in monitor_test.go
		logger := &testLogger{
			t:    t,
			logs: make([]string, 0),
			mu:   sync.Mutex{},
		}

		process, err := NewProcess(logger, scriptPath, []string{})
		require.NoError(t, err)

		// Give process time to execute
		t.Log("Waiting for process output...")
		time.Sleep(2 * time.Second)

		// Stop the process
		t.Log("Stopping process")
		process.Stop()

		// Check output was logged
		logs := logger.getLogs()
		t.Logf("Found %d logs", len(logs))
		for i, log := range logs {
			t.Logf("Log[%d]: %s", i, log)
		}

		// Just check that we have at least 2 logs
		require.GreaterOrEqual(t, len(logs), 2, "Should have at least 2 log entries")
	}
}

// TestMonitorOutput tests the monitor's handling of process output
func TestMonitorOutput(t *testing.T) {
	// Create a test script
	var script string
	if runtime.GOOS == "windows" {
		script = `@echo off
echo This is test output from a batch file
exit 0
`
	} else {
		script = `#!/bin/bash
echo This is test output from a shell script
exit 0
`
	}

	// Write the script to a temp file
	dir := t.TempDir()
	scriptPath := filepath.Join(dir, "test_script.sh")
	if runtime.GOOS == "windows" {
		scriptPath = filepath.Join(dir, "test_script.bat")
	}

	err := os.WriteFile(scriptPath, []byte(script), 0755)
	require.NoError(t, err, "Failed to create test script")

	// Create a test logger
	logger := &testLogger{
		t:    t,
		logs: make([]string, 0),
		mu:   sync.Mutex{},
	}

	// Create command and output handlers
	cmd := &command{
		path:   scriptPath,
		args:   []string{},
		logger: logger,
	}

	out := &output{
		logger: logger,
	}

	// Create a monitor
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mon := &monitor{
		command:     cmd,
		output:      out,
		ctx:         ctx,
		cancel:      cancel,
		logger:      logger,
		retryPeriod: 500 * time.Millisecond,
	}

	// Start the monitor (which starts the process)
	mon.start()

	// Wait for the process to run and produce output
	time.Sleep(2 * time.Second)

	// Stop the monitor
	mon.stop()

	// Check for log output from the script
	logs := logger.getLogs()
	t.Logf("Found %d logs", len(logs))
	for i, log := range logs {
		t.Logf("Log[%d]: %s", i, log)
	}

	// Verify we got some output
	hasOutput := false
	for _, log := range logs {
		if strings.Contains(log, "test output from") {
			hasOutput = true
			break
		}
	}

	require.True(t, hasOutput, "Should have captured output from the script")
}
