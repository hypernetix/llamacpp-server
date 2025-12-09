package llmservice

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/hypernetix/llamacpp_server/internal/logging"
)

type Process interface {
	Stop()
}

type process struct {
	monitor *monitor
}

func (p *process) Stop() {
	p.monitor.stop()
}

func NewProcess(logger logging.SprintfLogger, path string, args []string) (Process, error) {
	ctx, cancel := context.WithCancel(context.Background())

	command := &command{
		path:   path,
		args:   args,
		logger: logger.With("module", "process.command"),
	}
	output := &output{
		logger: logger.With("module", "process.output"),
	}
	monitor := &monitor{
		command:     command,
		output:      output,
		ctx:         ctx,
		cancel:      cancel,
		logger:      logger.With("module", "process.monitor"),
		retryPeriod: 10 * time.Second, // Default retry period
	}

	monitor.start()

	return &process{
		monitor: monitor,
	}, nil
}

type monitor struct {
	command     *command
	output      *output
	logger      logging.SprintfLogger
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
	retryPeriod time.Duration
}

func (l *monitor) loop() {
	defer l.wg.Done()

	l.logger.Debugf("Starting process monitor loop")
	defer func() {
		l.logger.Debugf("Process monitor loop stopped")
	}()

	retryPeriod := 10 * time.Second
	if l.retryPeriod > 0 {
		retryPeriod = l.retryPeriod
	}

	for loopCount := 1; ; loopCount++ {
		l.logger.Debugf("Starting process (iteration %d): %s %v", loopCount, l.command.path, l.command.args)
		cmd, stdout, err := l.command.start(l.ctx)
		if err != nil {
			l.logger.Errorf("Failed to start process: %v, retrying in %v seconds", err, retryPeriod)

			select {
			case <-l.ctx.Done():
				l.logger.Debugf("Cancelled while waiting for process to start")
				return
			case <-time.After(retryPeriod):
				continue
			}
		}

		l.logger.Debugf("Process started, PID: %d, setting up output reader", cmd.Process.Pid)
		var outputWg sync.WaitGroup
		outputWg.Add(1)
		go func() {
			defer func() {
				outputWg.Done()
				l.logger.Debugf("Output reader finished")
			}()
			l.logger.Debugf("Output reader starting")
			l.output.read(stdout)
		}()

		l.logger.Debugf("Waiting for process (PID: %d) to exit", cmd.Process.Pid)
		err = cmd.Wait()
		l.logger.Debugf("Process exited (PID: %d), waiting for output reader", cmd.Process.Pid)
		// Wait for reader to finish after process exits
		outputWg.Wait()

		if err != nil {
			l.logger.Errorf("Process exited with error: %v", err)
		} else {
			l.logger.Debugf("Process exited successfully")
		}

		select {
		case <-l.ctx.Done():
			l.logger.Debugf("Cancelled while waiting for process to exit")
			return
		default:
			l.logger.Debugf("Process exited, will restart")
		}
	}
}

func (l *monitor) start() {
	l.wg.Add(1)
	go l.loop()
}

func (l *monitor) stop() {
	l.logger.Debugf("Stopping monitor...")
	l.cancel()
	l.logger.Debugf("Waiting for monitor to stop...")
	l.wg.Wait()
	l.logger.Debugf("Monitor stopped")
}

type output struct {
	logger logging.SprintfLogger
}

func (r *output) read(stdout io.ReadCloser) {
	defer stdout.Close()
	lineCount := 0
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		lineCount++
		text := scanner.Text()
		r.logger.Debugf(text)
	}
	if err := scanner.Err(); err != nil {
		r.logger.Errorf("stdout.Read failed: %v", err)
	}
	r.logger.Debugf("Read %d lines from process", lineCount)
}

type command struct {
	path   string
	args   []string
	logger logging.SprintfLogger
}

func (r *command) start(ctx context.Context) (*exec.Cmd, io.ReadCloser, error) {
	absPath, err := filepath.Abs(r.path)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get absolute path: %v", err)
	}

	r.logger.Debugf("Running process: '%s', args: %v", absPath, r.args)

	// Make sure the process is executable
	err = os.Chmod(absPath, 0700)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to make process executable: %v", err)
	}

	workDir := filepath.Dir(absPath)

	r.logger.Debugf("Working directory: '%s', abs path: '%s'", workDir, absPath)

	cmd := exec.CommandContext(ctx, absPath, r.args...)
	cmd.Dir = workDir
	cmd.Env = os.Environ()

	// Platform-specific setup is handled in process_windows.go or process_unix.go
	setupProcessAttributes(cmd)

	// wait after sending the interrupt signal, before sending the kill signal
	cmd.WaitDelay = 30 * time.Second
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("exec.CommandContext failed: %v", err)
	}
	cmd.Stderr = cmd.Stdout

	err = cmd.Start()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to start the app: %v", err)
	}

	return cmd, stdout, nil
}
