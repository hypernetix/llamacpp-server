//go:build !windows

package llmservice

import (
	"os/exec"
	"syscall"
)

// setupProcessAttributes configures Unix-specific process attributes
func setupProcessAttributes(cmd *exec.Cmd) {
	// On Unix, create a new process group that we can signal as a whole
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}

	// Set the cancel function for the command
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		// On Unix, send SIGTERM to the process group
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
	}
}
