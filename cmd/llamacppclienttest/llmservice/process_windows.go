//go:build windows

package llmservice

import (
	"os/exec"
	"syscall"
)

// setupProcessAttributes configures Windows-specific process attributes
func setupProcessAttributes(cmd *exec.Cmd) {
	// On Windows, just use the default process creation
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}

	// Set the cancel function for the command
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}

		sendCtrlBreak(cmd.Process.Pid)

		// Return nil to allow the Wait() flow to continue
		return nil
	}
}

// sendCtrlBreak sends a Ctrl-Break event to the process with id pid
func sendCtrlBreak(pid int) error {
	d, e := syscall.LoadDLL("kernel32.dll")
	if e != nil {
		return e
	}
	p, e := d.FindProc("GenerateConsoleCtrlEvent")
	if e != nil {
		return e
	}
	r, _, e := p.Call(uintptr(syscall.CTRL_BREAK_EVENT), uintptr(pid))
	if r == 0 {
		return e // syscall.GetLastError()
	}
	return nil
}
