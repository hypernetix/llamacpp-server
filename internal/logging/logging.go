package logging

import (
	"fmt"
	"strings"
)

type SprintfLogger interface {
	Level() string
	Debugf(msg string, args ...interface{})
	Infof(msg string, args ...interface{})
	Warnf(msg string, args ...interface{})
	Errorf(msg string, args ...interface{})
	With(args ...interface{}) SprintfLogger
}

func NewSprintfLogger() SprintfLogger {
	return &sprintfLogger{}
}

type sprintfLogger struct {
	prefix string
}

func (l *sprintfLogger) Level() string {
	return "debug"
}

func (l *sprintfLogger) logf(msg string, args ...interface{}) {
	str := fmt.Sprintf(msg, args...)
	if l.prefix != "" {
		str = fmt.Sprintf("%s | %s", l.prefix, str)
	}
	if !strings.HasSuffix(str, "\n") {
		fmt.Println(str)
	} else {
		fmt.Print(str)
	}
}

func (l *sprintfLogger) Debugf(msg string, args ...interface{}) {
	l.logf(msg, args...)
}

func (l *sprintfLogger) Infof(msg string, args ...interface{}) {
	l.logf(msg, args...)
}

func (l *sprintfLogger) Warnf(msg string, args ...interface{}) {
	l.logf(msg, args...)
}

func (l *sprintfLogger) Errorf(msg string, args ...interface{}) {
	l.logf(msg, args...)
}

func (l *sprintfLogger) With(args ...interface{}) SprintfLogger {
	var parts []string
	for i := 0; i < len(args); i = i + 2 {
		argK := args[i]
		argV := interface{}("(nil)")
		if i+1 < len(args) {
			argV = args[i+1]
		}
		parts = append(parts, fmt.Sprintf("%v: %v", argK, argV))
	}
	return &sprintfLogger{prefix: strings.Join(parts, ", ")}
}
