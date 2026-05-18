package bridge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"time"
)

type Config struct {
	PythonBinary string
	ScriptPath   string
	Timeout      time.Duration
}

type Client struct {
	config Config
	logger *log.Logger
}

type Request struct {
	Message      string                 `json:"message,omitempty"`
	SelfCheck    bool                   `json:"self_check,omitempty"`
	RevisionNote string                 `json:"revision_note,omitempty"`
	WorkflowState map[string]any        `json:"workflow_state,omitempty"`
}

type Response struct {
	OK        bool           `json:"ok"`
	Mode      string         `json:"mode"`
	Message   string         `json:"message"`
	SelfCheck bool           `json:"self_check"`
	Trace     map[string]any `json:"trace,omitempty"`
	Request   map[string]any `json:"request,omitempty"`
	Output    string         `json:"output,omitempty"`
	Review    map[string]any `json:"review,omitempty"`
	Error     string         `json:"error,omitempty"`
	Meta      map[string]any `json:"meta,omitempty"`
}

func NewPythonBridge(config Config, logger *log.Logger) *Client {
	if config.Timeout <= 0 {
		config.Timeout = 5 * time.Minute
	}
	return &Client{config: config, logger: logger}
}

func (c *Client) Preview(ctx context.Context, request Request) (Response, error) {
	return c.invoke(ctx, "preview", request)
}

func (c *Client) Generate(ctx context.Context, request Request) (Response, error) {
	return c.invoke(ctx, "generate", request)
}

func (c *Client) Continue(ctx context.Context, request Request) (Response, error) {
	return c.invoke(ctx, "continue", request)
}

func (c *Client) invoke(ctx context.Context, mode string, request Request) (Response, error) {
	if c == nil {
		return Response{}, fmt.Errorf("bridge client is nil")
	}
	if mode != "continue" && request.Message == "" {
		return Response{}, fmt.Errorf("message cannot be empty")
	}

	payload := map[string]any{
		"mode":          mode,
		"message":       request.Message,
		"self_check":    request.SelfCheck,
		"revision_note": request.RevisionNote,
		"workflow_state": request.WorkflowState,
	}
	input, err := json.Marshal(payload)
	if err != nil {
		return Response{}, fmt.Errorf("marshal bridge request: %w", err)
	}

	runCtx, cancel := context.WithTimeout(ctx, c.config.Timeout)
	defer cancel()

	command := exec.CommandContext(runCtx, c.config.PythonBinary, c.config.ScriptPath)
	command.Stdin = bytes.NewReader(input)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr

	if err := command.Run(); err != nil {
		if stderr.Len() > 0 && c.logger != nil {
			c.logger.Printf("python bridge stderr: %s", stderr.String())
		}
		if runCtx.Err() == context.DeadlineExceeded {
			return Response{}, context.DeadlineExceeded
		}
		return Response{}, fmt.Errorf("bridge failed: %w", err)
	}

	var response Response
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		return Response{}, fmt.Errorf("decode bridge response: %w", err)
	}
	if !response.OK {
		if response.Error == "" {
			response.Error = "bridge returned a non-ok response"
		}
		return response, fmt.Errorf("%s", response.Error)
	}
	return response, nil
}
