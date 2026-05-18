package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"qwen_public_gateway/internal/bridge"
)

type ServerConfig struct {
	Bridge       *bridge.Client
	Logger       *log.Logger
	MaxBodyBytes int64
}

type Server struct {
	bridge       *bridge.Client
	logger       *log.Logger
	maxBodyBytes int64
	mux          *http.ServeMux
}

type requestPayload struct {
	Message       string         `json:"message,omitempty"`
	SelfCheck     bool           `json:"self_check,omitempty"`
	RevisionNote  string         `json:"revision_note,omitempty"`
	WorkflowState map[string]any `json:"workflow_state,omitempty"`
}

type responseEnvelope struct {
	OK      bool           `json:"ok"`
	Mode    string         `json:"mode"`
	Elapsed string         `json:"elapsed,omitempty"`
	Result  any            `json:"result,omitempty"`
	Error   string         `json:"error,omitempty"`
	Service string         `json:"service"`
	Time    string         `json:"time"`
	Meta    map[string]any `json:"meta,omitempty"`
}

func NewServer(config ServerConfig) *Server {
	if config.MaxBodyBytes <= 0 {
		config.MaxBodyBytes = 1 << 20
	}
	server := &Server{
		bridge:       config.Bridge,
		logger:       config.Logger,
		maxBodyBytes: config.MaxBodyBytes,
		mux:          http.NewServeMux(),
	}
	server.routes()
	return server
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *Server) routes() {
	s.mux.HandleFunc("/healthz", s.handleHealth)
	s.mux.HandleFunc("/readyz", s.handleReady)
	s.mux.HandleFunc("/v1/preview", s.handlePreview)
	s.mux.HandleFunc("/v1/generate", s.handleGenerate)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":      true,
		"service": "go-public-gateway",
		"time":    time.Now().Format(time.RFC3339Nano),
	})
}

func (s *Server) handleReady(w http.ResponseWriter, _ *http.Request) {
	if s.bridge == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"ok":    false,
			"error": "bridge client is not configured",
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok": true,
	})
}

func (s *Server) handlePreview(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	s.handleMode(w, r, "preview")
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	s.handleMode(w, r, "generate")
}

func (s *Server) handleMode(w http.ResponseWriter, r *http.Request, mode string) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if s.bridge == nil {
		writeError(w, http.StatusServiceUnavailable, "bridge client is not configured")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, s.maxBodyBytes)
	defer r.Body.Close()

	start := time.Now()
	payload, err := decodeRequest(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	ctx := r.Context()
	if strings.TrimSpace(payload.RevisionNote) != "" || len(payload.WorkflowState) > 0 {
		if strings.TrimSpace(payload.RevisionNote) == "" {
			writeError(w, http.StatusBadRequest, "revision_note cannot be empty")
			return
		}
		if len(payload.WorkflowState) == 0 {
			writeError(w, http.StatusBadRequest, "workflow_state cannot be empty")
			return
		}

		result, err := s.bridge.Continue(ctx, bridge.Request{
			RevisionNote:  payload.RevisionNote,
			WorkflowState: payload.WorkflowState,
		})
		if err != nil {
			status := http.StatusBadGateway
			if errors.Is(err, context.DeadlineExceeded) {
				status = http.StatusGatewayTimeout
			}
			writeJSON(w, status, responseEnvelope{
				OK:      false,
				Mode:    "continue",
				Error:   err.Error(),
				Service: "go-public-gateway",
				Time:    time.Now().Format(time.RFC3339Nano),
				Meta: map[string]any{
					"elapsed_ms": time.Since(start).Milliseconds(),
				},
			})
			return
		}

		writeJSON(w, http.StatusOK, responseEnvelope{
			OK:      true,
			Mode:    mode,
			Elapsed: time.Since(start).String(),
			Result:  result,
			Service: "go-public-gateway",
			Time:    time.Now().Format(time.RFC3339Nano),
		})
		return
	}

	if strings.TrimSpace(payload.Message) == "" {
		writeError(w, http.StatusBadRequest, "message cannot be empty")
		return
	}

	result, err := s.bridge.Generate(ctx, bridge.Request{Message: payload.Message, SelfCheck: payload.SelfCheck})
	if err != nil {
		status := http.StatusBadGateway
		if errors.Is(err, context.DeadlineExceeded) {
			status = http.StatusGatewayTimeout
		}
		writeJSON(w, status, responseEnvelope{
			OK:      false,
			Mode:    "generate",
			Error:   err.Error(),
			Service: "go-public-gateway",
			Time:    time.Now().Format(time.RFC3339Nano),
			Meta: map[string]any{
				"elapsed_ms": time.Since(start).Milliseconds(),
			},
		})
		return
	}

	writeJSON(w, http.StatusOK, responseEnvelope{
		OK:      true,
		Mode:    mode,
		Elapsed: time.Since(start).String(),
		Result:  result,
		Service: "go-public-gateway",
		Time:    time.Now().Format(time.RFC3339Nano),
	})
}

func decodeRequest(reader io.Reader) (requestPayload, error) {
	var payload requestPayload
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&payload); err != nil {
		return requestPayload{}, err
	}
	return payload, nil
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]any{
		"ok":    false,
		"error": message,
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
