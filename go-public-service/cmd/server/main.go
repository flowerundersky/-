package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"qwen_public_gateway/internal/bridge"
	"qwen_public_gateway/internal/httpapi"
)

func main() {
	logger := log.New(os.Stdout, "go-gateway: ", log.LstdFlags|log.Lmicroseconds)

	workDir, err := os.Getwd()
	if err != nil {
		logger.Fatalf("get working directory: %v", err)
	}

	bridgeScript := os.Getenv("BRIDGE_SCRIPT")
	if bridgeScript == "" {
		bridgeScript = filepath.Join(workDir, "python_bridge.py")
	}

	pythonBinary := resolvePythonBinary(workDir)

	requestTimeout := 5 * time.Minute
	if rawTimeout := os.Getenv("REQUEST_TIMEOUT_SECONDS"); rawTimeout != "" {
		if parsed, parseErr := time.ParseDuration(rawTimeout + "s"); parseErr == nil {
			requestTimeout = parsed
		}
	}

	client := bridge.NewPythonBridge(bridge.Config{
		PythonBinary: pythonBinary,
		ScriptPath:   bridgeScript,
		Timeout:      requestTimeout,
	}, logger)

	handler := httpapi.NewServer(httpapi.ServerConfig{
		Bridge:       client,
		Logger:       logger,
		MaxBodyBytes: 1 << 20,
	})

	server := &http.Server{
		Addr:         addrFromEnv(),
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 5 * time.Minute,
		IdleTimeout:  60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		logger.Printf("listening on %s", server.Addr)
		if serveErr := server.ListenAndServe(); serveErr != nil && serveErr != http.ErrServerClosed {
			logger.Fatalf("server error: %v", serveErr)
		}
	}()

	<-ctx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Fatalf("shutdown error: %v", err)
	}
}

func addrFromEnv() string {
	if addr := os.Getenv("ADDR"); addr != "" {
		return addr
	}
	return ":8088"
}

func resolvePythonBinary(workDir string) string {
	if pythonBinary := os.Getenv("PYTHON_BIN"); pythonBinary != "" {
		return pythonBinary
	}

	venvPython := filepath.Clean(filepath.Join(workDir, "..", ".venv", "bin", "python"))
	if info, err := os.Stat(venvPython); err == nil && !info.IsDir() {
		return venvPython
	}

	return "python3"
}
