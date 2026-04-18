// Command lydian-ingestion is the main entry point for the Go ingestion
// microservice. It wires together all internal packages using dependency
// injection, configures structured logging, and orchestrates a graceful
// shutdown on SIGINT / SIGTERM.
//
// Environment variables (all optional with sane defaults):
//
//	FEED_MODE       = "mock" | "websocket" (default: "mock")
//	MOCK_FILE       = path to JSON news feed file (default: "testdata/mock_news_feed.json")
//	MOCK_INTERVAL   = inter-message delay, e.g. "500ms" (default: "500ms")
//	WS_URL          = WebSocket URL for live mode (default: "ws://localhost:8765/feed")
//	TARGET_URL      = FastAPI ingest endpoint (default: "http://sentinel:8000/ingest")
//	BUFFER_CAPACITY = RingBuffer size (default: "256")
//	LOG_LEVEL       = "debug" | "info" | "warn" | "error" (default: "info")
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"lydian-ingestion/internal/buffer"
	"lydian-ingestion/internal/dispatcher"
	"lydian-ingestion/internal/feed"
)

func main() {
	// ── Logger ──────────────────────────────────────────────────────────────
	logger := newLogger(getenv("LOG_LEVEL", "info"))
	logger.Info("lydian-ingestion starting")

	// ── Configuration from environment ──────────────────────────────────────
	feedMode := getenv("FEED_MODE", "mock")
	mockFile := getenv("MOCK_FILE", "testdata/mock_news_feed.json")
	mockIntervalStr := getenv("MOCK_INTERVAL", "500ms")
	wsURL := getenv("WS_URL", "ws://localhost:8765/feed")
	targetURL := getenv("TARGET_URL", "http://sentinel:8000/ingest")
	bufCapStr := getenv("BUFFER_CAPACITY", "256")

	mockInterval, err := time.ParseDuration(mockIntervalStr)
	if err != nil {
		logger.Error("invalid MOCK_INTERVAL", slog.String("value", mockIntervalStr))
		os.Exit(1)
	}

	bufCap, err := strconv.Atoi(bufCapStr)
	if err != nil || bufCap <= 0 {
		logger.Error("invalid BUFFER_CAPACITY", slog.String("value", bufCapStr))
		os.Exit(1)
	}

	// ── Dependency graph ─────────────────────────────────────────────────────
	//
	// FeedClient → RingBuffer → Dispatcher → Python API
	//
	// Each component holds a reference only to what it directly needs.

	var client feed.FeedClient
	switch feedMode {
	case "mock":
		client = feed.NewMockClient(mockFile, mockInterval, logger)
	case "websocket":
		client = feed.NewWSClient(wsURL, logger)
	default:
		logger.Error("unknown FEED_MODE", slog.String("mode", feedMode))
		os.Exit(1)
	}

	buf := buffer.NewRingBuffer(bufCap, logger)

	disp := dispatcher.New(buf, dispatcher.Config{
		TargetURL:      targetURL,
		RequestTimeout: 5 * time.Second,
	}, logger)

	// ── Graceful shutdown ────────────────────────────────────────────────────
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	logger.Info("connecting to feed",
		slog.String("client", client.Name()),
		slog.String("mode", feedMode),
	)

	items, err := client.Connect(ctx)
	if err != nil {
		logger.Error("feed connect failed", slog.String("error", err.Error()))
		os.Exit(1)
	}

	// Launch dispatcher in its own goroutine.
	go disp.Run(ctx)

	// Fan items from the feed channel into the ring buffer.
	// This goroutine exits when items is closed (feed exhausted or ctx cancelled).
	for item := range items {
		buf.Push(item)
	}

	// Feed is done (either exhausted or context cancelled).
	// Allow the dispatcher to drain what's in the buffer within 5s.
	drainCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	logger.Info("feed complete; draining buffer",
		slog.Int("buffer_len", buf.Len()),
	)

	for buf.Len() > 0 {
		select {
		case <-drainCtx.Done():
			logger.Warn("drain timeout reached — exiting with non-empty buffer",
				slog.Int("remaining", buf.Len()),
			)
			goto done
		case <-time.After(50 * time.Millisecond):
		}
	}

done:
	logger.Info("lydian-ingestion exiting cleanly",
		slog.Int64("total_dropped", buf.DroppedCount()),
	)
}

// getenv returns the value of the named environment variable, or fallback
// if the variable is not set or is empty.
func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// newLogger builds a structured JSON logger for production or a text logger
// for development based on the requested level string.
func newLogger(level string) *slog.Logger {
	var l slog.Level
	switch level {
	case "debug":
		l = slog.LevelDebug
	case "warn":
		l = slog.LevelWarn
	case "error":
		l = slog.LevelError
	default:
		l = slog.LevelInfo
	}
	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: l})
	return slog.New(handler)
}
