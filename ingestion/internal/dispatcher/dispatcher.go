// Package dispatcher reads NewsItems from a RingBuffer and forwards them to
// the downstream Python Sentinel API via HTTP POST. It implements exponential
// backoff with jitter on transient failures and respects context cancellation
// for clean shutdown.
package dispatcher

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"time"

	"lydian-ingestion/internal/buffer"
	"lydian-ingestion/internal/models"
)

const (
	// maxRetries is the number of POST attempts before an item is abandoned.
	maxRetries = 4
	// baseBackoff is the starting delay for exponential backoff.
	baseBackoff = 100 * time.Millisecond
	// maxBackoff caps the delay to avoid runaway waits.
	maxBackoff = 5 * time.Second
)

// Dispatcher reads from a RingBuffer and POSTs each item to a configured
// HTTP endpoint. It uses a shared http.Client with connection pooling for
// efficiency and injects all dependencies, avoiding global state.
type Dispatcher struct {
	buf        *buffer.RingBuffer
	targetURL  string
	httpClient *http.Client
	logger     *slog.Logger
}

// Config holds the configuration for a Dispatcher.
type Config struct {
	TargetURL      string
	RequestTimeout time.Duration
}

// New constructs a Dispatcher. The caller owns the RingBuffer lifecycle.
func New(buf *buffer.RingBuffer, cfg Config, logger *slog.Logger) *Dispatcher {
	if cfg.RequestTimeout == 0 {
		cfg.RequestTimeout = 5 * time.Second
	}
	return &Dispatcher{
		buf:       buf,
		targetURL: cfg.TargetURL,
		httpClient: &http.Client{
			Timeout: cfg.RequestTimeout,
			// Connection pooling via default transport.
			Transport: http.DefaultTransport,
		},
		logger: logger,
	}
}

// Run starts the dispatch loop. It blocks until ctx is cancelled. Designed
// to be launched in its own goroutine:
//
//	go dispatcher.Run(ctx)
func (d *Dispatcher) Run(ctx context.Context) {
	d.logger.Info("dispatcher: started", slog.String("target", d.targetURL))

	for {
		select {
		case <-ctx.Done():
			d.logger.Info("dispatcher: shutting down",
				slog.Int("buffer_remaining", d.buf.Len()),
				slog.Int64("total_dropped", d.buf.DroppedCount()),
			)
			return
		case item, ok := <-d.buf.Pop():
			if !ok {
				d.logger.Info("dispatcher: buffer channel closed, exiting")
				return
			}
			d.dispatchWithRetry(ctx, item)
		}
	}
}

// dispatchWithRetry serialises item to JSON and POSTs it to the target URL,
// retrying up to maxRetries times with exponential backoff + jitter.
// Transient HTTP errors (5xx, timeouts) trigger retries; 4xx errors are
// treated as permanent failures and logged without retrying.
func (d *Dispatcher) dispatchWithRetry(ctx context.Context, item models.NewsItem) {
	payload, err := json.Marshal(item)
	if err != nil {
		// This should not happen for well-typed models; log and skip.
		d.logger.Error("dispatcher: marshal error — item abandoned",
			slog.String("id", item.ID),
			slog.String("error", err.Error()),
		)
		return
	}

	for attempt := 1; attempt <= maxRetries; attempt++ {
		if ctx.Err() != nil {
			return // honour shutdown signal between retries
		}

		err := d.post(ctx, payload)
		if err == nil {
			d.logger.Info("dispatcher: item delivered",
				slog.String("id", item.ID),
				slog.Int("attempt", attempt),
			)
			return
		}

		if attempt == maxRetries {
			d.logger.Error("dispatcher: max retries exhausted — item abandoned",
				slog.String("id", item.ID),
				slog.String("error", err.Error()),
			)
			return
		}

		delay := backoffDelay(attempt)
		d.logger.Warn("dispatcher: transient error, will retry",
			slog.String("id", item.ID),
			slog.Int("attempt", attempt),
			slog.String("error", err.Error()),
			slog.Duration("retry_in", delay),
		)

		select {
		case <-ctx.Done():
			return
		case <-time.After(delay):
		}
	}
}

// post serialises and sends a single HTTP POST. Returns an error for any
// non-2xx response or network-level failure.
func (d *Dispatcher) post(ctx context.Context, payload []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, d.targetURL, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "lydian-ingestion/1.0")

	resp, err := d.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("http post: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusUnprocessableEntity {
		// 4xx: client error — retrying will not help.
		return fmt.Errorf("permanent error: HTTP %d — item will be abandoned", resp.StatusCode)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// 5xx or unexpected: treat as transient.
		return fmt.Errorf("transient error: HTTP %d", resp.StatusCode)
	}
	return nil
}

// backoffDelay returns the delay for the given attempt using exponential
// backoff with ±25% jitter to avoid thundering-herd on the downstream API.
func backoffDelay(attempt int) time.Duration {
	exp := math.Pow(2, float64(attempt-1))
	base := time.Duration(float64(baseBackoff) * exp)
	if base > maxBackoff {
		base = maxBackoff
	}
	// Add ±25% jitter.
	jitter := time.Duration(rand.Float64()*0.5*float64(base)) - base/4
	return base + jitter
}
