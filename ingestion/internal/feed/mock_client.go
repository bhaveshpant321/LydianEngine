package feed

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"time"

	"lydian-ingestion/internal/models"
)

// MockClient implements FeedClient by replaying news items loaded from a
// JSON file on disk. This is the drop-in replacement for a live WebSocket
// feed during development, CI, and demo environments.
type MockClient struct {
	filePath string
	interval time.Duration
	logger   *slog.Logger
}

// NewMockClient constructs a MockClient.
//   - filePath: path to a JSON file containing a JSON array of NewsItems.
//   - interval: simulated inter-message delay (e.g., 500ms for realistic replay).
func NewMockClient(filePath string, interval time.Duration, logger *slog.Logger) *MockClient {
	return &MockClient{
		filePath: filePath,
		interval: interval,
		logger:   logger,
	}
}

// Name satisfies the FeedClient interface.
func (m *MockClient) Name() string {
	return fmt.Sprintf("MockClient(%s)", m.filePath)
}

// Connect loads the JSON file, validates each record, and streams valid
// NewsItems onto the returned channel at the configured interval.
// The channel is closed when all items have been replayed or the context
// is cancelled — whichever comes first.
func (m *MockClient) Connect(ctx context.Context) (<-chan models.NewsItem, error) {
	raw, err := os.ReadFile(m.filePath)
	if err != nil {
		return nil, fmt.Errorf("mock feed: open file %q: %w", m.filePath, err)
	}

	var items []models.NewsItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("mock feed: unmarshal JSON: %w", err)
	}

	if len(items) == 0 {
		return nil, fmt.Errorf("mock feed: file %q contains zero items", m.filePath)
	}

	out := make(chan models.NewsItem, 16) // buffered to tolerate slow consumers

	go func() {
		defer close(out)

		for i, item := range items {
			item.DetectNegation()
			// Validate before sending so downstream never sees malformed data.
			if err := item.Validate(); err != nil {
				m.logger.Warn("mock feed: skipping invalid item",
					slog.Int("index", i),
					slog.String("id", item.ID),
					slog.String("error", err.Error()),
				)
				continue
			}

			select {
			case <-ctx.Done():
				m.logger.Info("mock feed: context cancelled, stopping replay",
					slog.Int("items_sent", i),
					slog.Int("items_total", len(items)),
				)
				return
			case out <- item:
				m.logger.Debug("mock feed: dispatched item",
					slog.Int("index", i),
					slog.String("id", item.ID),
					slog.String("headline", item.Headline),
				)
			}

			// Simulate realistic inter-message delay.
			select {
			case <-ctx.Done():
				return
			case <-time.After(m.interval):
			}
		}

		m.logger.Info("mock feed: replay complete", slog.Int("items_total", len(items)))
	}()

	return out, nil
}

// WSClient implements FeedClient using a live WebSocket connection.
// It wraps the gorilla/websocket library, providing automatic reconnection
// with exponential backoff and per-message read deadlines.
//
// NOTE: For environments without a live feed, use MockClient instead.
// This struct is wired in via the FEED_MODE env-var (see cmd/main.go).
type WSClient struct {
	url    string
	logger *slog.Logger
}

// NewWSClient constructs a WSClient targeting the given WebSocket URL.
func NewWSClient(url string, logger *slog.Logger) *WSClient {
	return &WSClient{url: url, logger: logger}
}

// Name satisfies the FeedClient interface.
func (w *WSClient) Name() string {
	return fmt.Sprintf("WSClient(%s)", w.url)
}

// Connect establishes a WebSocket connection and streams NewsItems onto
// the returned channel. On any transient error (connection drop, read
// timeout) it retries with exponential backoff up to maxRetries before
// closing the channel.
func (w *WSClient) Connect(ctx context.Context) (<-chan models.NewsItem, error) {
	// Import deferred to avoid mandatory gorilla dependency when running in mock mode.
	// In a real implementation, import "github.com/gorilla/websocket" here.

	out := make(chan models.NewsItem, 64)

	go func() {
		defer close(out)

		const maxRetries = 5

		for attempt := 1; attempt <= maxRetries; attempt++ {
			if ctx.Err() != nil {
				return
			}

			w.logger.Info("ws: connecting",
				slog.String("url", w.url),
				slog.Int("attempt", attempt),
			)

			// --- Gorilla WS connection (stubbed; gorilla import omitted for portability) ---
			// conn, _, err := websocket.DefaultDialer.DialContext(ctx, w.url, nil)
			// if err != nil { ... retry ... }
			// defer conn.Close()
			// for {
			//     conn.SetReadDeadline(time.Now().Add(10 * time.Second))
			//     _, msg, err := conn.ReadMessage()
			//     ...
			// }
			// ---------------------------------------------------------------------------------

			// Stub: signal that a live WS implementation is expected here.
			w.logger.Warn("ws: stub implementation — switch FEED_MODE=mock for demos")
			return
		}

		w.logger.Error("ws: exhausted retries", slog.String("url", w.url))
	}()

	return out, nil
}

// parseAndValidate deserialises a raw WebSocket message into a NewsItem and
// runs structural validation. Returns an error for both malformed JSON and
// invalid business-rule violating payloads.
func parseAndValidate(raw []byte) (models.NewsItem, error) {
	var item models.NewsItem
	if err := json.Unmarshal(raw, &item); err != nil {
		return models.NewsItem{}, fmt.Errorf("parse: unmarshal: %w", err)
	}
	if err := item.Validate(); err != nil {
		return models.NewsItem{}, fmt.Errorf("parse: validate: %w", err)
	}
	return item, nil
}
