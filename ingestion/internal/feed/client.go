// Package feed defines the FeedClient interface and its implementations.
// The interface/implementation split allows callers to swap live WebSocket
// feeds for deterministic mock feeds without changing any business logic.
package feed

import (
	"context"

	"lydian-ingestion/internal/models"
)

// FeedClient is the contract that both live and mock feed sources must satisfy.
// Implementations must be safe for concurrent use by a single goroutine (the
// caller is responsible for consuming the channel before calling Connect again).
type FeedClient interface {
	// Connect establishes the data source and begins streaming NewsItems onto
	// the returned channel. The channel is closed when the context is cancelled,
	// the source is exhausted, or an unrecoverable error occurs.
	//
	// Implementations MUST:
	//   - never send a nil NewsItem on the channel.
	//   - close the channel upon exit (so range loops terminate naturally).
	//   - respect ctx.Done() and return promptly when cancelled.
	Connect(ctx context.Context) (<-chan models.NewsItem, error)

	// Name returns a human-readable label for this feed source, used in logs.
	Name() string
}
