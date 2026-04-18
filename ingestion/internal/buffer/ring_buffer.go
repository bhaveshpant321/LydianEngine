// Package buffer provides a thread-safe, channel-backed ring buffer for
// NewsItems. It is designed for the producer-consumer pattern where a fast
// feed goroutine publishes items and a slower dispatcher goroutine consumes
// them. On overflow the oldest unconsumed item is silently dropped, which
// matches the semantics of a real-time data feed (stale data has low value).
package buffer

import (
	"log/slog"
	"sync/atomic"

	"lydian-ingestion/internal/models"
)

// RingBuffer is a fixed-capacity, channel-backed FIFO queue. It wraps a
// Go channel to achieve thread-safety without explicit mutexes. Capacity
// must be a positive integer; the zero value is invalid.
type RingBuffer struct {
	ch       chan models.NewsItem
	dropped  atomic.Int64 // monotonic drop counter for observability
	capacity int
	logger   *slog.Logger
}

// NewRingBuffer creates a RingBuffer with the given capacity. The underlying
// channel provides the synchronisation guarantee; no additional locking is
// needed for Push or Pop callers.
func NewRingBuffer(capacity int, logger *slog.Logger) *RingBuffer {
	if capacity <= 0 {
		panic("buffer: capacity must be a positive integer")
	}
	return &RingBuffer{
		ch:       make(chan models.NewsItem, capacity),
		capacity: capacity,
		logger:   logger,
	}
}

// Push attempts to enqueue item without blocking. If the buffer is full it
// drains the oldest item first (non-blocking read) and then enqueues the
// new one — preserving recency. This ensures the downstream consumer always
// processes the most recent events.
func (r *RingBuffer) Push(item models.NewsItem) {
	select {
	case r.ch <- item:
		// Fast path: buffer had capacity.
	default:
		// Slow path: buffer is full; evict the oldest item.
		select {
		case old := <-r.ch:
			dropped := r.dropped.Add(1)
			r.logger.Warn("buffer: overflow — dropped oldest item",
				slog.String("evicted_id", old.ID),
				slog.Int64("total_dropped", dropped),
			)
		default:
			// Another goroutine drained the slot between our two selects; safe to ignore.
		}
		// Now there is guaranteed space (we are the only writer draining+writing).
		r.ch <- item
	}
}

// Pop returns the channel so callers can consume items using range or select.
// This exposes the underlying channel as read-only to prevent external writes.
//
//	for item := range buf.Pop() { ... }
func (r *RingBuffer) Pop() <-chan models.NewsItem {
	return r.ch
}

// Len returns the current number of items waiting in the buffer.
// Useful for monitoring / Prometheus metrics exporters.
func (r *RingBuffer) Len() int {
	return len(r.ch)
}

// Capacity returns the maximum number of items the buffer can hold.
func (r *RingBuffer) Capacity() int {
	return r.capacity
}

// DroppedCount returns the total number of items evicted due to overflow
// since the buffer was created.
func (r *RingBuffer) DroppedCount() int64 {
	return r.dropped.Load()
}
