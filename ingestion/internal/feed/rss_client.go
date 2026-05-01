package feed

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/mmcdole/gofeed"
	"lydian-ingestion/internal/models"
)

// RSSClient implements FeedClient by polling one or more RSS feeds at a
// regular interval. It tracks seen items to ensure duplicates aren't sent
// to the downstream Sentinel service.
type RSSClient struct {
	urls     []string
	interval time.Duration
	logger   *slog.Logger
	seen     map[string]time.Time
}

// NewRSSClient constructs an RSSClient.
func NewRSSClient(urls []string, interval time.Duration, logger *slog.Logger) *RSSClient {
	return &RSSClient{
		urls:     urls,
		interval: interval,
		logger:   logger,
		seen:     make(map[string]time.Time),
	}
}

// Name satisfies the FeedClient interface.
func (r *RSSClient) Name() string {
	return fmt.Sprintf("RSSClient(%d feeds)", len(r.urls))
}

// Connect starts a polling loop that fetches RSS feeds and streams new
// NewsItems onto the returned channel.
func (r *RSSClient) Connect(ctx context.Context) (<-chan models.NewsItem, error) {
	out := make(chan models.NewsItem, 64)
	fp := gofeed.NewParser()

	go func() {
		defer close(out)

		// Immediate first fetch
		r.poll(ctx, fp, out)

		ticker := time.NewTicker(r.interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				r.poll(ctx, fp, out)
			}
		}
	}()

	return out, nil
}

func (r *RSSClient) poll(ctx context.Context, fp *gofeed.Parser, out chan<- models.NewsItem) {
	for _, url := range r.urls {
		if ctx.Err() != nil {
			return
		}

		r.logger.Debug("rss: polling feed", slog.String("url", url))

		feed, err := fp.ParseURLWithContext(url, ctx)
		if err != nil {
			r.logger.Error("rss: fetch failed", 
				slog.String("url", url), 
				slog.String("error", err.Error()),
			)
			continue
		}

		newCount := 0
		for _, item := range feed.Items {
			id := item.GUID
			if id == "" {
				id = item.Link
			}

			if _, exists := r.seen[id]; exists {
				continue
			}

			// Clean up old 'seen' entries every 24h to prevent memory leak
			r.cleanupSeen()

			ts := time.Now()
			if item.PublishedParsed != nil {
				ts = *item.PublishedParsed
			}

				ni := models.NewsItem{
					ID:        id,
					Headline:  item.Title,
					Body:      item.Description,
					Source:    feed.Title,
					Timestamp: ts,
				}

				// Fallback: Use content if description is empty
				if strings.TrimSpace(ni.Body) == "" {
					ni.Body = item.Content
				}

				ni.DetectNegation()

			if err := ni.Validate(); err != nil {
				r.logger.Warn("rss: skipping invalid item", 
					slog.String("id", id), 
					slog.String("error", err.Error()),
				)
				continue
			}

			select {
			case out <- ni:
				r.seen[id] = time.Now()
				newCount++
			case <-ctx.Done():
				return
			}
		}

		if newCount > 0 {
			r.logger.Info("rss: fetched new items", 
				slog.String("feed", feed.Title), 
				slog.Int("count", newCount),
			)
		}
	}
}

func (r *RSSClient) cleanupSeen() {
	// Simple TTL for seen items: 48 hours
	if len(r.seen) < 1000 {
		return
	}
	cutoff := time.Now().Add(-48 * time.Hour)
	for id, t := range r.seen {
		if t.Before(cutoff) {
			delete(r.seen, id)
		}
	}
}
