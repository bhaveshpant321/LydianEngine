// Package models defines the shared data types for the ingestion service.
package models

import (
	"errors"
	"strings"
	"time"
)

// Severity enumerates the possible classification outcomes.
type Severity string

const (
	SeverityCritical Severity = "Critical"
	SeverityNoise    Severity = "Noise"
	SeverityUnknown  Severity = "Unknown"
)

// NewsItem is the canonical representation of a single news event flowing
// through the ingestion pipeline. All downstream consumers receive this type.
type NewsItem struct {
	// ID is a unique identifier for this event (e.g., UUID from upstream).
	ID string `json:"id"`
	// Headline is the short display title of the news article.
	Headline string `json:"headline"`
	// Body is the full article text or a meaningful excerpt.
	Body string `json:"body"`
	// Source identifies the originating publication or data vendor.
	Source string `json:"source"`
	// Timestamp records when the event was published, in RFC3339 format.
	Timestamp time.Time `json:"timestamp"`
	// Tickers contains zero or more equity symbols mentioned in the article.
	Tickers []string `json:"tickers,omitempty"`
	// PotentialNegation is flagged if keywords like 'no' or 'not' are found.
	PotentialNegation bool `json:"potential_negation"`
	// Severity is populated downstream by Agent A; left empty at ingestion.
	Severity Severity `json:"severity,omitempty"`
}

// Validate performs structural validation on a NewsItem and returns an
// aggregated error describing all constraint violations, so callers can
// emit a single, descriptive log line rather than fail on the first error.
func (n *NewsItem) Validate() error {
	var errs []string

	if strings.TrimSpace(n.ID) == "" {
		errs = append(errs, "id must not be empty")
	}
	if strings.TrimSpace(n.Headline) == "" {
		errs = append(errs, "headline must not be empty")
	}
	if strings.TrimSpace(n.Source) == "" {
		errs = append(errs, "source must not be empty")
	}
	if n.Timestamp.IsZero() {
		errs = append(errs, "timestamp must not be zero")
	}
	if len(n.Body) > 50_000 {
		errs = append(errs, "body exceeds maximum allowed length of 50,000 bytes")
	}

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

// DetectNegation scans the headline and body for inversion keywords.
func (n *NewsItem) DetectNegation() {
	keywords := []string{"not", "no ", "never", "fail", "cancel", "denies", "refutes", "halt", "stop", "none"}
	text := strings.ToLower(n.Headline + " " + n.Body)
	for _, kw := range keywords {
		if strings.Contains(text, kw) {
			n.PotentialNegation = true
			return
		}
	}
}
