from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class NewsItem(BaseModel):
    """Canonical representation of a single news event received from the Go
    ingestion service. All fields are validated on construction; invalid
    payloads raise ``ValidationError`` before reaching any agent logic."""

    id: Annotated[str, Field(min_length=1, description="Unique event identifier")]
    headline: Annotated[
        str,
        Field(min_length=1, max_length=512, description="Short article title"),
    ]
    body: Annotated[
        str,
        Field(
            min_length=1,
            max_length=50_000,
            description="Full article text or meaningful excerpt",
        ),
    ]
    source: Annotated[str, Field(min_length=1, description="Originating publication")]
    timestamp: Annotated[datetime, Field(description="UTC publication time")]
    tickers: Annotated[
        list[str],
        Field(default_factory=list, description="Equity symbols mentioned"),
    ]

    @field_validator("id", "headline", "body", "source", mode="before")
    @classmethod
    def strip_and_validate_strings(cls, v: object) -> str:
        """Ensures core strings are stripped and non-empty after stripping."""
        if not isinstance(v, str):
            raise ValueError("field must be a string")
        v = v.strip()
        if not v:
            raise ValueError("field cannot be empty or whitespace-only")
        return v

    @field_validator("tickers", mode="before")
    @classmethod
    def normalise_tickers(cls, v: object) -> list[str]:
        """Accepts None, an empty list, or a list of strings. Strips
        whitespace and upper-cases each ticker symbol."""
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("tickers must be a list")
        return [t.strip().upper() for t in v if isinstance(t, str) and t.strip()]

    @model_validator(mode="after")
    def timestamp_not_in_future(self) -> "NewsItem":
        """Rejects timestamps more than 60 seconds in the future, which
        typically indicates a clock skew bug in the upstream producer."""
        from datetime import timezone

        now_utc = datetime.now(tz=timezone.utc)
        ts = self.timestamp
        if ts.tzinfo is None:
            # Treat naive timestamps as UTC.
            from datetime import timezone as tz

            ts = ts.replace(tzinfo=tz.utc)
        delta = (ts - now_utc).total_seconds()
        if delta > 60:
            raise ValueError(
                f"timestamp is {delta:.0f}s in the future — possible clock skew"
            )
        return self

    @property
    def text_for_embedding(self) -> str:
        """Returns the concatenated representation used for vector embedding.
        Including the source normalises style differences across vendors."""
        return f"[{self.source}] {self.headline}. {self.body[:2048]}"


# ─────────────────────────────────────────────────────────────────────────────


class HistoricalEvent(BaseModel):
    """A single historical market event retrieved from the LanceDB vector
    store, enriched with its similarity distance for downstream ranking."""

    event_id: str = Field(description="Unique identifier from market_history.csv")
    headline: str = Field(description="Historical event headline")
    date: str = Field(description="Date of the historical event (ISO 8601)")
    impact: str = Field(
        description="Documented market impact (e.g., 'S&P -4.2% intraday')"
    )
    category: str = Field(
        description="Event category (e.g., 'Central Bank', 'Credit Event')"
    )
    similarity_score: float = Field(
        ge=0.0, description="Cosine similarity to the triggering news item (0–1)"
    )


class AlertResponse(BaseModel):
    """The structured response payload returned by the /analyze endpoint.
    This is the contract between the Sentinel API and all downstream consumers
    (dashboards, trading systems, compliance tools)."""

    # ── Identity ─────────────────────────────────────────────────────────────
    news_id: str = Field(description="ID of the originating news item")
    headline: str = Field(description="Headline of the originating news item")

    # ── Classification ───────────────────────────────────────────────────────
    severity: str = Field(
        description="Agent A verdict: 'Critical' | 'Noise'",
        pattern=r"^(Critical|Noise)$",
    )
    filter_reasoning: str = Field(
        description="Agent A's free-text justification for the severity verdict"
    )

    # ── Historical context (populated only for Critical items) ───────────────
    historical_context: list[HistoricalEvent] = Field(
        default_factory=list,
        description="Top-3 similar historical events from Agent B (empty if Noise)",
    )

    # ── Observability ────────────────────────────────────────────────────────
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="End-to-end latency from /analyze request receipt to response",
    )
    agents_invoked: list[str] = Field(
        default_factory=list,
        description="Names of agents that ran (for tracing / debugging)",
    )


class IngestAck(BaseModel):
    """Lightweight acknowledgement returned by the /ingest endpoint after the
    Go service pushes a raw NewsItem for background processing."""

    status: str = Field(default="accepted", description="Always 'accepted' on success")
    news_id: str = Field(description="Echo of the received news item's id")
    queued_at: datetime = Field(description="Server-side receive timestamp (UTC)")
