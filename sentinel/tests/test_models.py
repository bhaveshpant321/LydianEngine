"""Tests for Pydantic model validation."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from lydian.schemas.models import NewsItem, AlertResponse, HistoricalEvent


# ── NewsItem validation ───────────────────────────────────────────────────────

class TestNewsItemValidation:

    def _valid_payload(self) -> dict:
        return {
            "id": "evt-001",
            "headline": "Fed cuts rates by 50bps",
            "body": "The Federal Reserve voted unanimously to cut rates.",
            "source": "Reuters",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "tickers": ["SPY", "TLT"],
        }

    def test_valid_news_item(self):
        item = NewsItem(**self._valid_payload())
        assert item.id == "evt-001"
        assert "SPY" in item.tickers

    def test_empty_id_rejected(self):
        data = self._valid_payload()
        data["id"] = ""
        with pytest.raises(ValidationError, match="id"):
            NewsItem(**data)

    def test_empty_headline_rejected(self):
        data = self._valid_payload()
        data["headline"] = "   "
        with pytest.raises(ValidationError):
            NewsItem(**data)

    def test_body_too_long_rejected(self):
        data = self._valid_payload()
        data["body"] = "x" * 50_001
        with pytest.raises(ValidationError):
            NewsItem(**data)

    def test_future_timestamp_rejected(self):
        data = self._valid_payload()
        future = datetime.now(tz=timezone.utc) + timedelta(hours=1)
        data["timestamp"] = future.isoformat()
        with pytest.raises(ValidationError, match="future"):
            NewsItem(**data)

    def test_tickers_normalised_to_uppercase(self):
        data = self._valid_payload()
        data["tickers"] = [" spy ", "tlt", " GLD"]
        item = NewsItem(**data)
        assert item.tickers == ["SPY", "TLT", "GLD"]

    def test_none_tickers_becomes_empty_list(self):
        data = self._valid_payload()
        data["tickers"] = None
        item = NewsItem(**data)
        assert item.tickers == []

    def test_text_for_embedding_contains_source(self):
        item = NewsItem(**self._valid_payload())
        assert "[Reuters]" in item.text_for_embedding


# ── AlertResponse validation ──────────────────────────────────────────────────

class TestAlertResponseValidation:

    def test_critical_alert_with_context(self):
        alert = AlertResponse(
            news_id="evt-001",
            headline="Shock rate cut",
            severity="Critical",
            filter_reasoning="Surprise central bank action.",
            historical_context=[
                HistoricalEvent(
                    event_id="HE-001",
                    headline="Fed Emergency Cut 2020",
                    date="2020-03-03",
                    impact="S&P +4.6%",
                    category="Central Bank",
                    similarity_score=0.92,
                )
            ],
            processing_time_ms=145.3,
            agents_invoked=["FilterAgent", "HistorianAgent"],
        )
        assert alert.severity == "Critical"
        assert len(alert.historical_context) == 1

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError, match="severity"):
            AlertResponse(
                news_id="x",
                headline="h",
                severity="Unknown",   # not in pattern
                filter_reasoning="r",
            )

    def test_noise_alert_empty_context(self):
        alert = AlertResponse(
            news_id="evt-002",
            headline="Routine data revision",
            severity="Noise",
            filter_reasoning="Within margin of error.",
        )
        assert alert.historical_context == []
        assert alert.agents_invoked == []
