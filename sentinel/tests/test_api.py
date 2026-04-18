"""Integration test for the FastAPI endpoints (no real LLM, mocked agents)."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


VALID_NEWS_ITEM = {
    "id": "test-001",
    "headline": "Federal Reserve announces emergency rate cut",
    "body": (
        "The Federal Reserve voted to cut rates by 50bps in an emergency "
        "inter-meeting action after credit markets showed signs of stress."
    ),
    "source": "Reuters",
    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    "tickers": ["SPY", "TLT"],
}

MOCK_AGENT_STATE = {
    "severity": "Critical",
    "filter_reasoning": "Central bank emergency action with systemic implications.",
    "historical_context": [
        {
            "event_id": "HE-001",
            "headline": "Fed Emergency Cut 2020",
            "date": "2020-03-03",
            "impact": "S&P 500 +4.6%",
            "category": "Central Bank",
            "similarity_score": 0.94,
        }
    ],
    "agents_invoked": ["FilterAgent", "HistorianAgent"],
}


@pytest.fixture
def mock_app():
    """Returns the FastAPI app with all external calls mocked out."""
    with (
        patch("lydian.storage.seed.seed", return_value=25),
        patch("lydian.storage.vector_store.init_vector_store", new_callable=AsyncMock),
        patch("lydian.agents.graph.run", new_callable=AsyncMock, return_value=MOCK_AGENT_STATE),
    ):
        from lydian.main import create_app
        app = create_app()
        yield app


@pytest.mark.asyncio
async def test_health_endpoint(mock_app):
    async with AsyncClient(transport=ASGITransport(app=mock_app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "timestamp" in body


@pytest.mark.asyncio
async def test_analyze_returns_critical_alert(mock_app):
    async with AsyncClient(transport=ASGITransport(app=mock_app), base_url="http://test") as ac:
        response = await ac.post("/analyze", json=VALID_NEWS_ITEM)
    assert response.status_code == 200
    body = response.json()
    assert body["severity"] == "Critical"
    assert len(body["historical_context"]) == 1
    assert "processing_time_ms" in body
    assert "X-Processing-Time-Ms" in response.headers


@pytest.mark.asyncio
async def test_ingest_returns_202(mock_app):
    async with AsyncClient(transport=ASGITransport(app=mock_app), base_url="http://test") as ac:
        response = await ac.post("/ingest", json=VALID_NEWS_ITEM)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["news_id"] == "test-001"


@pytest.mark.asyncio
async def test_analyze_malformed_input_returns_422(mock_app):
    async with AsyncClient(transport=ASGITransport(app=mock_app), base_url="http://test") as ac:
        response = await ac.post("/analyze", json={"bad": "payload"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_empty_headline_returns_422(mock_app):
    payload = VALID_NEWS_ITEM.copy()
    payload["headline"] = ""
    async with AsyncClient(transport=ASGITransport(app=mock_app), base_url="http://test") as ac:
        response = await ac.post("/analyze", json=payload)
    assert response.status_code == 422
