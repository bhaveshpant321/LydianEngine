"""main.py — Sentinel FastAPI Application.

This module wires together all subsystems and exposes the public API surface:

    POST /ingest   — receives raw NewsItems from the Go ingestion service
    POST /analyze  — runs the full LangGraph agent pipeline
    GET  /health   — liveness probe for Docker/Kubernetes

Architecture notes:
- Lifespan context manager handles startup (seed + model load) and shutdown.
- Request timing middleware logs any response that exceeds the 200ms SLA.
- All heavy computation is async / thread-pooled; no blocking calls on the
  event loop.
- Pydantic validation happens at the FastAPI layer before any agent logic runs,
  so agents always receive well-formed input.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lydian.agents import graph as agent_graph
from lydian.core.config import get_settings
from lydian.schemas.models import AlertResponse, IngestAck, NewsItem
from lydian.storage import seed as db_seed
from lydian.storage import vector_store

# ---- Logging ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- In-memory pending queue -------------------------------------------------
# Lightweight asyncio queue for items received via /ingest that have not yet
# been processed.  A background task drains this queue continuously.
# In production, swap for Redis Streams / Kafka for durability.
_pending_queue: asyncio.Queue[NewsItem] = asyncio.Queue(maxsize=1024)


# ---- Lifespan ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Runs on application startup and shutdown.

    Startup:
    1. Seed LanceDB from market_history.csv (idempotent).
    2. Initialise the embedding model and open the LanceDB table.
    3. Start the background queue drainer.

    Shutdown:
    4. Signal the drainer to stop and wait for it to finish.
    """
    import os
    cfg = get_settings()

    # Explicitly export HF token so transformers/huggingface_hub can see it
    if cfg.hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token
        os.environ["HF_TOKEN"] = cfg.hf_token

    logger.info("startup: seeding LanceDB ...")

    # Seeding is CPU-bound; run in thread pool.
    try:
        count = await asyncio.to_thread(db_seed.seed, False)
        logger.info("startup: LanceDB ready (%d rows)", count)
    except Exception as exc:
        logger.error("startup: seed failed: %s — continuing without history", exc)

    logger.info("startup: initialising vector store ...")
    await vector_store.init_vector_store()

    logger.info("startup: pre-warming classification model ...")
    from lydian.agents import filter_agent
    await filter_agent.prewarm()

    # Background task: drains _pending_queue and runs the agent graph.
    drainer_task = asyncio.create_task(_queue_drainer(), name="queue-drainer")

    logger.info("startup: complete. Lydian Engine is ready.")
    yield

    # Shutdown: cancel the drainer and give it 5s to finish in-flight work.
    drainer_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(drainer_task), timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    logger.info("shutdown: complete.")


async def _queue_drainer() -> None:
    """Background coroutine that processes items from the pending queue.

    Items are fire-and-forget from the /ingest endpoint; results are logged
    but not returned to the caller.  In a production system, results would
    be emitted to a message bus or stored in a time-series DB.
    """
    while True:
        try:
            item: NewsItem = await _pending_queue.get()
            try:
                state = await agent_graph.run(item)
                logger.info(
                    "drainer: processed item '%s' -> severity=%s, "
                    "history_hits=%d",
                    item.id,
                    state.get("severity"),
                    len(state.get("historical_context", [])),
                )
            except Exception as exc:
                logger.error("drainer: agent graph error for '%s': %s", item.id, exc)
            finally:
                _pending_queue.task_done()
        except asyncio.CancelledError:
            logger.info("drainer: shutting down, %d items remaining", _pending_queue.qsize())
            return


# ---- App factory -------------------------------------------------------------

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="Lydian Engine",
        description=(
            "Real-time agentic financial audit engine. "
            "Classifies news as Critical/Noise and retrieves historical market parallels."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — restrict in production via environment variable.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ---- Latency SLA middleware ------------------------------------------
    @app.middleware("http")
    async def latency_sla_middleware(request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        response.headers["X-Processing-Time-Ms"] = f"{elapsed_ms:.2f}"

        if elapsed_ms > cfg.request_timeout_ms and request.url.path == "/analyze":
            logger.warning(
                "SLA BREACH: /analyze took %.1f ms (SLA: %.0f ms)",
                elapsed_ms,
                cfg.request_timeout_ms,
            )
        return response

    # ---- Routes -------------------------------------------------------------

    @app.get("/health", tags=["Observability"])
    async def health() -> dict:
        """Liveness probe. Returns 200 when the service is ready."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "queue_depth": _pending_queue.qsize(),
        }

    @app.post(
        "/ingest",
        response_model=IngestAck,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Ingestion"],
        summary="Receive a raw news item from the Go ingestion service",
    )
    async def ingest(item: NewsItem) -> IngestAck:
        """Accepts a NewsItem from the Go service and enqueues it for async
        processing.  Returns 202 immediately so the Go dispatcher is not blocked.

        Returns 503 if the internal queue is full (back-pressure signal).
        """
        try:
            _pending_queue.put_nowait(item)
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Sentinel queue at capacity ({_pending_queue.maxsize}). "
                    "Retry after a short delay."
                ),
            )

        return IngestAck(
            status="accepted",
            news_id=item.id,
            queued_at=datetime.now(tz=timezone.utc),
        )

    @app.post(
        "/analyze",
        response_model=AlertResponse,
        tags=["Analysis"],
        summary="Synchronously classify a news item and retrieve historical context",
    )
    async def analyze(item: NewsItem) -> AlertResponse:
        """Runs the full two-agent LangGraph pipeline synchronously and returns
        a structured AlertResponse.

        - Agent A classifies the item as Critical or Noise.
        - If Critical, Agent B retrieves the top-3 historical parallels.

        Target latency: < 200 ms (logged as SLA breach if exceeded).
        """
        t0 = time.perf_counter()

        try:
            state = await agent_graph.run(item)
        except Exception as exc:
            logger.exception("analyze: unhandled agent error for item '%s'", item.id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent pipeline failed: {exc}",
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return AlertResponse(
            news_id=item.id,
            headline=item.headline,
            severity=state.get("severity", "Noise"),
            filter_reasoning=state.get("filter_reasoning", ""),
            historical_context=state.get("historical_context", []),
            max_similarity=round(state.get("max_similarity", 0.0), 4),
            is_black_swan=state.get("is_black_swan", False),
            processing_time_ms=round(elapsed_ms, 2),
            agents_invoked=state.get("agents_invoked", []),
        )

    return app


# ---- Entry point ------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "sentinel.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        log_level="info",
        reload=False,
    )
