from __future__ import annotations

import asyncio
import logging
from typing import Any

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

from lydian.core.config import get_settings
from lydian.schemas.models import HistoricalEvent

logger = logging.getLogger(__name__)

# ─── Module-level singletons (initialised once during app lifespan) ───────────
# These are intentionally module-level (not global in the mutable sense):
# they are set once by `init_vector_store()` during the FastAPI lifespan
# and treated as read-only thereafter.  All async callers share the same
# instances, which is safe because both lancedb tables and SentenceTransformer
# inference are thread-safe for concurrent reads.

_embedder: SentenceTransformer | None = None
_table: Any | None = None  # lancedb.Table


async def init_vector_store() -> None:
    """Load the embedding model and open the LanceDB table.
    Must be called once during application startup before any RAG queries.
    Uses ``asyncio.to_thread`` to avoid blocking the event loop during the
    (slow) model weight loading."""
    global _embedder, _table  # noqa: PLW0603 — intentional singleton init

    cfg = get_settings()

    logger.info(
        "vector_store: loading embedding model",
        extra={"model": cfg.embedding_model},
    )
    _embedder = await asyncio.to_thread(
        SentenceTransformer,
        cfg.embedding_model,
        cache_folder=cfg.hf_cache_dir,
    )

    db = await asyncio.to_thread(lancedb.connect, cfg.lancedb_path)
    table_names = await asyncio.to_thread(db.table_names)

    if cfg.lancedb_table_name not in table_names:
        logger.warning(
            "vector_store: table '%s' not found — run storage/seed.py first",
            cfg.lancedb_table_name,
        )
        _table = None
        return

    _table = await asyncio.to_thread(db.open_table, cfg.lancedb_table_name)
    row_count = await asyncio.to_thread(lambda: _table.count_rows())
    logger.info(
        "vector_store: ready",
        extra={"table": cfg.lancedb_table_name, "rows": row_count},
    )


async def embed(text: str) -> list[float]:
    """Embed *text* using the loaded SentenceTransformer model.

    Runs in a thread pool to avoid blocking the async event loop.
    Raises ``RuntimeError`` if called before ``init_vector_store``.

    Target latency: < 50 ms on CPU for BGE-Small-v1.5 (384-dim).
    """
    if _embedder is None:
        raise RuntimeError(
            "vector_store.embed() called before init_vector_store(). "
            "Ensure the FastAPI lifespan has completed startup."
        )
    vector: np.ndarray = await asyncio.to_thread(
        _embedder.encode,
        text,
        normalize_embeddings=True,  # cosine similarity via dot product
        show_progress_bar=False,
    )
    return vector.tolist()


async def search(query_text: str, k: int | None = None) -> list[HistoricalEvent]:
    """Retrieve the *k* most similar historical events for *query_text*.

    Returns an empty list (not raises) if the table is not initialised,
    so the agent graph can degrade gracefully without crashing.

    Args:
        query_text: The text to embed and query against.
        k:          Number of results. Defaults to ``settings.rag_top_k``.

    Returns:
        List of ``HistoricalEvent`` instances ordered by descending similarity.
    """
    cfg = get_settings()
    k = k or cfg.rag_top_k

    if _table is None:
        logger.warning("vector_store.search(): table not initialised — returning empty")
        return []

    try:
        query_vector = await embed(query_text)
    except Exception as exc:
        logger.error("vector_store.search(): embed failed: %s", exc)
        return []

    def _sync_search() -> list[dict]:
        results = (
            _table.search(query_vector)
            .metric("cosine")
            .limit(k)
            .to_list()
        )
        return results

    try:
        rows = await asyncio.to_thread(_sync_search)
    except Exception as exc:
        logger.error("vector_store.search(): lancedb query failed: %s", exc)
        return []

    events: list[HistoricalEvent] = []
    for row in rows:
        try:
            # LanceDB cosine search returns `_distance` in [0, 2];
            # convert to similarity in [0, 1].
            distance = float(row.get("_distance", 1.0))
            similarity = max(0.0, 1.0 - distance / 2.0)
            events.append(
                HistoricalEvent(
                    event_id=str(row.get("event_id", "unknown")),
                    headline=str(row.get("headline", "")),
                    date=str(row.get("date", "")),
                    impact=str(row.get("impact", "")),
                    category=str(row.get("category", "")),
                    similarity_score=round(similarity, 4),
                )
            )
        except Exception as exc:
            logger.warning("vector_store.search(): skipping malformed row: %s", exc)
            continue

    logger.debug(
        "vector_store.search(): retrieved %d events for query %.80s",
        len(events),
        query_text,
    )
    return events
