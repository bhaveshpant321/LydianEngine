from __future__ import annotations

import asyncio
import logging
from typing import Any

import lancedb
import numpy as np

from huggingface_hub import InferenceClient
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
    """Load the embedding model and open the LanceDB table."""
    global _embedder, _table
    cfg = get_settings()

    if cfg.inference_mode == "local":
        logger.info("vector_store: loading local embedding model: %s", cfg.embedding_model)
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = await asyncio.to_thread(
                SentenceTransformer,
                cfg.embedding_model,
                cache_folder=cfg.hf_cache_dir,
            )
        except Exception as exc:
            logger.error("vector_store: failed to load local model: %s. Falling back to cloud-only.", exc)
            _embedder = None
    else:
        logger.info("vector_store: cloud mode active — using HF Inference API for embeddings")
        _embedder = None # Handled by InferenceClient

    db = await asyncio.to_thread(lancedb.connect, cfg.lancedb_path)
    table_names = await asyncio.to_thread(db.table_names)

    if cfg.lancedb_table_name not in table_names:
        logger.warning("vector_store: table '%s' not found", cfg.lancedb_table_name)
        _table = None
        return

    _table = await asyncio.to_thread(db.open_table, cfg.lancedb_table_name)
    row_count = await asyncio.to_thread(lambda: _table.count_rows())
    logger.info("vector_store: ready (rows=%d)", row_count)


async def embed(text: str) -> list[float]:
    """Embed *text* using either local or cloud models."""
    cfg = get_settings()

    if cfg.inference_mode == "local" and _embedder:
        vector: np.ndarray = await asyncio.to_thread(
            _embedder.encode,
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
    
    # Cloud Embedding Fallback (or primary if mode is cloud)
    try:
        client = InferenceClient(token=cfg.hf_token)
        # Use feature_extraction for embedding models
        response = await asyncio.to_thread(
            client.feature_extraction,
            text,
            model=cfg.embedding_model
        )
        # BGE returns a 1D or 2D list depending on input
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], list): return response[0]
            return response
        return [0.0] * cfg.embedding_dim # Fail-safe
    except Exception as exc:
        logger.error("vector_store: cloud embed failed: %s", exc)
        return [0.0] * cfg.embedding_dim


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


async def upsert_item(item: NewsItem, category: str = "Live Feed") -> None:
    """Persist a new news item into the historical vector store.
    This enables 'Growing Memory' (self-learning), where today's news
    becomes tomorrow's historical context for the Historian agent."""
    if _table is None:
        logger.warning("vector_store.upsert_item(): table not initialised — skipping")
        return

    try:
        # Use the same pre-processing logic as seeding to ensure search compatibility
        full_text = item.text_for_embedding
        vector = await embed(full_text)

        row = {
            "event_id": item.id,
            "headline": item.headline,
            "date": item.timestamp.strftime("%Y-%m-%d"),
            "impact": "Live classification: Audit pending.",
            "category": category,
            "full_text": full_text,
            "vector": vector,
        }

        # LanceDB Table.add is synchronous in the python lib, 
        # so we run it in a thread pool.
        await asyncio.to_thread(_table.add, [row])
        logger.info(
            "vector_store: persisted item '%s' as historical context",
            item.id,
            extra={"headline": item.headline},
        )
    except Exception as exc:
        logger.error("vector_store.upsert_item() failed: %s", exc)
