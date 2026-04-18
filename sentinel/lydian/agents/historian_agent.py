"""historian_agent.py — Agent B: The RAG Historian.

Retrieves the top-k most similar historical market events from LanceDB
in response to a news item classified as 'Critical' by Agent A.

Design choices:
- Pure async: all LanceDB I/O runs through vector_store.search() which
  already wraps blocking calls in asyncio.to_thread().
- Returns an empty list (not raises) on any retrieval failure so the graph
  can still return a partial AlertResponse to the caller.
- Logs retrieval latency for SLA monitoring.
"""
from __future__ import annotations

import logging
import time

from lydian.schemas.models import HistoricalEvent, NewsItem
from lydian.storage import vector_store

logger = logging.getLogger(__name__)


async def retrieve_historical_context(item: NewsItem) -> list[HistoricalEvent]:
    """Retrieve the top-k most similar historical market events for *item*.

    Uses the ``text_for_embedding`` property of NewsItem (headline + body
    excerpt) as the query vector.  Similarity is computed via cosine distance
    in the BGE-small-v1.5 embedding space.

    Args:
        item: The Critical news item triggering the RAG lookup.

    Returns:
        A list of up to ``settings.rag_top_k`` HistoricalEvent objects,
        ordered by descending similarity.  Returns an empty list on error.
    """
    t0 = time.perf_counter()

    try:
        events = await vector_store.search(item.text_for_embedding)
    except Exception as exc:
        # Should not reach here since vector_store.search() never raises,
        # but we guard defensively.
        logger.error(
            "historian_agent: unexpected error for item '%s': %s",
            item.id,
            exc,
        )
        return []

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "historian_agent: retrieved %d events in %.1f ms for item '%s'",
        len(events),
        elapsed_ms,
        item.id,
    )

    if not events:
        logger.warning(
            "historian_agent: no historical events found for item '%s' — "
            "is the vector store seeded?",
            item.id,
        )

    return events
