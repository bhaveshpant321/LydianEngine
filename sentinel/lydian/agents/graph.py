"""graph.py — LangGraph Agent State Machine.

Defines the two-node DAG:

    [filter_node] --Critical--> [historian_node] --> END
                  --Noise-----> END

State flows as a typed TypedDict through the graph.  Each node mutates a
specific subset of the state and returns only the fields it changes,
following LangGraph's partial-state-update convention.

The graph is compiled once at module import time and cached as a singleton
for the process lifetime.
"""
from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from lydian.agents import filter_agent, historian_agent
from lydian.core.config import get_settings
from lydian.schemas.models import HistoricalEvent, NewsItem

logger = logging.getLogger(__name__)


# ---- Typed graph state -------------------------------------------------------

class AgentState(TypedDict):
    """The mutable state passed between nodes in the LangGraph DAG."""
    news_item: NewsItem
    severity: str           # "Critical" | "Noise"
    filter_reasoning: str   
    historical_context: list[HistoricalEvent]
    max_similarity: float   # Max score from RAG step
    is_black_swan: bool     # Flag for novel events
    agents_invoked: list[str]


# ---- Node implementations ----------------------------------------------------

async def search_node(state: AgentState) -> dict:
    """Agent B (Historian/Search): Retrieve historical context BEFORE classification.
    This enables few-shot learning and short-circuiting logic.
    """
    item = state["news_item"]
    logger.info("graph: search_node (RAG-First) processing item '%s'", item.id)

    events = await historian_agent.retrieve_historical_context(item)
    max_sim = max([e.similarity_score for e in events]) if events else 0.0

    return {
        "historical_context": events,
        "max_similarity": max_sim,
        "agents_invoked": state.get("agents_invoked", []) + ["SearchAgent"],
    }


async def filter_node(state: AgentState) -> dict:
    """Agent A: Context-aware classification using retrieved RAG results."""
    item = state["news_item"]
    context = state["historical_context"]
    max_sim = state["max_similarity"]
    cfg = get_settings()

    logger.info("graph: filter_node (Few-Shot) processing item '%s'", item.id)

    verdict, reasoning = await filter_agent.classify(item, context_events=context)

    # Black Swan Logic
    is_black_swan = False
    if verdict == "Critical" and max_sim < cfg.black_swan_threshold:
        is_black_swan = True
        reasoning = f"⚠️ BLACK SWAN DETECTED: {reasoning}"
        logger.warning("graph: item '%s' flagged as Black Swan (sim=%.2f)", item.id, max_sim)

    return {
        "severity": verdict,
        "filter_reasoning": reasoning,
        "is_black_swan": is_black_swan,
        "agents_invoked": state.get("agents_invoked", []) + ["FilterAgent"],
    }


async def archivist_node(state: AgentState) -> dict:
    """Agent C (Archivist): Persist the news item into long-term history.
    This enables the system's memory to grow, but with strict quality gates
    to prevent database pollution.
    """
    item = state["news_item"]
    severity = state.get("severity")
    max_sim = state.get("max_similarity", 0.0)
    is_black_swan = state.get("is_black_swan", False)

    # Quality Gate: 
    # Only archive if:
    # 1. It was auto-classified (High Confidence)
    # 2. It was flagged as a Black Swan (Genuine Novelty)
    # 3. It's a standard Critical event with a decent similarity baseline (>0.4)
    # This prevents 'weak' criticals from polluting the core history.
    
    should_archive = (
        (max_sim >= get_settings().short_circuit_critical_threshold) or 
        is_black_swan or 
        (severity == "Critical" and max_sim > 0.4)
    )

    if should_archive:
        logger.info("graph: archivist_node quality gate passed — persisting item '%s'", item.id)
        from lydian.storage import vector_store
        # If it's a black swan, mark it specifically in history
        category = "Black Swan" if is_black_swan else "Live Feed"
        await vector_store.upsert_item(item, category=category)
    else:
        logger.warning("graph: archivist_node quality gate FAILED for item '%s' — skipping persistence", item.id)

    return {
        "agents_invoked": state.get("agents_invoked", []) + ["ArchivistAgent"],
    }


# ---- Routing logic -----------------------------------------------------------

def route_after_search(state: AgentState) -> str:
    """Conditional edge: implement short-circuiting logic based on RAG scores."""
    cfg = get_settings()
    max_sim = state["max_similarity"]
    item = state["news_item"]

    # 1. Short-circuit: High Confidence Critical
    # Bypass short-circuit if a negation keyword was detected (prevents 'semantic inversion' errors)
    if max_sim >= cfg.short_circuit_critical_threshold and not item.potential_negation:
        logger.info("graph: short-circuiting to CRITICAL for item '%s' (sim=%.2f)", item.id, max_sim)
        state["severity"] = "Critical"
        state["filter_reasoning"] = f"Auto-classified: High similarity (%.2f) to historical critical events." % max_sim
        return "archivist_node"

    # 2. Short-circuit: High Confidence Noise (Safeguarded by Keyword Check)
    critical_keywords = {"fed", "rate", "war", "crash", "seized", "fail", "cpi", "inflation"}
    text = (item.headline + " " + item.body).lower()
    has_keywords = any(kw in text for kw in critical_keywords)

    if max_sim <= cfg.short_circuit_noise_threshold and not has_keywords:
        logger.info("graph: short-circuiting to NOISE for item '%s' (sim=%.2f)", item.id, max_sim)
        state["severity"] = "Noise"
        state["filter_reasoning"] = "Auto-classified: Low similarity and no critical keywords detected."
        return END

    # 3. Standard Path: Needs SLM reasoning
    return "filter_node"


def route_after_filter(state: AgentState) -> str:
    """Route Critical items to Archivist, Noise to END."""
    if state.get("severity") == "Critical":
        return "archivist_node"
    return END


# ---- Graph compilation (singleton) ------------------------------------------

def _build_graph() -> object:
    """Construct and compile the StateGraph exactly once."""
    builder = StateGraph(AgentState)

    builder.add_node("search_node", search_node)
    builder.add_node("filter_node", filter_node)
    builder.add_node("archivist_node", archivist_node)

    builder.set_entry_point("search_node")

    # After search, decide if we short-circuit or filter
    builder.add_conditional_edges(
        "search_node",
        route_after_search,
        {
            "archivist_node": "archivist_node",
            "filter_node": "filter_node",
            END: END,
        },
    )

    # After filter, decide if we archive or end
    builder.add_conditional_edges(
        "filter_node",
        route_after_filter,
        {
            "archivist_node": "archivist_node",
            END: END,
        },
    )

    builder.add_edge("archivist_node", END)

    compiled = builder.compile()
    logger.info("graph: StateGraph compiled successfully")
    return compiled


# Singleton: compiled once at import time.
GRAPH = _build_graph()


# ---- Public entry point ------------------------------------------------------

async def run(news_item: NewsItem) -> AgentState:
    """Run the full agent graph for *news_item* and return the final state.

    Args:
        news_item: The validated NewsItem received from the Go service.

    Returns:
        The final AgentState after all applicable nodes have executed.

    Raises:
        Exception: Any unhandled error from the node execution is propagated
                   so the FastAPI endpoint can return a 500 with context.
    """
    initial_state: AgentState = {
        "news_item": news_item,
        "severity": "Unknown",
        "filter_reasoning": "",
        "historical_context": [],
        "max_similarity": 0.0,
        "is_black_swan": False,
        "agents_invoked": [],
    }

    final_state: AgentState = await GRAPH.ainvoke(initial_state)
    return final_state
