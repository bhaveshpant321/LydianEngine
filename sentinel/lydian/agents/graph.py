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
from lydian.schemas.models import HistoricalEvent, NewsItem

logger = logging.getLogger(__name__)


# ---- Typed graph state -------------------------------------------------------

class AgentState(TypedDict):
    """The mutable state passed between nodes in the LangGraph DAG.

    Fields are set incrementally: filter_node sets severity + reasoning;
    historian_node sets historical_context.
    """
    news_item: NewsItem
    severity: str           # "Critical" | "Noise" (set by filter_node)
    filter_reasoning: str   # Agent A's justification
    historical_context: list[HistoricalEvent]  # Agent B's results
    agents_invoked: list[str]   # audit trail for the response payload


# ---- Node implementations ----------------------------------------------------

async def filter_node(state: AgentState) -> dict:
    """Agent A: classify the news item as Critical or Noise."""
    item = state["news_item"]
    logger.info("graph: filter_node processing item '%s'", item.id)

    verdict, reasoning = await filter_agent.classify(item)

    return {
        "severity": verdict,
        "filter_reasoning": reasoning,
        "agents_invoked": state.get("agents_invoked", []) + ["FilterAgent"],
    }


async def historian_node(state: AgentState) -> dict:
    """Agent B: retrieve top-k historical parallels for a Critical item."""
    item = state["news_item"]
    logger.info("graph: historian_node processing item '%s'", item.id)

    events = await historian_agent.retrieve_historical_context(item)

    return {
        "historical_context": events,
        "agents_invoked": state.get("agents_invoked", []) + ["HistorianAgent"],
    }


# ---- Routing logic -----------------------------------------------------------

def route_after_filter(state: AgentState) -> str:
    """Conditional edge: route Critical items to the Historian, Noise to END.

    This function is called by LangGraph after filter_node completes.
    Returning the string name of the next node tells the graph where to go.
    """
    if state.get("severity") == "Critical":
        logger.debug("graph: routing to historian_node (Critical)")
        return "historian_node"
    logger.debug("graph: routing to END (Noise)")
    return END


# ---- Graph compilation (singleton) ------------------------------------------

def _build_graph() -> object:
    """Construct and compile the StateGraph exactly once."""
    builder = StateGraph(AgentState)

    builder.add_node("filter_node", filter_node)
    builder.add_node("historian_node", historian_node)

    builder.set_entry_point("filter_node")

    # Conditional edge: filter_node -> historian_node | END
    builder.add_conditional_edges(
        "filter_node",
        route_after_filter,
        {
            "historian_node": "historian_node",
            END: END,
        },
    )

    # historian_node always goes to END.
    builder.add_edge("historian_node", END)

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
        "agents_invoked": [],
    }

    final_state: AgentState = await GRAPH.ainvoke(initial_state)
    return final_state
