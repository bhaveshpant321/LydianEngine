"""filter_agent.py — Agent A: The SLM Financial News Filter.

Classifies incoming news items as 'Critical' or 'Noise' using a sub-1B
instruction-tuned language model (default: Llama-3.2-1B-Instruct).

Design choices:
- Runs inference in asyncio.to_thread() to avoid blocking the event loop.
- Hard timeout enforced at the Python level; violations default to 'Noise'.
- Structured prompt + regex parser gives deterministic classification even
  when the LLM generates verbose reasoning before the verdict.
- Loads the model lazily on first call and caches it for the process lifetime.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from functools import lru_cache
from typing import Literal

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient

from lydian.core.config import get_settings
from lydian.schemas.models import NewsItem

logger = logging.getLogger(__name__)

# ---- Verdict type ------------------------------------------------------------
Verdict = Literal["Critical", "Noise"]

# ---- Regex to extract verdict from LLM output --------------------------------
# Matches 'VERDICT: Critical' or 'VERDICT: Noise' (case-insensitive).
_VERDICT_RE = re.compile(r"VERDICT:\s*(critical|noise)", re.IGNORECASE)

# ---- Classification prompt template ------------------------------------------
_SYSTEM_PROMPT = (
    "You are a real-time financial news triage system. "
    "Your ONLY job is to read a news snippet and decide if it is "
    "'Critical' or 'Noise' for institutional equity and fixed-income traders.\n\n"
    "CRITICAL means: the event has potential to move markets by >= 1% in the "
    "next 24 hours. Examples: surprise central bank decisions, systemic bank "
    "failures, geopolitical escalations, large earnings misses/beats for "
    "mega-cap stocks, major credit rating changes, commodity supply shocks.\n\n"
    "NOISE means: the event is routine, expected, or low-impact. Examples: "
    "minor data revisions within survey error, scheduled speeches reiterating "
    "existing policy, analyst notes without new thesis.\n\n"
    "Output format - you MUST end with exactly one of these two lines:\n"
    "VERDICT: Critical\n"
    "VERDICT: Noise"
)

_USER_TEMPLATE = (
    "SOURCE: {source}\n"
    "HEADLINE: {headline}\n"
    "BODY EXCERPT: {body_excerpt}\n"
    "TICKERS: {tickers}\n\n"
    "Classify this news item and end with your VERDICT."
)


@lru_cache(maxsize=1)
def _load_pipeline() -> object:
    """Load the causal LM pipeline once and cache it.
    Called lazily on the first classify() invocation."""
    cfg = get_settings()
    logger.info("filter_agent: loading model '%s' ...", cfg.filter_model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.filter_model_id, cache_dir=cfg.hf_cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.filter_model_id,
        cache_dir=cfg.hf_cache_dir,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.filter_max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )
    logger.info("filter_agent: model loaded OK")
    return pipe


async def prewarm() -> None:
    """Pre-load the model into memory (local mode only).
    Must be called during application startup to avoid first-request latency.
    """
    cfg = get_settings()
    if cfg.inference_mode == "local":
        logger.info("filter_agent: pre-warming local model ...")
        await asyncio.to_thread(_load_pipeline)
    else:
        logger.info("filter_agent: cloud mode active — skipping local pre-warm")


def _build_prompt(item: NewsItem) -> str:
    """Construct the instruct-formatted prompt for Llama-3."""
    tickers = ", ".join(item.tickers) if item.tickers else "N/A"
    body_excerpt = item.body[:512].replace("\n", " ")
    user_msg = _USER_TEMPLATE.format(
        source=item.source,
        headline=item.headline,
        body_excerpt=body_excerpt,
        tickers=tickers,
    )
    # Llama-3 chat format (works for most instruction-tuned variants).
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{_SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _parse_verdict(raw_output: str) -> tuple[Verdict, str]:
    """Extract the structured verdict from raw LLM text.

    Returns a (verdict, reasoning) tuple. Falls back to ('Noise', ...) if
    the model fails to produce a parseable verdict — prevents false positives
    on parsing failures.
    """
    match = _VERDICT_RE.search(raw_output)
    if match:
        verdict: Verdict = "Critical" if match.group(1).lower() == "critical" else "Noise"
        # Everything before the VERDICT line is the reasoning.
        reasoning = raw_output[: match.start()].strip() or "No reasoning provided."
        return verdict, reasoning

    logger.warning(
        "filter_agent: no VERDICT found in output — defaulting to Noise. "
        "Output prefix: %.120s",
        raw_output,
    )
    return "Noise", f"Parser fallback. Raw output: {raw_output[:200]}"


async def _call_cloud_api(item: NewsItem) -> tuple[Verdict, str]:
    """Call the HuggingFace Serverless Inference API with a Mock fallback."""
    cfg = get_settings()

    # Rule-based fallback for testing without cloud tokens
    def _mock_classify(news: NewsItem) -> tuple[Verdict, str]:
        critical_keywords = {"fed", "rate", "bank", "inflation", "cpi", "war", "crash", "earnings"}
        text = (news.headline + " " + news.body).lower()
        if any(kw in text for kw in critical_keywords):
            return "Critical", "Mock Filter: Detected high-impact keywords."
        return "Noise", "Mock Filter: No high-impact keywords detected."

    client = InferenceClient(
        model=cfg.filter_model_id,
        token=cfg.hf_token,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(
            source=item.source,
            headline=item.headline,
            body_excerpt=item.body[:512],
            tickers=", ".join(item.tickers) if item.tickers else "N/A"
        )}
    ]

    try:
        t0 = time.perf_counter()
        # Using chat_completion as required by the Llama-3.2-Instruct API provider
        response = await asyncio.to_thread(
            client.chat_completion,
            messages=messages,
            max_tokens=cfg.filter_max_new_tokens,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        content = response.choices[0].message.content
        logger.info("filter_agent: cloud inference successful (%.1f ms)", elapsed_ms)
        return _parse_verdict(content)
    except Exception as exc:
        if "403" in str(exc) or "401" in str(exc) or "permissions" in str(exc).lower():
            logger.warning("filter_agent: Cloud API Auth failed (403/401) -- using Deterministic Mock Fallback")
            return _mock_classify(item)
        
        logger.error("filter_agent: cloud API unexpected error: %s", exc)
        return "Noise", f"Cloud API error: {exc}"


def _run_inference(item: NewsItem) -> tuple[Verdict, str]:
    """Synchronous inference entry point — called via asyncio.to_thread (local mode)."""
    pipe = _load_pipeline()
    prompt = _build_prompt(item)

    t0 = time.perf_counter()
    results = pipe(prompt)  # type: ignore[operator]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    raw: str = results[0]["generated_text"] if results else ""
    logger.debug(
        "filter_agent: inference complete in %.1f ms for item '%s'",
        elapsed_ms,
        item.id,
    )
    return _parse_verdict(raw)


async def classify(item: NewsItem) -> tuple[Verdict, str]:
    """Async entry point for Agent A.

    Toggles between 'local' and 'cloud' inference. On 'local', enforces
    a hard timeout to preserve SLA. Cloud API has its own network timeout.
    """
    cfg = get_settings()

    if cfg.inference_mode == "cloud":
        try:
            return await _call_cloud_api(item)
        except Exception as exc:
            logger.error("filter_agent: cloud API error: %s", exc)
            return "Noise", f"Cloud API error: {exc}"

    # Local mode with timeout
    timeout_s = cfg.filter_timeout_ms / 1000.0
    try:
        verdict, reasoning = await asyncio.wait_for(
            asyncio.to_thread(_run_inference, item),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "filter_agent: local timeout (%.0f ms) exceeded for item '%s' — "
            "defaulting to Noise",
            cfg.filter_timeout_ms,
            item.id,
        )
        return "Noise", (
            f"Local inference timed out after {cfg.filter_timeout_ms:.0f} ms. "
            "Defaulted to Noise to preserve latency SLA."
        )
    except Exception as exc:
        logger.error(
            "filter_agent: unexpected local error for item '%s': %s",
            item.id,
            exc,
        )
        return "Noise", f"Local inference error: {exc}"

    return verdict, reasoning
