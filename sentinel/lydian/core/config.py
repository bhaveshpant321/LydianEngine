from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent.resolve()

class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables.

    All variables are prefixed with ``SENTINEL_`` in the environment
    (e.g. ``SENTINEL_LANCEDB_PATH``).  Type coercion and validation are
    handled by Pydantic.
    """

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    lancedb_path: str = Field(
        default=str(BASE_DIR / "data" / "lancedb"),
        description="Filesystem path for LanceDB (disk-native vector store)",
    )
    lancedb_table_name: str = Field(
        default="market_history",
        description="Name of the LanceDB table that holds historical embeddings",
    )
    market_history_csv: str = Field(
        default=str(BASE_DIR / "data" / "market_history.csv"),
        description="Path to the CSV file used to seed the vector store",
    )

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model id for the sentence-transformers embedder",
    )
    embedding_dim: int = Field(
        default=384,
        description="Output dimension of the chosen embedding model",
    )
    rag_top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of similar historical events to retrieve per query",
    )

    # ── LLM filter agent ─────────────────────────────────────────────────────
    filter_model_id: str = Field(
        default="google/gemma-2-2b-it",
        description="HuggingFace model id for Agent A (the SLM filter/classifier)",
    )
    filter_max_new_tokens: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Maximum tokens Agent A may generate per classification",
    )
    filter_timeout_ms: float = Field(
        default=150.0,
        gt=0,
        description=(
            "Hard timeout for Agent A inference in milliseconds. "
            "Exceeded items are classified as 'Noise' (safe default)."
        ),
    )

    # ── API / performance ────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    request_timeout_ms: float = Field(
        default=200.0,
        description="Target end-to-end latency SLA for /analyze (ms). "
        "Violations are logged as warnings.",
    )

    # ── HuggingFace ───────────────────────────────────────────────────────────
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_cache_dir: str = Field(
        default=str(BASE_DIR / "data" / "models"),
        description="Local cache directory for downloaded HuggingFace models",
    )
    inference_mode: Literal["local", "cloud"] = Field(
        default="local",
        description="Toggle between 'local' (transformers) and 'cloud' (HF Inference API)",
    )
    feed_mode: Literal["mock", "api"] = Field(
        default="mock",
        description="Ingestion mode: 'mock' uses testdata; 'api' expects live Go service",
    )

    @field_validator("embedding_dim")
    @classmethod
    def validate_dim(cls, v: int) -> int:
        allowed = {128, 256, 384, 512, 768, 1024}
        if v not in allowed:
            raise ValueError(f"embedding_dim must be one of {allowed}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.  Using lru_cache ensures
    the .env file is read exactly once per process, avoiding I/O overhead
    on every request."""
    return Settings()
