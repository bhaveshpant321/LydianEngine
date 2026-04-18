"""seed.py — One-time seeding script.

Reads market_history.csv, computes BGE-small embeddings, and writes the
records into LanceDB.  Run this before starting the API server, or let
the FastAPI lifespan call it automatically if the table is empty.

Usage:
    python -m sentinel.storage.seed
    # or
    SENTINEL_MARKET_HISTORY_CSV=/path/to/file python -m sentinel.storage.seed
"""
from __future__ import annotations

import asyncio
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

from lydian.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Schema for the LanceDB table ──────────────────────────────────────────────
SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string()),
        pa.field("headline", pa.string()),
        pa.field("date", pa.string()),
        pa.field("impact", pa.string()),
        pa.field("category", pa.string()),
        pa.field("full_text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)),
    ]
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    """Load and basic-validate market_history.csv rows."""
    required = {"event_id", "headline", "date", "impact", "category", "full_text"}
    rows: list[dict[str, str]] = []

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV missing required columns: {missing}")

        for i, row in enumerate(reader, start=2):  # 2 = first data row (1 = header)
            # Skip rows with any blank required field.
            blanks = [k for k in required if not row.get(k, "").strip()]
            if blanks:
                logger.warning("row %d: skipping — blank fields: %s", i, blanks)
                continue
            rows.append(dict(row))

    logger.info("csv: loaded %d valid rows from %s", len(rows), path)
    return rows


def _embed_batch(
    model: SentenceTransformer, texts: list[str], batch_size: int = 64
) -> list[list[float]]:
    """Embed texts in batches and return as Python lists of float32."""
    logger.info("embedding: encoding %d texts (batch_size=%d) …", len(texts), batch_size)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vectors]


def seed(force_reseed: bool = False) -> int:
    """Run the seeding pipeline synchronously.

    Args:
        force_reseed: If True, drop and recreate the table even if it exists.

    Returns:
        Number of rows written.
    """
    cfg = get_settings()
    csv_path = Path(cfg.market_history_csv)
    db_path = Path(cfg.lancedb_path)

    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))
    table_names = db.table_names()

    if cfg.lancedb_table_name in table_names and not force_reseed:
        tbl = db.open_table(cfg.lancedb_table_name)
        count = tbl.count_rows()
        if count > 0:
            logger.info(
                "table '%s' already has %d rows — skipping seed. "
                "Use force_reseed=True to override.",
                cfg.lancedb_table_name,
                count,
            )
            return count

    # Load embedding model.
    logger.info("loading embedding model: %s", cfg.embedding_model)
    model = SentenceTransformer(
        cfg.embedding_model, cache_folder=cfg.hf_cache_dir
    )

    rows = _load_csv(csv_path)
    if not rows:
        logger.error("no valid rows to embed — aborting")
        sys.exit(1)

    texts = [r["full_text"] for r in rows]
    vectors = _embed_batch(model, texts)

    records: list[dict[str, Any]] = []
    for row, vec in zip(rows, vectors):
        records.append(
            {
                "event_id": row["event_id"],
                "headline": row["headline"],
                "date": row["date"],
                "impact": row["impact"],
                "category": row["category"],
                "full_text": row["full_text"],
                "vector": vec,
            }
        )

    if cfg.lancedb_table_name in table_names:
        db.drop_table(cfg.lancedb_table_name)

    tbl = db.create_table(cfg.lancedb_table_name, data=records, schema=SCHEMA)
    written = tbl.count_rows()
    logger.info("seed complete: wrote %d rows to table '%s'", written, cfg.lancedb_table_name)
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed LanceDB with market history")
    parser.add_argument(
        "--force", action="store_true", help="Drop and recreate the table if it exists"
    )
    args = parser.parse_args()
    count = seed(force_reseed=args.force)
    print(f"\n✓ Done. {count} events indexed in LanceDB.")
