import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Any

# Configure logging to be less verbose for the benchmark
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")

from lydian.agents import graph as agent_graph
from lydian.schemas.models import NewsItem
from lydian.storage import vector_store

async def run_benchmark():
    # 1. Load data
    data_path = Path("lydian/data/benchmark_data_50.json")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, "r") as f:
        items_raw = json.load(f)

    # 2. Init Vector Store (needed for Search node)
    print("Initializing Vector Store...")
    await vector_store.init_vector_store()

    results = []
    print(f"Starting benchmark on {len(items_raw)} items...\n")

    t_start = time.perf_counter()

    for i, raw in enumerate(items_raw):
        # Remove ground_truth for the model
        gt = raw.pop("ground_truth")
        item = NewsItem(**raw)
        
        print(f"[{i+1}/{len(items_raw)}] Processing: {item.headline[:50]}...", end="\r")
        
        t0 = time.perf_counter()
        state = await agent_graph.run(item)
        elapsed = (time.perf_counter() - t0) * 1000
        
        results.append({
            "id": item.id,
            "headline": item.headline,
            "prediction": state["severity"],
            "ground_truth": gt,
            "latency": elapsed,
            "max_sim": state.get("max_similarity", 0.0),
            "agents": state.get("agents_invoked", []),
            "black_swan": state.get("is_black_swan", False)
        })

    t_end = time.perf_counter()
    total_time = t_end - t_start

    # 3. Calculate Metrics
    tp = sum(1 for r in results if r["prediction"] == "Critical" and r["ground_truth"] == "Critical")
    fp = sum(1 for r in results if r["prediction"] == "Critical" and r["ground_truth"] == "Noise")
    tn = sum(1 for r in results if r["prediction"] == "Noise" and r["ground_truth"] == "Noise")
    fn = sum(1 for r in results if r["prediction"] == "Noise" and r["ground_truth"] == "Critical")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    # Short-circuit stats
    sc_total = sum(1 for r in results if "FilterAgent" not in r["agents"])
    sc_critical = sum(1 for r in results if "FilterAgent" not in r["agents"] and r["prediction"] == "Critical")
    sc_noise = sum(1 for r in results if "FilterAgent" not in r["agents"] and r["prediction"] == "Noise")
    
    # Black Swan stats
    black_swans = sum(1 for r in results if r["black_swan"])

    # 4. Generate Report
    print("\n" + "="*50)
    print(" LYDIAN ENGINE BENCHMARK REPORT ")
    print("="*50)
    print(f"Total Items:      {len(results)}")
    print(f"Total Time:       {total_time:.2f}s")
    print(f"Avg Latency:      {avg_latency:.2f}ms")
    print("-" * 50)
    print(f"Precision:        {precision:.2%}")
    print(f"Recall:           {recall:.2%}")
    print(f"F1 Score:         {f1:.4f}")
    print("-" * 50)
    print(f"Short-Circuited:  {sc_total} ({sc_total/len(results):.1%})")
    print(f"  -> SC Critical: {sc_critical}")
    print(f"  -> SC Noise:    {sc_noise}")
    print(f"Black Swans:      {black_swans}")
    print("="*50)

    # Detailed failures
    failures = [r for r in results if r["prediction"] != r["ground_truth"]]
    if failures:
        print("\nTOP FAILURES:")
        for f in failures[:5]:
            print(f"- [{f['prediction']} vs {f['ground_truth']}] {f['headline'][:60]}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
