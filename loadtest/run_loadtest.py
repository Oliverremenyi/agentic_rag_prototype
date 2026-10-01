"""Simplified load test (constitution Principle VI).

Fires N queries (50-200) through the full agentic workflow sequentially,
records per-query latency, and reports summary statistics plus a bottleneck
hypothesis and optimization proposals. No external load-testing tool is
required since the system under test is an in-process Python call, not an
HTTP service.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.graph import run_agent  # noqa: E402

QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BOTTLENECK_NOTES = (
    "The dominant cost per query is local LLM decode time: classify_query, "
    "route_subquestion (per sub-question), and synthesize_answer each make at "
    "least one LLM call, so a compound question issues several sequential LLM "
    "calls before an answer is returned. A comparison question is the most "
    "expensive shape: classify_query, two route_subquestion + execute_retrieval "
    "round trips (one per fact), then one more route_subquestion pass before "
    "execute_tool can run (waiting on both dependencies to resolve) - "
    "execute_tool itself is pure Python arithmetic and adds negligible time. "
    "FAISS retrieval is sub-millisecond at this corpus size and is not the "
    "bottleneck in any question shape."
)

OPTIMIZATION_PROPOSALS = [
    "Cache route_subquestion's routing decision for repeated/near-duplicate "
    "sub-questions instead of re-invoking the LLM every time.",
    "Use a smaller/more aggressively quantized local model for the routing and "
    "classification calls (which only need a one-word/short-JSON output), "
    "reserving the larger model only for final answer synthesis.",
]


def run_loadtest(num_queries: int) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    latencies_ms: list[float] = []

    for i in range(num_queries):
        item = questions[i % len(questions)]
        start = time.perf_counter()
        run_agent(item["question"])
        latencies_ms.append((time.perf_counter() - start) * 1000)
        if (i + 1) % 10 == 0 or i + 1 == num_queries:
            print(f"Completed {i + 1}/{num_queries} queries")

    sorted_latencies = sorted(latencies_ms)

    def percentile(p: float) -> float:
        idx = min(int(len(sorted_latencies) * p), len(sorted_latencies) - 1)
        return sorted_latencies[idx]

    summary = {
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "num_queries": num_queries,
        "p50_ms": round(statistics.median(sorted_latencies), 1),
        "p95_ms": round(percentile(0.95), 1),
        "max_ms": round(max(sorted_latencies), 1),
        "mean_ms": round(statistics.mean(sorted_latencies), 1),
        "bottleneck_notes": BOTTLENECK_NOTES,
        "optimization_proposals": OPTIMIZATION_PROPOSALS,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{summary['run_id']}.json"
    out_path.write_text(
        json.dumps({**summary, "latencies_ms": latencies_ms}, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Results written to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=100)
    args = parser.parse_args()
    if not 50 <= args.num_queries <= 200:
        raise SystemExit("--num-queries must be between 50 and 200 per constitution Principle VI")
    run_loadtest(args.num_queries)
