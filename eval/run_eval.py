"""Functional evaluation runner (constitution Principle VI).

Runs the fixed 10-20 question set in eval/questions.json through the full
agentic workflow and scores each answer by simple keyword-containment
against `expected_contains` (any one match counts as a pass — see
research.md decision 6 for why a stricter LLM-as-judge was deferred).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.graph import run_agent  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def score(answer: str, expected_contains: list[str]) -> bool:
    lowered = answer.lower()
    return any(phrase.lower() in lowered for phrase in expected_contains)


def run_eval() -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results = []

    for item in questions:
        start = time.perf_counter()
        response = run_agent(item["question"])
        latency_ms = (time.perf_counter() - start) * 1000
        passed = score(response["answer"], item["expected_contains"])
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "answer": response["answer"],
                "passed": passed,
                "latency_ms": round(latency_ms, 1),
            }
        )
        print(f"{item['id']} [{'PASS' if passed else 'FAIL'}] {item['question']}")

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    summary = {
        "total": total,
        "passed": passed_count,
        "pass_rate": round(passed_count / total, 3) if total else 0.0,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
    )

    print(f"\nPass rate: {passed_count}/{total} ({summary['pass_rate']:.0%})")
    print(f"Results written to {out_path}")
    return summary


if __name__ == "__main__":
    run_eval()
