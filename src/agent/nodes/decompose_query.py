"""Node 2: decompose_query.

Normalizes and de-duplicates the sub-questions produced by classify_query,
and short-circuits off-topic questions so no retrieval/tool work is wasted
on them (spec.md Edge Cases: off-topic input).
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.agent.state import GraphState, SubQuestion


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def decompose_query(state: GraphState) -> GraphState:
    if not state.get("is_on_topic", True):
        return {
            "sub_questions": [],
            "step_trace": [
                {
                    "node": "decompose_query",
                    "summary": "Question is off-topic; skipping decomposition and retrieval",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    seen: set[str] = set()
    deduped: list[SubQuestion] = []
    for sub_question in state.get("sub_questions", []):
        text = sub_question["text"].strip()
        key = _normalize(text)
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append({**sub_question, "text": text})

    if not deduped:
        deduped = [
            {
                "sub_id": "sq0",
                "text": state["question"],
                "route": None,
                "status": "pending",
                "depends_on": [],
                "answer": None,
                "sources": [],
            }
        ]

    return {
        "sub_questions": deduped,
        "step_trace": [
            {
                "node": "decompose_query",
                "summary": f"Normalized to {len(deduped)} sub-question(s)",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }
