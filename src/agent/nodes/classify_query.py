"""Node 1: classify_query.

Determines whether the incoming question is on-topic for the curated annual
report corpus, and splits it into either independent sub-questions or, for a
comparison question, two retrieval facts plus a dependent comparison step
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from src.agent.state import GraphState, SubQuestion
from src.llm.local_llm import invoke_llm

_OFF_TOPIC_MARKERS = (
    "weather",
    "joke",
    "recipe",
    "sports score",
    "movie recommendation",
)

_CLASSIFY_PROMPT = """You help a user analyze a curated set of companies' annual \
reports, answering factual questions and comparisons between companies or metrics.

Given the user's message, respond with ONLY a JSON object of ONE of these two forms:

Simple (a direct question, or a compound question whose parts do not need to be \
numerically compared against each other):
{{"on_topic": true|false, "kind": "simple", "sub_questions": ["...", "..."]}}

Comparison (the user explicitly wants two values compared - which is higher, by how \
much, or a ratio):
{{"on_topic": true, "kind": "comparison", "fact_a": "<question that retrieves the \
first value>", "fact_b": "<question that retrieves the second value>"}}

- "on_topic" is false only for messages entirely unrelated to companies, annual \
reports, or financial questions (e.g. small talk, weather).
- For "simple", a single already-atomic question is a list with just that one item.

User message: {question}
"""


def _fallback_classification(question: str) -> dict:
    lowered = question.lower()
    if any(marker in lowered for marker in _OFF_TOPIC_MARKERS):
        return {"on_topic": False, "kind": "simple", "sub_questions": []}
    parts = re.split(r"\band\b|\?\s*(?=\w)", question)
    parts = [p.strip(" ?").strip() for p in parts if p.strip(" ?").strip()]
    return {"on_topic": True, "kind": "simple", "sub_questions": parts or [question]}


def _parse_llm_classification(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        on_topic = bool(data["on_topic"])
        kind = data.get("kind", "simple")
        if kind == "comparison":
            fact_a = str(data["fact_a"]).strip()
            fact_b = str(data["fact_b"]).strip()
            if not fact_a or not fact_b:
                return None
            return {"on_topic": on_topic, "kind": "comparison", "fact_a": fact_a, "fact_b": fact_b}
        sub_questions = [str(q) for q in data.get("sub_questions", [])]
        if on_topic and not sub_questions:
            return None
        return {"on_topic": on_topic, "kind": "simple", "sub_questions": sub_questions}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _build_simple_sub_questions(sub_texts: list[str]) -> list[SubQuestion]:
    return [
        {
            "sub_id": f"sq{i}",
            "text": text,
            "route": None,
            "status": "pending",
            "depends_on": [],
            "answer": None,
            "sources": [],
        }
        for i, text in enumerate(sub_texts)
    ]


def _build_comparison_sub_questions(fact_a: str, fact_b: str) -> list[SubQuestion]:
    sub_a: SubQuestion = {
        "sub_id": "sq0",
        "text": fact_a,
        "route": None,
        "status": "pending",
        "depends_on": [],
        "answer": None,
        "sources": [],
    }
    sub_b: SubQuestion = {
        "sub_id": "sq1",
        "text": fact_b,
        "route": None,
        "status": "pending",
        "depends_on": [],
        "answer": None,
        "sources": [],
    }
    comparison: SubQuestion = {
        "sub_id": "sq2",
        "text": f"Compare: '{fact_a}' vs '{fact_b}'",
        "route": "tool",
        "status": "pending",
        "depends_on": ["sq0", "sq1"],
        "answer": None,
        "sources": [],
    }
    return [sub_a, sub_b, comparison]


def classify_query(state: GraphState) -> GraphState:
    question = state["question"]
    try:
        raw = invoke_llm(_CLASSIFY_PROMPT.format(question=question))
    except Exception:  # noqa: BLE001 - LLM outage must not crash the graph
        raw = ""
    parsed = _parse_llm_classification(raw) or _fallback_classification(question)

    on_topic = parsed["on_topic"]
    if parsed["kind"] == "comparison":
        sub_questions = _build_comparison_sub_questions(parsed["fact_a"], parsed["fact_b"])
        summary = (
            f"Classified as on-topic comparison: '{parsed['fact_a']}' vs '{parsed['fact_b']}'"
        )
    else:
        sub_questions = _build_simple_sub_questions(parsed["sub_questions"])
        summary = (
            f"Classified as {'on-topic' if on_topic else 'off-topic'}, "
            f"{len(sub_questions)} sub-question(s) identified"
        )

    return {
        "is_on_topic": on_topic,
        "sub_questions": sub_questions,
        "step_trace": [
            {
                "node": "classify_query",
                "summary": summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }
