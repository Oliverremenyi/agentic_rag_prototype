"""Node 5: execute_tool.

Resolves the comparison sub-question by parsing the two numeric values its
`depends_on` retrieval sub-questions produced, then calling the non-retrieval
`compare_financial_values` tool. Never fetches data itself - if either
dependency's answer is missing or non-numeric, the comparison fails
gracefully instead of guessing.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from src.agent.state import GraphState
from src.tools.financial_tool import compare_financial_values

# Requires either a leading currency symbol or a trailing magnitude word, so a
# bare year mentioned in prose (e.g. "net revenues in 2021 were $13.1 billion")
# is never mistaken for the actual figure - a real failure found when running
# this against a real LLM's retrieved text (research.md decision 6 follow-up).
_QUALIFIED_NUMBER_RE = re.compile(
    r"(?:\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)(?:\s*(billion|million|thousand|bn|mn|k)\b)?"
    r"|([0-9][0-9,]*(?:\.[0-9]+)?)\s*(billion|million|thousand|bn|mn|k)\b)",
    re.IGNORECASE,
)
# Fallback: any plain number, used only if no qualified match exists.
_PLAIN_NUMBER_RE = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?")
_MULTIPLIERS = {
    "billion": 1e9,
    "bn": 1e9,
    "million": 1e6,
    "mn": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _parse_number(text: str | None) -> float | None:
    if not text:
        return None

    match = _QUALIFIED_NUMBER_RE.search(text)
    if match:
        raw = match.group(1) or match.group(3)
        suffix = match.group(2) or match.group(4)
        value = _to_float(raw)
        if value is not None and suffix:
            value *= _MULTIPLIERS[suffix.lower()]
        if value is not None:
            return value

    # No dollar sign or magnitude word anywhere - fall back to a plain number,
    # but skip bare 4-digit numbers that look like a year rather than a value.
    for candidate in _PLAIN_NUMBER_RE.finditer(text):
        raw = candidate.group(0)
        if _YEAR_RE.match(raw.replace(",", "")):
            continue
        return _to_float(raw)
    return None


def execute_tool(state: GraphState) -> GraphState:
    sub_questions = [dict(sq) for sq in state.get("sub_questions", [])]
    by_id = {sq["sub_id"]: sq for sq in sub_questions}
    target = next(
        (sq for sq in sub_questions if sq["status"] == "pending" and sq["route"] == "tool"),
        None,
    )
    if target is None:
        return {}

    dep_ids = target.get("depends_on", [])
    fact_a = by_id.get(dep_ids[0]) if len(dep_ids) > 0 else None
    fact_b = by_id.get(dep_ids[1]) if len(dep_ids) > 1 else None

    label_a = fact_a["text"] if fact_a else "value A"
    label_b = fact_b["text"] if fact_b else "value B"
    value_a = _parse_number(fact_a["answer"]) if fact_a else None
    value_b = _parse_number(fact_b["answer"]) if fact_b else None

    tool_input = {"value_a": value_a, "value_b": value_b, "label_a": label_a, "label_b": label_b}

    if value_a is None or value_b is None:
        missing = [
            label for label, value in ((label_a, value_a), (label_b, value_b)) if value is None
        ]
        target["status"] = "failed"
        target["answer"] = None
        target["sources"] = []
        summary = (
            "Comparison could not be completed: missing or non-numeric value for "
            + ", ".join(missing)
        )
        tool_call = {
            "tool_name": "compare_financial_values",
            "input": tool_input,
            "output": None,
            "success": False,
            "sub_id": target["sub_id"],
        }
    else:
        result = compare_financial_values(value_a, value_b, label_a, label_b)
        if result["greater"] == "equal":
            answer = f"'{label_a}' and '{label_b}' are equal ({value_a:,.2f})."
        else:
            winner_label, winner_value = (
                (label_a, value_a) if result["greater"] == "a" else (label_b, value_b)
            )
            loser_label, loser_value = (
                (label_b, value_b) if result["greater"] == "a" else (label_a, value_a)
            )
            ratio_note = f", a {result['ratio']:.2f}x ratio" if result["ratio"] is not None else ""
            answer = (
                f"'{winner_label}' ({winner_value:,.2f}) is greater than "
                f"'{loser_label}' ({loser_value:,.2f}) by {abs(result['difference']):,.2f}"
                f"{ratio_note}."
            )
        target["status"] = "resolved"
        target["answer"] = answer
        target["sources"] = sorted(
            {*(fact_a.get("sources", []) if fact_a else []), *(fact_b.get("sources", []) if fact_b else [])}
        )
        summary = f"Compared '{label_a}' vs '{label_b}': {answer}"
        tool_call = {
            "tool_name": "compare_financial_values",
            "input": tool_input,
            "output": dict(result),
            "success": True,
            "sub_id": target["sub_id"],
        }

    updated = [
        target if sq["sub_id"] == target["sub_id"] else sq for sq in sub_questions
    ]

    return {
        "sub_questions": updated,
        "tool_calls": [tool_call],
        "step_trace": [
            {
                "node": "execute_tool",
                "summary": summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }
