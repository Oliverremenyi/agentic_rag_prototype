"""Node 3: route_subquestion.

Autonomously routes the next ready, pending sub-question to retrieval or a
direct answer (constitution Principle I: autonomous decision-making via
conditional routing). A comparison sub-question's route is pre-set to "tool"
by classify_query and is never reclassified here. This node is revisited in a
loop (see src/agent/graph.py) until every sub-question is resolved or failed.

A sub-question with `depends_on` (the comparison step) only becomes eligible
once every dependency is `resolved`; if a dependency ends up `failed`, the
dependent sub-question is marked `failed` too rather than waiting forever.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from src.agent.state import GraphState, SubQuestion
from src.llm.local_llm import invoke_llm

_ROUTE_PROMPT = """Classify how to answer this question about a company's annual \
report. Respond with ONLY one word: "retrieval" or "direct".

- "retrieval": the question asks about a documented fact, figure, or disclosure that \
would be found in a company's annual report.
- "direct": greetings, thanks, or messages answerable without looking anything up.

Question: {text}
"""


def _fallback_route(text: str) -> str:
    if len(text.split()) <= 3:
        return "direct"
    return "retrieval"


def _classify_route(text: str) -> str:
    try:
        raw = invoke_llm(_ROUTE_PROMPT.format(text=text))
    except Exception:  # noqa: BLE001 - LLM outage must not crash the graph
        return _fallback_route(text)
    match = re.search(r"retrieval|direct", raw.lower())
    if match:
        return match.group(0)
    return _fallback_route(text)


def _is_ready(sub_question: SubQuestion, by_id: dict[str, SubQuestion]) -> bool:
    return all(by_id[dep]["status"] == "resolved" for dep in sub_question.get("depends_on", []))


def _has_failed_dependency(sub_question: SubQuestion, by_id: dict[str, SubQuestion]) -> bool:
    return any(by_id[dep]["status"] == "failed" for dep in sub_question.get("depends_on", []))


def route_subquestion(state: GraphState) -> GraphState:
    sub_questions = [dict(sq) for sq in state.get("sub_questions", [])]
    by_id = {sq["sub_id"]: sq for sq in sub_questions}

    # A sub-question whose dependency failed can never become ready - fail it
    # now instead of looping forever waiting on a value that will never arrive.
    blocked = next(
        (sq for sq in sub_questions if sq["status"] == "pending" and _has_failed_dependency(sq, by_id)),
        None,
    )
    if blocked is not None:
        blocked["status"] = "failed"
        blocked["answer"] = None
        summary = (
            f"Cannot resolve '{blocked['text']}': a required prior value could not be determined"
        )
        updated = [blocked if sq["sub_id"] == blocked["sub_id"] else sq for sq in sub_questions]
        return {
            "sub_questions": updated,
            "step_trace": [
                {
                    "node": "route_subquestion",
                    "summary": summary,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    pending = next(
        (sq for sq in sub_questions if sq["status"] == "pending" and _is_ready(sq, by_id)),
        None,
    )

    if pending is None:
        return {
            "step_trace": [
                {
                    "node": "route_subquestion",
                    "summary": "No pending sub-questions remain",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ]
        }

    if pending["route"] is None:
        pending["route"] = _classify_route(pending["text"])

    summary = f"Routed sub-question '{pending['text']}' to {pending['route']}"

    if pending["route"] == "direct":
        try:
            answer = invoke_llm(
                "Answer this brief chat message in one short sentence: "
                f"{pending['text']}"
            )
        except Exception:  # noqa: BLE001 - LLM outage must not crash the graph
            answer = "Hello! (LLM unavailable for a fuller reply right now.)"
        pending["status"] = "resolved"
        pending["answer"] = answer
        summary = f"Directly answered '{pending['text']}'"

    updated = [
        pending if sq["sub_id"] == pending["sub_id"] else sq for sq in sub_questions
    ]

    return {
        "sub_questions": updated,
        "step_trace": [
            {
                "node": "route_subquestion",
                "summary": summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }


def route_decision(state: GraphState) -> str:
    """Conditional-edge function: decide where to go after route_subquestion."""
    sub_questions = state.get("sub_questions", [])
    by_id = {sq["sub_id"]: sq for sq in sub_questions}
    pending = next(
        (sq for sq in sub_questions if sq["status"] == "pending" and _is_ready(sq, by_id)),
        None,
    )
    if pending is None:
        return "synthesize_answer"
    if pending["route"] == "retrieval":
        return "execute_retrieval"
    if pending["route"] == "tool":
        return "execute_tool"
    # "direct" was already resolved in-place; loop back to pick up the next one
    return "route_subquestion"
