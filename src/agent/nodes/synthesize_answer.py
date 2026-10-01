"""Node 6: synthesize_answer.

Combines all resolved (and unresolved) sub-question results into one final,
coherent answer, aggregating citations from retrieval and incorporating tool
results.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.agent.state import GraphState
from src.llm.local_llm import invoke_llm

_OFF_TOPIC_ANSWER = (
    "I can only help with questions about the curated set of companies' annual "
    "reports — that question is outside what I can answer."
)

_COMBINE_PROMPT = """Combine the following sub-answers into one short, coherent \
response to the user's original question. If a sub-answer says information is \
missing, mention that gap explicitly rather than guessing.

Original question: {question}

Sub-answers:
{sub_answers}

Combined answer:"""


def synthesize_answer(state: GraphState) -> GraphState:
    if not state.get("is_on_topic", True) or not state.get("sub_questions"):
        return {
            "final_answer": _OFF_TOPIC_ANSWER,
            "had_sufficient_info": False,
            "sources": [],
            "step_trace": [
                {
                    "node": "synthesize_answer",
                    "summary": "Returned off-topic response without synthesis",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    sub_questions = state["sub_questions"]
    resolved_with_answer = [sq for sq in sub_questions if sq.get("answer")]
    had_sufficient_info = len(resolved_with_answer) > 0
    sources = sorted({s for sq in sub_questions for s in sq.get("sources", [])})

    comparison_sq = next((sq for sq in sub_questions if sq.get("depends_on")), None)
    if comparison_sq is not None:
        if comparison_sq.get("answer"):
            final_answer = comparison_sq["answer"]
            had_sufficient_info = True
        else:
            by_id = {sq["sub_id"]: sq for sq in sub_questions}
            missing_facts = [
                by_id[dep]["text"]
                for dep in comparison_sq.get("depends_on", [])
                if not by_id.get(dep, {}).get("answer")
            ]
            final_answer = (
                "I can't complete that comparison because I don't have a numeric "
                "value for: " + "; ".join(missing_facts or ["one of the required figures"]) + "."
            )
            had_sufficient_info = False
        return {
            "final_answer": final_answer,
            "had_sufficient_info": had_sufficient_info,
            "sources": sources,
            "step_trace": [
                {
                    "node": "synthesize_answer",
                    "summary": "Synthesized final answer from the comparison sub-question",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    if len(sub_questions) == 1:
        sub = sub_questions[0]
        if sub.get("answer"):
            final_answer = sub["answer"]
        else:
            final_answer = (
                "I don't have enough information in the current documentation or "
                f"tools to answer: '{sub['text']}'."
            )
    else:
        sub_answers_text = "\n".join(
            f"- Q: {sq['text']}\n  A: {sq.get('answer') or 'No information found.'}"
            for sq in sub_questions
        )
        try:
            final_answer = invoke_llm(
                _COMBINE_PROMPT.format(question=state["question"], sub_answers=sub_answers_text)
            )
        except Exception:  # noqa: BLE001 - LLM outage must not crash the graph
            final_answer = "Here is what I found for each part of your question:\n" + sub_answers_text

    return {
        "final_answer": final_answer,
        "had_sufficient_info": had_sufficient_info,
        "sources": sources,
        "step_trace": [
            {
                "node": "synthesize_answer",
                "summary": (
                    f"Synthesized final answer from {len(sub_questions)} sub-question(s), "
                    f"{len(sources)} source(s) cited"
                ),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }
