"""Node 4: execute_retrieval.

Resolves the current pending, retrieval-routed sub-question by calling the
`rag_retrieve` tool (which wraps the dedicated RAG subgraph).
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.agent.state import GraphState
from src.tools.rag_tool import rag_retrieve


def execute_retrieval(state: GraphState) -> GraphState:
    sub_questions = [dict(sq) for sq in state.get("sub_questions", [])]
    target = next(
        (sq for sq in sub_questions if sq["status"] == "pending" and sq["route"] == "retrieval"),
        None,
    )
    if target is None:
        return {}

    try:
        result = rag_retrieve(target["text"])
        success = True
        error_text = None
    except Exception as exc:  # noqa: BLE001 - tool failures must not crash the graph
        result = {"answer": "", "chunks": [], "found_relevant": False}
        success = False
        error_text = str(exc)

    if success and result["found_relevant"]:
        target["status"] = "resolved"
        target["answer"] = result["answer"]
        target["sources"] = sorted(
            {f"{c['company_name']} — {c['section_title']}" for c in result["chunks"]}
        )
        summary = f"Retrieved and answered '{target['text']}' from {len(result['chunks'])} source(s)"
    elif success:
        target["status"] = "resolved"
        target["answer"] = None
        target["sources"] = []
        summary = f"No relevant documentation found for '{target['text']}'"
    else:
        target["status"] = "failed"
        target["answer"] = None
        target["sources"] = []
        summary = f"Retrieval failed for '{target['text']}': {error_text}"

    updated = [
        target if sq["sub_id"] == target["sub_id"] else sq for sq in sub_questions
    ]

    return {
        "sub_questions": updated,
        "tool_calls": [
            {   "tool_name": "rag_retrieve",
                "input": target["text"],
                "output": result.get("answer") or None,
                "success": success,
                "sub_id": target["sub_id"],
            }
        ],
        "step_trace": [
            {
                "node": "execute_retrieval",
                "summary": summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }
