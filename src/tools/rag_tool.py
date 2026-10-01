"""The `rag_retrieve` tool: wraps the dedicated RAG subgraph for use by the
main agentic workflow."""

from __future__ import annotations

from src.rag.subgraph import RagResult, run_rag

TOOL_NAME = "rag_retrieve"


def rag_retrieve(query: str) -> RagResult:
    """Retrieve and synthesize an answer to `query` from the ingested document set."""
    return run_rag(query)
