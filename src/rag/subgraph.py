"""Dedicated, modular RAG subgraph (constitution Principle II).

This is a self-contained LangGraph graph with its own two internal nodes
(retrieve -> synthesize). It is invoked by the main workflow's `rag_retrieve`
tool and is NOT counted toward the main graph's 5-node minimum.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src import config
from src.llm.local_llm import invoke_llm
from src.rag.retriever import get_retriever


class RagChunkResult(TypedDict):
    chunk_id: str
    doc_id: str
    company_name: str
    section_title: str
    text: str


class RagState(TypedDict, total=False):
    query: str
    chunks: list[RagChunkResult]
    answer: str
    found_relevant: bool


def _retrieve_passages(state: RagState) -> RagState:
    retriever = get_retriever()
    hits = retriever.search(state["query"])
    relevant = [
        (chunk, score) for chunk, score in hits if score >= config.RETRIEVAL_RELEVANCE_THRESHOLD
    ]
    chunks: list[RagChunkResult] = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "company_name": chunk.company_name,
            "section_title": chunk.section_title,
            "text": chunk.text,
        }
        for chunk, _ in relevant
    ]
    return {"chunks": chunks, "found_relevant": len(chunks) > 0}


def _synthesize_passage_answer(state: RagState) -> RagState:
    if not state.get("found_relevant"):
        return {"answer": ""}

    context = "\n\n".join(
        f"[Source: {c['company_name']} — {c['section_title']}]\n{c['text']}"
        for c in state["chunks"]
    )
    prompt = (
        "You are answering a question using only the context below. "
        "If the context does not contain the answer, say so plainly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        "Answer concisely, citing source titles in brackets where relevant:"
    )
    answer = invoke_llm(prompt)
    return {"answer": answer}


def _build_rag_subgraph():
    graph = StateGraph(RagState)
    graph.add_node("retrieve_passages", _retrieve_passages)
    graph.add_node("synthesize_passage_answer", _synthesize_passage_answer)
    graph.set_entry_point("retrieve_passages")
    graph.add_edge("retrieve_passages", "synthesize_passage_answer")
    graph.add_edge("synthesize_passage_answer", END)
    return graph.compile()


_rag_app = _build_rag_subgraph()


class RagResult(TypedDict):
    answer: str
    chunks: list[RagChunkResult]
    found_relevant: bool


def run_rag(query: str) -> RagResult:
    """Public entry point used by the rag_retrieve tool (contracts/agent-invocation.md)."""
    result = _rag_app.invoke({"query": query})
    return {
        "answer": result.get("answer", ""),
        "chunks": result.get("chunks", []),
        "found_relevant": result.get("found_relevant", False),
    }
