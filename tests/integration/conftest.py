"""Shared fixtures for full-graph integration tests.

All tests here run the real compiled graph (src.agent.graph.run_agent) but
stub out the LLM calls and the RAG retrieval tool so tests are deterministic
and require no network access to Ollama. `compare_financial_values` is left
real since it is pure Python arithmetic with no external dependency.
"""

from __future__ import annotations

import pytest

from src.agent.nodes import classify_query as classify_query_mod
from src.agent.nodes import execute_retrieval as execute_retrieval_mod
from src.agent.nodes import route_subquestion as route_subquestion_mod


@pytest.fixture
def use_fallback_classification_and_routing(monkeypatch):
    """Force classify_query and route_subquestion onto their deterministic
    keyword-based fallback paths instead of calling a real LLM."""
    monkeypatch.setattr(classify_query_mod, "invoke_llm", lambda prompt: "not json at all")
    monkeypatch.setattr(route_subquestion_mod, "invoke_llm", lambda prompt: "not json at all")


@pytest.fixture
def fake_rag_retrieve(monkeypatch):
    """Install a fake rag_retrieve. `covered_keyword` decides which queries
    are treated as covered by the documentation."""

    def _install(answers_by_keyword: dict[str, str]):
        def fake(query: str):
            lowered = query.lower()
            for keyword, answer in answers_by_keyword.items():
                if keyword in lowered:
                    return {
                        "answer": answer,
                        "chunks": [
                            {
                                "chunk_id": "c0",
                                "doc_id": "d0",
                                "company_name": "Acme Corp",
                                "section_title": "Financial Highlights",
                                "text": answer,
                            }
                        ],
                        "found_relevant": True,
                    }
            return {"answer": "", "chunks": [], "found_relevant": False}

        monkeypatch.setattr(execute_retrieval_mod, "rag_retrieve", fake)

    return _install
