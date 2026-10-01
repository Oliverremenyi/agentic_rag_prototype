from src.agent.nodes import classify_query as classify_query_mod
from src.agent.graph import run_agent

_COMPARISON_CLASSIFICATION = (
    '{"on_topic": true, "kind": "comparison", '
    '"fact_a": "What was Company A revenue?", "fact_b": "What was Company B revenue?"}'
)


def test_comparison_question_with_both_values_present(
    monkeypatch, use_fallback_classification_and_routing, fake_rag_retrieve
):
    monkeypatch.setattr(classify_query_mod, "invoke_llm", lambda prompt: _COMPARISON_CLASSIFICATION)
    fake_rag_retrieve(
        {
            "company a": "Company A's revenue was $50 million.",
            "company b": "Company B's revenue was $20 million.",
        }
    )

    response = run_agent(
        "Which had higher revenue, Company A or Company B, and by how much?"
    )

    assert response["had_sufficient_info"] is True
    tool_calls = [c for c in response["tool_calls"] if c["tool_name"] == "compare_financial_values"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["success"] is True
    assert "50,000,000.00" in response["answer"]
    assert "20,000,000.00" in response["answer"]


def test_comparison_question_with_missing_value_reports_gracefully(
    monkeypatch, use_fallback_classification_and_routing, fake_rag_retrieve
):
    monkeypatch.setattr(classify_query_mod, "invoke_llm", lambda prompt: _COMPARISON_CLASSIFICATION)
    fake_rag_retrieve({"company a": "Company A's revenue was $50 million."})

    response = run_agent(
        "Which had higher revenue, Company A or Company B, and by how much?"
    )

    assert response["had_sufficient_info"] is False
    tool_calls = [c for c in response["tool_calls"] if c["tool_name"] == "compare_financial_values"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["success"] is False
    assert "can't complete that comparison" in response["answer"]
