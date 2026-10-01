from src.agent.graph import run_agent


def test_covered_direct_question_is_grounded_and_cited(
    use_fallback_classification_and_routing, fake_rag_retrieve
):
    fake_rag_retrieve({"environment": "Run uv sync then migrate."})

    response = run_agent("How do I set up my local development environment?")

    assert response["had_sufficient_info"] is True
    assert response["answer"] == "Run uv sync then migrate."
    assert "Acme Corp — Financial Highlights" in response["sources"]
    nodes_visited = [step["node"] for step in response["step_trace"]]
    assert "execute_retrieval" in nodes_visited
    assert "synthesize_answer" in nodes_visited


def test_uncovered_direct_question_reports_insufficient_information(
    use_fallback_classification_and_routing, fake_rag_retrieve
):
    fake_rag_retrieve({"environment": "Run uv sync then migrate."})

    response = run_agent("What is the capital of France?")

    assert response["had_sufficient_info"] is False
    assert "don't have enough information" in response["answer"]
