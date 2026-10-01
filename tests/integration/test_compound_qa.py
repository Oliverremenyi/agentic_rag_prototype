from src.agent.nodes import synthesize_answer as synthesize_answer_mod
from src.agent.graph import run_agent


def _echo_combine(prompt: str) -> str:
    marker = "Sub-answers:\n"
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\n\nCombined answer:")
    return "Combined: " + prompt[start:end]


def test_compound_question_partially_covered(
    use_fallback_classification_and_routing, fake_rag_retrieve, monkeypatch
):
    monkeypatch.setattr(synthesize_answer_mod, "invoke_llm", _echo_combine)
    fake_rag_retrieve({"environment": "Run uv sync then migrate."})

    response = run_agent(
        "How do I set up my local development environment, and what is the capital of Mars?"
    )

    sub_texts = [sq["text"] for sq in response["sub_questions"]]
    assert len(sub_texts) == 2

    assert "Run uv sync then migrate." in response["answer"]
    assert "No information found." in response["answer"]
    # The covered sub-question contributes a source; the uncovered one does not
    assert "Acme Corp — Financial Highlights" in response["sources"]
