from src.agent.nodes import classify_query as classify_query_mod
from src.agent.nodes import synthesize_answer as synthesize_answer_mod
from src.agent.nodes.classify_query import classify_query
from src.agent.nodes.synthesize_answer import synthesize_answer


def test_classify_query_falls_back_to_heuristics_on_bad_llm_output(monkeypatch):
    monkeypatch.setattr(classify_query_mod, "invoke_llm", lambda prompt: "not json at all")

    state = {"question": "How do I set up my local development environment?"}
    result = classify_query(state)

    assert result["is_on_topic"] is True
    assert len(result["sub_questions"]) == 1
    assert result["sub_questions"][0]["status"] == "pending"
    assert result["step_trace"][0]["node"] == "classify_query"


def test_classify_query_detects_off_topic_via_fallback(monkeypatch):
    monkeypatch.setattr(classify_query_mod, "invoke_llm", lambda prompt: "not json at all")

    state = {"question": "What's a good joke to tell at lunch?"}
    result = classify_query(state)

    assert result["is_on_topic"] is False


def test_classify_query_uses_llm_json_when_valid(monkeypatch):
    monkeypatch.setattr(
        classify_query_mod,
        "invoke_llm",
        lambda prompt: '{"on_topic": true, "sub_questions": ["a", "b"]}',
    )

    result = classify_query({"question": "irrelevant, patched"})

    assert result["is_on_topic"] is True
    assert [sq["text"] for sq in result["sub_questions"]] == ["a", "b"]


def test_synthesize_answer_single_subquestion_passthrough():
    state = {
        "is_on_topic": True,
        "question": "How do I set up my dev environment?",
        "sub_questions": [
            {
                "sub_id": "sq0",
                "text": "How do I set up my dev environment?",
                "route": "retrieval",
                "status": "resolved",
                "answer": "Run uv sync then migrate.",
                "sources": ["Dev Setup"],
            }
        ],
    }
    result = synthesize_answer(state)

    assert result["final_answer"] == "Run uv sync then migrate."
    assert result["had_sufficient_info"] is True
    assert result["sources"] == ["Dev Setup"]


def test_synthesize_answer_off_topic_short_circuits():
    result = synthesize_answer({"is_on_topic": False, "question": "tell me a joke", "sub_questions": []})

    assert result["had_sufficient_info"] is False
    assert "annual reports" in result["final_answer"].lower()


def test_synthesize_answer_no_info_found():
    state = {
        "is_on_topic": True,
        "question": "What is the capital of France?",
        "sub_questions": [
            {
                "sub_id": "sq0",
                "text": "What is the capital of France?",
                "route": "retrieval",
                "status": "resolved",
                "answer": None,
                "sources": [],
            }
        ],
    }
    result = synthesize_answer(state)

    assert result["had_sufficient_info"] is False
    assert "don't have enough information" in result["final_answer"]


def test_synthesize_answer_combines_multiple_subquestions(monkeypatch):
    monkeypatch.setattr(
        synthesize_answer_mod,
        "invoke_llm",
        lambda prompt: "Combined: run uv sync; contact Developer Experience for CI.",
    )
    state = {
        "is_on_topic": True,
        "question": "How do I set up my env, and who do I contact about CI?",
        "sub_questions": [
            {
                "sub_id": "sq0",
                "text": "How do I set up my env?",
                "route": "retrieval",
                "status": "resolved",
                "answer": "Run uv sync.",
                "sources": ["Dev Setup"],
            },
            {
                "sub_id": "sq1",
                "text": "Who do I contact about CI?",
                "route": "tool",
                "status": "resolved",
                "answer": "Contact: Tom Alvarez.",
                "sources": [],
            },
        ],
    }
    result = synthesize_answer(state)

    assert "uv sync" in result["final_answer"]
    assert result["had_sufficient_info"] is True
    assert result["sources"] == ["Dev Setup"]
