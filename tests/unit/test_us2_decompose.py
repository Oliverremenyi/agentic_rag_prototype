from src.agent.nodes.decompose_query import decompose_query


def test_decompose_query_dedupes_and_strips_empty():
    state = {
        "is_on_topic": True,
        "question": "irrelevant",
        "sub_questions": [
            {"sub_id": "sq0", "text": "How do I set up my env?", "route": None, "status": "pending", "answer": None, "sources": []},
            {"sub_id": "sq1", "text": "  How do I set up my env?  ", "route": None, "status": "pending", "answer": None, "sources": []},
            {"sub_id": "sq2", "text": "  ", "route": None, "status": "pending", "answer": None, "sources": []},
            {"sub_id": "sq3", "text": "Who do I contact about CI?", "route": None, "status": "pending", "answer": None, "sources": []},
        ],
    }

    result = decompose_query(state)

    texts = [sq["text"] for sq in result["sub_questions"]]
    assert texts == ["How do I set up my env?", "Who do I contact about CI?"]


def test_decompose_query_short_circuits_off_topic():
    state = {"is_on_topic": False, "question": "tell me a joke", "sub_questions": []}

    result = decompose_query(state)

    assert result["sub_questions"] == []


def test_decompose_query_falls_back_to_full_question_if_empty():
    state = {"is_on_topic": True, "question": "How do I set up my env?", "sub_questions": []}

    result = decompose_query(state)

    assert len(result["sub_questions"]) == 1
    assert result["sub_questions"][0]["text"] == "How do I set up my env?"
