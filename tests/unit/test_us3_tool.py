from src.tools.financial_tool import compare_financial_values
from src.agent.nodes.execute_tool import _parse_number, execute_tool


def _fact(sub_id: str, text: str, answer: str | None, sources: list[str] | None = None):
    return {
        "sub_id": sub_id,
        "text": text,
        "route": "retrieval",
        "status": "resolved",
        "depends_on": [],
        "answer": answer,
        "sources": sources or [],
    }


def _comparison(sub_id: str, text: str, depends_on: list[str]):
    return {
        "sub_id": sub_id,
        "text": text,
        "route": "tool",
        "status": "pending",
        "depends_on": depends_on,
        "answer": None,
        "sources": [],
    }


def test_parse_number_ignores_a_leading_year_and_finds_the_dollar_figure():
    # Regression: the first number in this sentence is the year "2021", not
    # the $13.1 billion figure - a real bug found running against a real LLM.
    text = "MGM Resorts International's consolidated net revenues in 2021 were $13.1 billion."

    assert _parse_number(text) == 13_100_000_000.0


def test_parse_number_returns_none_for_a_bare_year_with_no_other_number():
    assert _parse_number("The report covers fiscal year 2022.") is None


def test_parse_number_handles_plain_currency_with_commas():
    assert _parse_number("Total net revenues were $2,207,885.") == 2_207_885.0


def test_compare_financial_values_reports_greater_and_ratio():
    result = compare_financial_values(100.0, 25.0, "A", "B")

    assert result["greater"] == "a"
    assert result["difference"] == 75.0
    assert result["ratio"] == 4.0


def test_compare_financial_values_handles_equal():
    result = compare_financial_values(10.0, 10.0, "A", "B")

    assert result["greater"] == "equal"
    assert result["difference"] == 0.0


def test_execute_tool_resolves_comparison_with_numeric_dependencies():
    state = {
        "sub_questions": [
            _fact("sq0", "What was Company A's revenue?", "Company A's revenue was $50 million.", ["Company A — Financials"]),
            _fact("sq1", "What was Company B's revenue?", "Company B's revenue was $20 million.", ["Company B — Financials"]),
            _comparison("sq2", "Compare: revenue", ["sq0", "sq1"]),
        ]
    }

    result = execute_tool(state)

    comparison = next(sq for sq in result["sub_questions"] if sq["sub_id"] == "sq2")
    assert comparison["status"] == "resolved"
    assert "50,000,000.00" in comparison["answer"]
    assert "20,000,000.00" in comparison["answer"]
    assert set(comparison["sources"]) == {"Company A — Financials", "Company B — Financials"}

    call = result["tool_calls"][0]
    assert call["tool_name"] == "compare_financial_values"
    assert call["success"] is True
    assert call["input"]["value_a"] == 50_000_000.0
    assert call["input"]["value_b"] == 20_000_000.0


def test_execute_tool_fails_gracefully_when_a_value_is_missing():
    state = {
        "sub_questions": [
            _fact("sq0", "What was Company A's revenue?", "Company A's revenue was $50 million."),
            _fact("sq1", "What was Company B's revenue?", None),
            _comparison("sq2", "Compare: revenue", ["sq0", "sq1"]),
        ]
    }

    result = execute_tool(state)

    comparison = next(sq for sq in result["sub_questions"] if sq["sub_id"] == "sq2")
    assert comparison["status"] == "failed"
    assert comparison["answer"] is None
    assert result["tool_calls"][0]["success"] is False


def test_execute_tool_fails_gracefully_when_answer_is_non_numeric():
    state = {
        "sub_questions": [
            _fact("sq0", "What was Company A's revenue?", "Not disclosed in the report."),
            _fact("sq1", "What was Company B's revenue?", "Company B's revenue was $20 million."),
            _comparison("sq2", "Compare: revenue", ["sq0", "sq1"]),
        ]
    }

    result = execute_tool(state)

    comparison = next(sq for sq in result["sub_questions"] if sq["sub_id"] == "sq2")
    assert comparison["status"] == "failed"
    assert result["tool_calls"][0]["success"] is False
