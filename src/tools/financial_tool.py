"""The `compare_financial_values` tool: a deterministic, non-retrieval capability
(constitution Principle III, tool 2 of 2 — the required non-retrieval tool).

Computes over two numeric values that the agent already retrieved earlier in the
same query — it never fetches external financial data itself.
"""

from __future__ import annotations

from typing import TypedDict

TOOL_NAME = "compare_financial_values"


class ComparisonResult(TypedDict):
    greater: str  # "a" | "b" | "equal"
    difference: float
    ratio: float | None


def compare_financial_values(
    value_a: float, value_b: float, label_a: str, label_b: str
) -> ComparisonResult:
    if value_a > value_b:
        greater = "a"
    elif value_b > value_a:
        greater = "b"
    else:
        greater = "equal"

    difference = value_a - value_b
    ratio = value_a / value_b if value_b != 0 else None

    return {"greater": greater, "difference": difference, "ratio": ratio}
