"""Shared LangGraph state for the main agentic workflow.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Route = Literal["retrieval", "tool", "direct_answer"]
SubQuestionStatus = Literal["pending", "resolved", "failed"]


class SubQuestion(TypedDict):
    sub_id: str
    text: str
    route: Route | None
    status: SubQuestionStatus
    depends_on: list[str]
    answer: str | None
    sources: list[str]


class ToolCall(TypedDict):
    tool_name: str
    input: str | dict
    output: str | dict | None
    success: bool
    sub_id: str


class StepTraceEntry(TypedDict):
    node: str
    summary: str
    timestamp: str


class GraphState(TypedDict, total=False):
    query_id: str
    question: str
    is_on_topic: bool
    sub_questions: list[SubQuestion]
    tool_calls: Annotated[list[ToolCall], operator.add]
    step_trace: Annotated[list[StepTraceEntry], operator.add]
    final_answer: str
    had_sufficient_info: bool
    sources: list[str]
