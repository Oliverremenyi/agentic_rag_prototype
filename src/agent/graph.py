"""Main LangGraph agentic workflow (constitution Principle I).

7 nodes, conditional routing, sub-question decomposition, and shared state for
intermediate results. The RAG subgraph (src/rag/subgraph.py) is invoked
indirectly via the rag_retrieve tool inside execute_retrieval and is NOT one
of these 7 nodes.

Graph shape:

    classify_query -> decompose_query -> route_subquestion --(retrieval)--> execute_retrieval -\
                                                ^  |--(tool)-------------> execute_tool ---------+--> (loop back)
                                                |  |--(direct, already resolved)-----------------/
                                                +--(none pending)--> synthesize_answer -> respond -> END
"""

from __future__ import annotations

import uuid
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agent.nodes.classify_query import classify_query
from src.agent.nodes.decompose_query import decompose_query
from src.agent.nodes.execute_retrieval import execute_retrieval
from src.agent.nodes.execute_tool import execute_tool
from src.agent.nodes.respond import respond
from src.agent.nodes.route_subquestion import route_decision, route_subquestion
from src.agent.nodes.synthesize_answer import synthesize_answer
from src.agent.state import GraphState


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_query", classify_query)
    graph.add_node("decompose_query", decompose_query)
    graph.add_node("route_subquestion", route_subquestion)
    graph.add_node("execute_retrieval", execute_retrieval)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("respond", respond)

    graph.set_entry_point("classify_query")
    graph.add_edge("classify_query", "decompose_query")
    graph.add_edge("decompose_query", "route_subquestion")
    graph.add_conditional_edges(
        "route_subquestion",
        route_decision,
        {
            "execute_retrieval": "execute_retrieval",
            "execute_tool": "execute_tool",
            "route_subquestion": "route_subquestion",
            "synthesize_answer": "synthesize_answer",
        },
    )
    graph.add_edge("execute_retrieval", "route_subquestion")
    graph.add_edge("execute_tool", "route_subquestion")
    graph.add_edge("synthesize_answer", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


_app = _build_graph()


class AgentResponse(TypedDict):
    answer: str
    sources: list[str]
    step_trace: list[dict]
    sub_questions: list[dict]
    tool_calls: list[dict]
    had_sufficient_info: bool


def run_agent(question: str) -> AgentResponse:
    """Public entry point per contracts/agent-invocation.md, used by the UI,
    the evaluation runner, and the load test runner."""
    initial_state: GraphState = {
        "query_id": str(uuid.uuid4()),
        "question": question,
        "sub_questions": [],
        "tool_calls": [],
        "step_trace": [],
    }
    final_state = _app.invoke(initial_state, config={"recursion_limit": 50})

    return {
        "answer": final_state.get("final_answer", ""),
        "sources": final_state.get("sources", []),
        "step_trace": list(final_state.get("step_trace", [])),
        "sub_questions": [
            {
                "text": sq["text"],
                "route": sq["route"],
                "answer": sq.get("answer"),
                "depends_on": sq.get("depends_on", []),
            }
            for sq in final_state.get("sub_questions", [])
        ],
        "tool_calls": list(final_state.get("tool_calls", [])),
        "had_sufficient_info": final_state.get("had_sufficient_info", False),
    }
