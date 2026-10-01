"""Node 7: respond.

Final node in the main graph. The heavy lifting (building the AgentResponse)
happens in graph.run_agent from the final GraphState; this node just records
completion in the step trace.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.agent.state import GraphState


def respond(state: GraphState) -> GraphState:
    return {
        "step_trace": [
            {
                "node": "respond",
                "summary": "Response ready",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    }
