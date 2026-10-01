"""Streamlit prototype UI (constitution deliverable requirement).

Shows the agent's main workflow steps and the RAG process/result for each
answer (FR-009 / SC-006).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from src.agent.graph import run_agent

st.set_page_config(page_title="Annual Report Assistant", page_icon="📊")
st.title("📊 Annual Report Analysis Assistant")
st.caption(
    "Ask a direct question, a compound question, or a comparison question "
    "(e.g. \"which had higher cash flow, X or Y, and by how much?\") about the "
    "curated set of companies' annual reports."
)

question = st.text_input(
    "Your question",
    placeholder="Which had higher cash flow from operations, Microsoft or Toshiba, and by how much?",
)
ask = st.button("Ask", type="primary")


def _format_tool_value(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value) if value is not None else "_none_"


if ask and question.strip():
    with st.spinner("Thinking..."):
        response = run_agent(question.strip())

    st.subheader("Answer")
    if not response["had_sufficient_info"]:
        st.warning(response["answer"])
    else:
        st.write(response["answer"])

    if response["sources"]:
        st.caption("Sources: " + ", ".join(response["sources"]))

    with st.expander("Step trace (how this answer was produced)"):
        for step in response["step_trace"]:
            st.markdown(f"- **{step['node']}** — {step['summary']} _(at {step['timestamp']})_")

    if response["sub_questions"]:
        with st.expander("Sub-questions"):
            for sq in response["sub_questions"]:
                depends_note = f" (depends on: {', '.join(sq['depends_on'])})" if sq["depends_on"] else ""
                st.markdown(
                    f"- **{sq['text']}** (route: `{sq['route']}`){depends_note} → {sq['answer'] or '_no answer_'}"
                )

    if response["tool_calls"]:
        with st.expander("Tool calls"):
            for call in response["tool_calls"]:
                status = "✅" if call["success"] else "❌"
                st.markdown(
                    f"- {status} `{call['tool_name']}`({_format_tool_value(call['input'])}) "
                    f"→ {_format_tool_value(call['output'])}"
                )
elif ask:
    st.info("Please enter a question first.")
