from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from graph.graph import insurance_graph

st.set_page_config(page_title="Insurance Agentic RAG", layout="wide")
st.title("Insurance Agentic RAG")
st.caption("Ask questions about the indexed insurance documents. Use the box below, then open Trace to see the LangGraph steps.")

if "memory" not in st.session_state:
    st.session_state.memory = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session"
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

example_questions = [
    "What documents are needed for a motor insurance claim?",
    "What is a waiting period in health insurance?",
    "What should I do after water damage at home?",
    "What does third-party motor insurance cover?",
]

st.subheader("Ask a Question")
with st.form("question_form", clear_on_submit=True):
    typed_question = st.text_input(
        "Insurance question",
        placeholder="Example: What documents are needed for a motor insurance claim?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and typed_question.strip():
    st.session_state.pending_question = typed_question.strip()

cols = st.columns(2)
for index, example in enumerate(example_questions):
    if cols[index % 2].button(example, key=f"example_{index}"):
        st.session_state.pending_question = example

st.divider()

for item in st.session_state.memory[-8:]:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])

question = st.session_state.pending_question
if question:
    st.session_state.pending_question = ""

    with st.chat_message("user"):
        st.write(question)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("Retrieving, grading, and answering..."):
            result = insurance_graph.invoke(
                {"question": question, "memory": st.session_state.memory},
                config=config,
            )
        st.write(result.get("answer", "No answer generated."))

        with st.expander("Trace"):
            st.json(
                {
                    "analysis": result.get("analyzed_question"),
                    "rewritten_question": result.get("rewritten_question"),
                    "relevance_score": result.get("relevance_score"),
                    "evaluation": result.get("evaluation"),
                    "response_time_seconds": result.get("response_time_seconds"),
                    "sources": [
                        doc.get("metadata", {})
                        for doc in result.get("graded_documents", result.get("documents", []))
                    ],
                }
            )

    st.session_state.memory = result.get("memory", st.session_state.memory)
