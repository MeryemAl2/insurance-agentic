from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    analyze_question_node,
    generate_answer_node,
    grade_documents_node,
    retrieve_documents_node,
    rewrite_question_node,
    route_after_grading,
    save_memory_node,
    self_evaluate_answer_node,
    summarize_documents_node,
)
from graph.state import InsuranceRAGState


def build_graph():
    graph = StateGraph(InsuranceRAGState)

    graph.add_node("analyze_question", analyze_question_node)
    graph.add_node("retrieve_documents", retrieve_documents_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("summarize_documents", summarize_documents_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("self_evaluate_answer", self_evaluate_answer_node)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "summarize_documents": "summarize_documents",
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_documents")
    graph.add_edge("summarize_documents", "generate_answer")
    graph.add_edge("generate_answer", "self_evaluate_answer")
    graph.add_edge("self_evaluate_answer", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=MemorySaver())


insurance_graph = build_graph()

