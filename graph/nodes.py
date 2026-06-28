from __future__ import annotations

import json
import os
import time

from graph.prompts import ANALYZE_QUESTION_PROMPT, GENERATE_ANSWER_PROMPT
from graph.state import InsuranceRAGState
from tools.evaluator import self_evaluate_answer
from tools.grader import grade_documents
from tools.llm import call_json_llm, call_llm
from tools.retriever import retrieve_documents
from tools.rewrite import rewrite_question
from tools.summarizer import format_documents_for_prompt, summarize_documents


def analyze_question_node(state: InsuranceRAGState) -> InsuranceRAGState:
    question = state["question"]
    analysis = call_json_llm(
        ANALYZE_QUESTION_PROMPT.format(question=question),
        {"intent": "answer_question", "insurance_line": "unknown", "entities": [], "risk_flags": []},
    )
    return {
        **state,
        "original_question": state.get("original_question", question),
        "analyzed_question": analysis,
        "rewrite_count": state.get("rewrite_count", 0),
        "memory": state.get("memory", []),
        "_started_at": state.get("_started_at", time.perf_counter()),
    }


def retrieve_documents_node(state: InsuranceRAGState) -> InsuranceRAGState:
    query = state.get("rewritten_question") or state["question"]
    documents = retrieve_documents(query)
    return {**state, "documents": documents}


def grade_documents_node(state: InsuranceRAGState) -> InsuranceRAGState:
    graded = grade_documents(state["question"], state.get("documents", []))
    return {
        **state,
        "graded_documents": graded["documents"],
        "relevance_score": graded["overall_score"],
        "documents_relevant": graded["relevant"],
    }


def rewrite_question_node(state: InsuranceRAGState) -> InsuranceRAGState:
    rewritten = rewrite_question(state["question"], state.get("graded_documents", state.get("documents", [])))
    return {
        **state,
        "rewritten_question": rewritten,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def summarize_documents_node(state: InsuranceRAGState) -> InsuranceRAGState:
    docs = state.get("graded_documents") or state.get("documents", [])
    summary = summarize_documents(state["question"], docs)
    return {**state, "summary": summary}


def generate_answer_node(state: InsuranceRAGState) -> InsuranceRAGState:
    docs = state.get("graded_documents") or state.get("documents", [])
    prompt = GENERATE_ANSWER_PROMPT.format(
        question=state["question"],
        memory=json.dumps(state.get("memory", [])[-5:], indent=2),
        summary=state.get("summary", ""),
        documents=format_documents_for_prompt(docs),
    )
    answer = call_llm(prompt, temperature=0.1)
    return {**state, "answer": answer}


def self_evaluate_answer_node(state: InsuranceRAGState) -> InsuranceRAGState:
    docs = state.get("graded_documents") or state.get("documents", [])
    evaluation = self_evaluate_answer(state["question"], docs, state.get("answer", ""))
    elapsed = time.perf_counter() - float(state.get("_started_at", time.perf_counter()))
    return {**state, "evaluation": evaluation, "response_time_seconds": elapsed}


def save_memory_node(state: InsuranceRAGState) -> InsuranceRAGState:
    memory = state.get("memory", [])
    memory.append(
        {
            "question": state["question"],
            "answer": state.get("answer", ""),
            "relevance_score": str(state.get("relevance_score", 0.0)),
            "quality_score": str(state.get("evaluation", {}).get("quality_score", "")),
        }
    )
    return {**state, "memory": memory[-20:]}


def route_after_grading(state: InsuranceRAGState) -> str:
    max_rewrites = int(os.getenv("MAX_REWRITES", "1"))
    if state.get("documents_relevant") or state.get("rewrite_count", 0) >= max_rewrites:
        return "summarize_documents"
    return "rewrite_question"

