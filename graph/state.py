from __future__ import annotations

from typing import Any, TypedDict


class InsuranceRAGState(TypedDict, total=False):
    question: str
    original_question: str
    analyzed_question: dict[str, Any]
    rewritten_question: str
    documents: list[dict[str, Any]]
    graded_documents: list[dict[str, Any]]
    relevance_score: float
    documents_relevant: bool
    summary: str
    answer: str
    evaluation: dict[str, Any]
    rewrite_count: int
    memory: list[dict[str, str]]
    response_time_seconds: float
    error: str
    _started_at: float
