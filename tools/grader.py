from __future__ import annotations

import os
from typing import Any

from graph.prompts import GRADE_DOCUMENTS_PROMPT
from tools.llm import call_json_llm


def _format_documents(documents: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{i}] source={doc.get('metadata', {}).get('source', 'unknown')}\n{doc.get('content', '')[:1200]}"
        for i, doc in enumerate(documents)
    )


def grade_documents(question: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    if not documents:
        return {"documents": [], "overall_score": 0.0, "relevant": False}

    prompt = GRADE_DOCUMENTS_PROMPT.format(question=question, documents=_format_documents(documents))
    fallback_score = sum(float(doc.get("similarity", 0.0)) for doc in documents) / max(len(documents), 1)
    graded = call_json_llm(prompt, {"scores": [], "overall_score": fallback_score})

    score_by_index = {
        int(item.get("index", -1)): float(item.get("score", 0.0))
        for item in graded.get("scores", [])
        if isinstance(item, dict)
    }
    enriched = []
    for index, doc in enumerate(documents):
        relevance = score_by_index.get(index, float(doc.get("similarity", 0.0)))
        enriched.append({**doc, "relevance_score": relevance})

    overall = float(graded.get("overall_score", fallback_score))
    threshold = float(os.getenv("MIN_RELEVANCE_SCORE", "0.55"))
    return {"documents": enriched, "overall_score": overall, "relevant": overall >= threshold}

