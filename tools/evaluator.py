from __future__ import annotations

from typing import Any

from graph.prompts import SELF_EVALUATE_PROMPT
from tools.llm import call_json_llm
from tools.summarizer import format_documents_for_prompt


def self_evaluate_answer(question: str, documents: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    prompt = SELF_EVALUATE_PROMPT.format(
        question=question,
        documents=format_documents_for_prompt(documents, max_chars=6000),
        answer=answer,
    )
    return call_json_llm(
        prompt,
        {
            "quality_score": 0.0,
            "groundedness_score": 0.0,
            "completeness_score": 0.0,
            "missing_info": "Evaluation model did not return valid JSON.",
            "final_verdict": "needs_review",
        },
    )

