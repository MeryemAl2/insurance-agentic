from __future__ import annotations

from typing import Any

from graph.prompts import REWRITE_QUESTION_PROMPT
from tools.llm import call_llm


def _format_weak_documents(documents: list[dict[str, Any]]) -> str:
    return "\n\n".join(doc.get("content", "")[:600] for doc in documents[:3]) or "No useful documents retrieved."


def rewrite_question(question: str, documents: list[dict[str, Any]]) -> str:
    prompt = REWRITE_QUESTION_PROMPT.format(question=question, documents=_format_weak_documents(documents))
    rewritten = call_llm(prompt, temperature=0.0).strip()
    return rewritten.strip('"') or question

