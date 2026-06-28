from __future__ import annotations

from typing import Any

from graph.prompts import SUMMARIZE_DOCUMENTS_PROMPT
from tools.llm import call_llm


def format_documents_for_prompt(documents: list[dict[str, Any]], max_chars: int = 8000) -> str:
    blocks = []
    used = 0
    for index, doc in enumerate(documents):
        source = doc.get("metadata", {}).get("source", "unknown")
        text = f"[{index}] {source}\n{doc.get('content', '')}\n"
        if used + len(text) > max_chars:
            break
        blocks.append(text)
        used += len(text)
    return "\n".join(blocks)


def summarize_documents(question: str, documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "No relevant document excerpts were found."
    prompt = SUMMARIZE_DOCUMENTS_PROMPT.format(
        question=question,
        documents=format_documents_for_prompt(documents),
    )
    return call_llm(prompt, temperature=0.0)

