from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "mock":
        return _call_mock(prompt)
    if provider == "gemini":
        return _call_gemini(prompt, temperature)
    if provider in {"grok", "xai"}:
        return _call_xai(prompt, temperature)
    if provider == "ollama":
        return _call_ollama(prompt, temperature)
    return _call_openai(prompt, temperature)


def call_json_llm(prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("LLM_PROVIDER", "openai").lower() == "mock":
        return _call_mock_json(prompt, fallback)

    raw = call_llm(prompt, temperature=0.0)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass
    return fallback


def _call_openai(prompt: str, temperature: float) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are precise, cautious, and helpful."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _call_gemini(prompt: str, temperature: float) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")

    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = requests.post(
        url,
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def _call_xai(prompt: str, temperature: float) -> str:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY is required when LLM_PROVIDER=xai or grok.")

    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
    model = os.getenv("XAI_MODEL", "grok-4.3")
    response = requests.post(
        f"{base_url}/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": temperature,
            "input": [
                {"role": "system", "content": "You are precise, cautious, and helpful."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("output_text"):
        return data["output_text"]
    output = data.get("output", [])
    chunks = []
    for item in output:
        for content in item.get("content", []):
            chunks.append(content.get("text", ""))
    return "".join(chunks)


def _call_ollama(prompt: str, temperature: float) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _call_mock(prompt: str) -> str:
    if "Rewrite the insurance question" in prompt:
        return _extract_section(prompt, "Original question:") or "insurance claim coverage requirements"

    if "Summarize the relevant insurance document excerpts" in prompt:
        excerpts = _extract_section(prompt, "Relevant excerpts:")
        sentences = _first_sentences(excerpts, limit=5)
        return " ".join(sentences) or "No relevant document excerpts were found."

    if "You are an insurance assistant." in prompt:
        question = _extract_between(prompt, "Question:", "Memory:").strip()
        summary = _extract_between(prompt, "Document summary:", "Relevant excerpts:").strip()
        excerpts = _extract_section(prompt, "Relevant excerpts:").strip()
        evidence = summary or " ".join(_first_sentences(excerpts, limit=4))
        if not evidence:
            evidence = "I could not find enough supporting text in the indexed documents."
        return (
            f"Test answer for: {question}\n\n"
            f"Based on the retrieved sample documents: {evidence}\n\n"
            "For a real decision, check the exact policy wording, exclusions, limits, and claim conditions."
        )

    return "Mock LLM response. Configure OpenAI or another provider for production-quality answers."


def _call_mock_json(prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if "Extract the user's intent" in prompt:
        question = _extract_section(prompt, "Question:")
        return {
            "intent": "answer_insurance_question",
            "insurance_line": _guess_line(question),
            "entities": [],
            "risk_flags": ["mock_llm"],
        }

    if "Score each document" in prompt:
        doc_count = len(re.findall(r"\[\d+\]", prompt))
        scores = [{"index": index, "score": 0.75, "reason": "Mock relevance score for testing."} for index in range(doc_count)]
        return {"scores": scores, "overall_score": 0.75}

    if "Evaluate the answer" in prompt:
        return {
            "quality_score": 0.7,
            "groundedness_score": 0.7,
            "completeness_score": 0.6,
            "missing_info": "Mock evaluation only. Use a real LLM for serious evaluation.",
            "final_verdict": "acceptable_for_smoke_test",
        }

    return fallback


def _extract_section(text: str, marker: str) -> str:
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].strip()


def _extract_between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    value = text.split(start, 1)[1]
    if end in value:
        value = value.split(end, 1)[0]
    return value


def _first_sentences(text: str, limit: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence for sentence in sentences if sentence][:limit]


def _guess_line(question: str) -> str:
    lowered = question.lower()
    if "motor" in lowered or "car" in lowered or "vehicle" in lowered:
        return "motor"
    if "health" in lowered or "hospital" in lowered or "medical" in lowered:
        return "health"
    if "home" in lowered or "house" in lowered or "water damage" in lowered:
        return "home"
    if "claim" in lowered:
        return "claims"
    return "unknown"
