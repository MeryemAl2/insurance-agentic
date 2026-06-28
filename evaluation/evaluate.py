from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from graph.graph import insurance_graph


def load_questions(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_suite(name: str, questions: list[dict[str, str]]) -> list[dict[str, object]]:
    results = []
    memory = []
    for item in questions:
        started = time.perf_counter()
        result = insurance_graph.invoke(
            {"question": item["question"], "memory": memory},
            config={"configurable": {"thread_id": f"eval-{name}-{item['id']}"}},
        )
        elapsed = time.perf_counter() - started
        memory = result.get("memory", memory)
        evaluation = result.get("evaluation", {})
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result.get("answer", ""),
                "answer_quality": evaluation.get("quality_score"),
                "groundedness": evaluation.get("groundedness_score"),
                "completeness": evaluation.get("completeness_score"),
                "response_time_seconds": elapsed,
                "retrieved_document_relevance": result.get("relevance_score", 0.0),
                "source_count": len(result.get("graded_documents", [])),
            }
        )
    return results


def print_summary(name: str, results: list[dict[str, object]]) -> None:
    def avg(key: str) -> float:
        values = [float(row[key]) for row in results if row.get(key) not in (None, "")]
        return statistics.mean(values) if values else 0.0

    print(f"\n{name}")
    print(f"Questions: {len(results)}")
    print(f"Average answer quality: {avg('answer_quality'):.2f}")
    print(f"Average response time: {avg('response_time_seconds'):.2f}s")
    print(f"Average retrieved relevance: {avg('retrieved_document_relevance'):.2f}")


def main() -> None:
    output_dir = ROOT / "evaluation" / "results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}
    for name, filename in {
        "simple": "questions_simple.json",
        "complex": "questions_complex.json",
    }.items():
        questions = load_questions(ROOT / "evaluation" / filename)
        results = run_suite(name, questions)
        all_results[name] = results
        print_summary(name, results)

    output_path = output_dir / "latest_results.json"
    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    main()
