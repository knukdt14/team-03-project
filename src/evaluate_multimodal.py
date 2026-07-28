"""Terminal evaluation for the multimodal Chroma vector store."""
import argparse
import csv
import sys
import time
from pathlib import Path

from bert_score import score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_chain import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions_csv",
        default=str(PROJECT_ROOT / "eval" / "questions_manual_verified_40.csv"),
    )
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_csv", default=str(PROJECT_ROOT / "eval" / "multimodal_eval_results.csv"))
    args = parser.parse_args()

    with open(args.questions_csv, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Allow diagnostic CSV files whose Korean question text contains commas.
    for row in rows:
        if row.get(None):
            row["question"] = ",".join([row.get("question", ""), *row.pop(None)])
    if not rows:
        raise ValueError("Question CSV contains no rows")

    pipeline = RAGPipeline(model_name=args.model, mode="local", top_k=args.top_k)
    has_references = all(row.get("reference_answer", "").strip() for row in rows)
    direct_answers, rag_answers, references, timings = [], [], [], []
    for index, row in enumerate(rows, start=1):
        question = row["question"]
        if has_references:
            references.append(row["reference_answer"])
        direct_answers.append(pipeline.answer_direct(question))
        started = time.perf_counter()
        rag_result = pipeline.answer_rag(question)
        rag_answers.append(rag_result["answer"])
        timings.append(time.perf_counter() - started)
        row.update({
            "source_pages": " | ".join(rag_result["source_pages"]),
            "grounded": rag_result.get("grounded", True),
        })
        print(f"[{index}/{len(rows)}] completed")

    if has_references:
        _, _, direct_f1 = score(direct_answers, references, lang="en", verbose=False)
        _, _, rag_f1 = score(rag_answers, references, lang="en", verbose=False)
    for row, direct, rag, elapsed in zip(rows, direct_answers, rag_answers, timings):
        row.update({"direct_answer": direct, "rag_answer": rag, "response_time_sec": elapsed})
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("=== Multimodal evaluation summary ===")
    print(f"questions: {len(rows)} / top_k: {args.top_k}")
    if has_references:
        print(f"Direct LLM F1: {direct_f1.mean().item():.4f}")
        print(f"Multimodal RAG F1: {rag_f1.mean().item():.4f}")
    else:
        print("No reference answers: saved answers and sources for manual review.")
    print(f"RAG response time: {sum(timings) / len(timings):.2f}s")


if __name__ == "__main__":
    main()
