"""
Query Expansion이 실제로 검색/답변 품질에 도움이 되는지 A/B 비교.
같은 벡터스토어(재빌드 불필요, 기존 chunk_size=600/overlap=100/top_k=4 그대로)에서
검색 쿼리만 "확장 키워드 포함" vs "원본 질문 그대로"로 바꿔서 BERTScore F1을 비교한다.
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from bert_score import score
from langchain_core.output_parsers import StrOutputParser

from src.rag_chain import RAGPipeline, RAG_PROMPT, format_docs_with_pages, clean_llm_output
from src.evaluation import BENCHMARK_DATASET

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODE = "local"

OUT_CSV = os.path.join(PROJECT_ROOT, "eval", "query_expansion_ab_results.csv")


def evaluate_mode(pipeline: RAGPipeline, use_expansion: bool):
    rag_chain = RAG_PROMPT | pipeline.llm | StrOutputParser()
    answers, ground_truths, types = [], [], []

    for item in BENCHMARK_DATASET:
        q = item["question"]
        search_q = pipeline.expand_query(q) if use_expansion else q
        docs = pipeline.retriever.invoke(search_q)
        context_str = format_docs_with_pages(docs)

        raw = rag_chain.invoke({"context": context_str, "question": q})
        answers.append(clean_llm_output(raw))
        ground_truths.append(item["ground_truth"])
        types.append(item["type"])

    P, R, F1 = score(cands=answers, refs=ground_truths, lang="ko", verbose=False, batch_size=8)
    df = pd.DataFrame({
        "type": types,
        "question": [d["question"] for d in BENCHMARK_DATASET],
        "answer": answers,
        "ground_truth": ground_truths,
        "f1": F1.tolist(),
    })
    return F1.mean().item(), df


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"[QE A/B] 모델: {MODEL_NAME} ({MODE}) / 평가 문항: {len(BENCHMARK_DATASET)}개")
    pipeline = RAGPipeline(model_name=MODEL_NAME, mode=MODE)

    summary_rows = []
    frames = []
    for label, use_exp in [("Query Expansion ON", True), ("Query Expansion OFF", False)]:
        print(f"\n[QE A/B] {label} 평가 중...")
        t0 = time.time()
        f1, df = evaluate_mode(pipeline, use_exp)
        elapsed = time.time() - t0
        print(f"[QE A/B]   F1={f1:.4f} ({elapsed:.1f}s)")
        summary_rows.append({"label": label, "bertscore_f1": round(f1, 4), "elapsed_sec": round(elapsed, 1)})
        df.insert(0, "mode", label)
        frames.append(df)

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    detail_csv = OUT_CSV.replace(".csv", "_detail.csv")
    pd.concat(frames, ignore_index=True).to_csv(detail_csv, index=False, encoding="utf-8-sig")

    print("\n=== 요약 ===")
    print(df_summary.to_string(index=False))
    gap = summary_rows[0]["bertscore_f1"] - summary_rows[1]["bertscore_f1"]
    print(f"\n[QE A/B] ON - OFF F1 격차: {gap:+.4f}")
    print(f"[QE A/B] 요약 저장: {OUT_CSV}")
    print(f"[QE A/B] 문항별 상세 저장: {detail_csv}")


if __name__ == "__main__":
    main()
