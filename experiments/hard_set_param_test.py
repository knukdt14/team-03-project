"""
"어려운 버전" 평가셋(src/hard_benchmark.py)으로 파라미터가 실제로 차이를 만드는지 검증.

기존 쉬운 세트에서는 top_k / Query Expansion을 바꿔도 F1이 소수점 4자리까지 동일했다
(질문-문단 어휘가 겹쳐서 검색이 항상 같은 문서를 반환). 어휘 간극을 만든 이 세트에서도
같은지 확인해서, "튜닝 효과 없음"이 파라미터 탓인지 평가셋 설계 탓인지 가린다.

벡터스토어 재빌드 없이 top_k와 Query Expansion만 바꿔가며 비교한다.
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
from src.hard_benchmark import HARD_BENCHMARK_DATASET

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODE = "local"

## => 쉬운 세트에서 아무 차이 없던 두 축을 그대로 재현 테스트
CONFIGS = [
    {"label": "top_k=2, QE off", "top_k": 2, "use_expansion": False},
    {"label": "top_k=4, QE off", "top_k": 4, "use_expansion": False},
    {"label": "top_k=6, QE off", "top_k": 6, "use_expansion": False},
    {"label": "top_k=4, QE on", "top_k": 4, "use_expansion": True},
]

OUT_CSV = os.path.join(PROJECT_ROOT, "eval", "hard_set_param_results.csv")


def evaluate(pipeline: RAGPipeline, top_k: int, use_expansion: bool):
    pipeline.retriever.search_kwargs["k"] = top_k
    rag_chain = RAG_PROMPT | pipeline.llm | StrOutputParser()

    answers, ground_truths, retrieved_top1 = [], [], []
    for item in HARD_BENCHMARK_DATASET:
        q = item["question"]
        search_q = pipeline.expand_query(q) if use_expansion else q
        docs = pipeline.retriever.invoke(search_q)
        context_str = format_docs_with_pages(docs)
        raw = rag_chain.invoke({"context": context_str, "question": q})
        answers.append(clean_llm_output(raw))
        ground_truths.append(item["ground_truth"])
        ## => 검색 결과가 실제로 달라지는지 보려고 top-1 문서 식별자도 기록
        if docs:
            m = docs[0].metadata
            retrieved_top1.append(f"{m.get('source_file','?')}#p{m.get('page','?')}")
        else:
            retrieved_top1.append("(none)")

    ## => LLM이 GPU에 올라간 채로 BERTScore가 돌기 때문에 batch_size를 낮춰 VRAM 여유 확보
    ##    (8GB 환경에서 3B + BERTScore 동시 로드 시 드라이버가 다운된 사례 있음)
    _, _, F1 = score(cands=answers, refs=ground_truths, lang="ko", verbose=False, batch_size=8)
    df = pd.DataFrame({
        "question": [d["question"] for d in HARD_BENCHMARK_DATASET],
        "answer": answers,
        "ground_truth": ground_truths,
        "top1_doc": retrieved_top1,
        "f1": F1.tolist(),
    })
    return F1.mean().item(), df


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"[Hard Set] 모델: {MODEL_NAME} / 어려운 문항: {len(HARD_BENCHMARK_DATASET)}개")
    pipeline = RAGPipeline(model_name=MODEL_NAME, mode=MODE)

    summary, frames = [], []
    for cfg in CONFIGS:
        print(f"\n[Hard Set] {cfg['label']} 평가 중...")
        t0 = time.time()
        f1, df = evaluate(pipeline, cfg["top_k"], cfg["use_expansion"])
        elapsed = time.time() - t0
        print(f"[Hard Set]   F1={f1:.4f} ({elapsed:.1f}s)")
        summary.append({
            "label": cfg["label"],
            "top_k": cfg["top_k"],
            "query_expansion": cfg["use_expansion"],
            "bertscore_f1": round(f1, 4),
            "elapsed_sec": round(elapsed, 1),
        })
        df.insert(0, "config", cfg["label"])
        frames.append(df)

    df_summary = pd.DataFrame(summary)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    detail_csv = OUT_CSV.replace(".csv", "_detail.csv")
    df_detail = pd.concat(frames, ignore_index=True)
    df_detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    print("\n=== 요약 ===")
    print(df_summary.to_string(index=False))
    spread = df_summary["bertscore_f1"].max() - df_summary["bertscore_f1"].min()
    print(f"\n[Hard Set] F1 최대-최소 편차: {spread:.4f}")

    ## => 설정별로 top-1 검색 문서가 실제로 달라졌는지(=파라미터가 검색에 영향을 줬는지) 확인
    pivot = df_detail.pivot_table(index="question", columns="config", values="top1_doc", aggfunc="first")
    differing = (pivot.nunique(axis=1) > 1).sum()
    print(f"[Hard Set] 설정에 따라 top-1 문서가 달라진 문항 수: {differing} / {len(pivot)}")
    print(f"[Hard Set] 저장: {OUT_CSV}, {detail_csv}")


if __name__ == "__main__":
    main()
