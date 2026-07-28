"""
지금까지 각 축에서 최적으로 나온 값을 "한꺼번에" 넣은 조합을 실제로 측정한다.

지금까지 실험은 축을 하나씩만 바꿨기 때문에, 최적값을 모두 결합한 조합은 측정된 적이 없다.
여기서 3B + chunk_size=400/overlap=66 + top_k=4 + Query Expansion ON 을 돌려서,
현재까지 최고 기록(3B / 600·100 / top_k=4 / QE off = F1 0.7648)을 실제로 넘는지 확인한다.

비교를 위해 같은 벡터스토어에서 QE off 조건도 함께 측정한다.
"""
import os
import sys
import gc
import time
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from bert_score import score
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

from src.vector_store import get_embedding_function, get_retriever
from src.load_pdf import load_and_split_multimodal_pdf
from src.models.llm_factory import LLMFactory
from src.rag_chain import (
    EXPAND_QUERY_PROMPT, RAG_PROMPT, DIRECT_LLM_PROMPT,
    format_docs_with_pages, clean_llm_output, _sanitize_expansion,
)
from src.evaluation import BENCHMARK_DATASET

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODE = "local"

## => 그리드서치 1위 조합(1.5B 기준)을 3B에 적용해 본다
CHUNK_SIZE = 400
OVERLAP = 66
TOP_K = 4

TUNE_DIR = os.path.join(PROJECT_ROOT, "vect", "_bestcombo_tmp")
OUT_CSV = os.path.join(PROJECT_ROOT, "eval", "best_combo_results.csv")
RAW_CSV = OUT_CSV.replace(".csv", "_raw.csv")


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"[Best] 모델={MODEL_NAME} chunk={CHUNK_SIZE}/{OVERLAP} top_k={TOP_K} / 문항 {len(BENCHMARK_DATASET)}개")

    print(f"[Best] 벡터스토어 빌드 중 (chunk_size={CHUNK_SIZE}, overlap={OVERLAP})...")
    t0 = time.time()
    if os.path.exists(TUNE_DIR):
        shutil.rmtree(TUNE_DIR, ignore_errors=True)
    embeddings = get_embedding_function()
    chunks = load_and_split_multimodal_pdf(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=TUNE_DIR)
    print(f"[Best]   빌드 완료: {vs._collection.count()} 청크 ({time.time() - t0:.1f}s)")

    retriever = get_retriever(vs, top_k=TOP_K)
    ## => 검색이 실제로 top_k개를 반환하는지 스모크 테스트(벡터스토어 손상 조기 발견용)
    probe = retriever.invoke(BENCHMARK_DATASET[0]["question"])
    print(f"[Best]   스모크 테스트: top_k={TOP_K} -> {len(probe)}개 반환")
    if len(probe) != TOP_K:
        print("[Best] !! 검색이 top_k와 다른 개수를 반환합니다. 벡터스토어 손상 의심 -> 중단")
        return

    llm = LLMFactory.get_llm(model_name=MODEL_NAME, mode=MODE)
    expand_chain = EXPAND_QUERY_PROMPT | llm | StrOutputParser()
    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    direct_chain = DIRECT_LLM_PROMPT | llm | StrOutputParser()

    def expand(q: str) -> str:
        try:
            e = _sanitize_expansion(clean_llm_output(expand_chain.invoke({"question": q})))
            return f"{q} {e}" if e else q
        except Exception:
            return q

    rows = []
    t0 = time.time()
    for i, item in enumerate(BENCHMARK_DATASET, 1):
        q = item["question"]

        direct_ans = clean_llm_output(direct_chain.invoke({"question": q}))

        # Query Expansion 미사용: 원본 질문으로만 검색
        ctx_off = format_docs_with_pages(retriever.invoke(q))
        ans_off = clean_llm_output(rag_chain.invoke({"context": ctx_off, "question": q}))

        # Query Expansion 사용: 확장 키워드를 붙여 검색
        ctx_on = format_docs_with_pages(retriever.invoke(expand(q)))
        ans_on = clean_llm_output(rag_chain.invoke({"context": ctx_on, "question": q}))

        rows.append({
            "type": item["type"], "question": q, "ground_truth": item["ground_truth"],
            "direct_answer": direct_ans, "rag_qe_off_answer": ans_off, "rag_qe_on_answer": ans_on,
        })
        print(f"[Best] {i}/{len(BENCHMARK_DATASET)} 완료 ({time.time() - t0:.0f}s)")
        os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)

    print("[Best] LLM을 GPU에서 해제하는 중...")
    del expand_chain, rag_chain, direct_chain, llm, retriever, vs
    free_gpu()

    print("[Best] BERTScore 계산 중...")
    gts = df["ground_truth"].tolist()
    for col, name in [("direct_answer", "direct"), ("rag_qe_off_answer", "rag_qe_off"), ("rag_qe_on_answer", "rag_qe_on")]:
        _, _, f1 = score(cands=df[col].tolist(), refs=gts, lang="ko", verbose=False, batch_size=8)
        df[f"{name}_f1"] = f1.tolist()

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    shutil.rmtree(TUNE_DIR, ignore_errors=True)

    print(f"\n=== 최적 조합 결과 ({MODEL_NAME}, chunk={CHUNK_SIZE}/{OVERLAP}, top_k={TOP_K}) ===")
    print(f" RAG Off (Direct LLM)     : {df['direct_f1'].mean():.4f}")
    print(f" RAG On,  QE OFF          : {df['rag_qe_off_f1'].mean():.4f}")
    print(f" RAG On,  QE ON           : {df['rag_qe_on_f1'].mean():.4f}")
    print(f"\n 기존 최고기록(3B/600·100/QE off) : 0.7648")
    best = max(df['rag_qe_off_f1'].mean(), df['rag_qe_on_f1'].mean())
    print(f" 이번 최고                        : {best:.4f}  ({best - 0.7648:+.4f})")
    print("\n--- 유형별 ---")
    print(df.groupby("type")[["direct_f1", "rag_qe_off_f1", "rag_qe_on_f1"]].mean().round(4).to_string())
    print(f"\n[Best] 저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
