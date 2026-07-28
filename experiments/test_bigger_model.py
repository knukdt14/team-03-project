"""
지정한 모델로 RAG Off vs RAG On(Strict RAG)을 61문항 비교한다. Query Expansion은 사용하지 않는다.

모델명을 인자로 받으므로 1.5B와 3B를 완전히 동일한 코드 경로로 비교할 수 있고,
그 결과로 "모델 크기가 커지면 RAG On/Off 격차가 어떻게 변하는지"를 확인한다.

사용법:
    python src/test_bigger_model.py "Qwen/Qwen2.5-3B-Instruct" "3b"
결과 파일: eval/rag_on_off_{라벨}_results.csv
"""
import os
import sys
import gc
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from bert_score import score
from langchain_core.output_parsers import StrOutputParser

from src.rag_chain import RAGPipeline, RAG_PROMPT, format_docs_with_pages, clean_llm_output
from src.evaluation import BENCHMARK_DATASET

## => 1.5B와 3B를 완전히 동일한 코드 경로로 비교하기 위해 모델명을 인자로 받는다.
##    사용법: python src/test_bigger_model.py [모델명] [출력라벨]
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "3b"
MODE = "local"

OUT_CSV = os.path.join(PROJECT_ROOT, "eval", f"rag_on_off_{LABEL}_results.csv")
## => 생성만 끝내고 채점 전에 먼저 저장해두는 체크포인트.
##    3B(fp16 약 6.2GB)가 GPU에 올라간 상태에서 BERTScore가 BERT 모델을 또 GPU에 올리면
##    8GB VRAM을 넘겨 드라이버째 다운되는 일이 있었음 -> 생성 결과부터 지키고 본다.
RAW_CSV = OUT_CSV.replace(".csv", "_raw.csv")


def free_gpu(*objs):
    """LLM 참조를 끊고 VRAM을 반납한다. BERTScore가 GPU를 쓰기 전에 호출."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"[{LABEL}] 모델: {MODEL_NAME} ({MODE}) / 평가 문항: {len(BENCHMARK_DATASET)}개 / Query Expansion: OFF")
    pipeline = RAGPipeline(model_name=MODEL_NAME, mode=MODE)
    rag_chain = RAG_PROMPT | pipeline.llm | StrOutputParser()

    rows = []
    t0 = time.time()
    for i, item in enumerate(BENCHMARK_DATASET, 1):
        q = item["question"]

        # RAG Off: 검색 없이 모델 사전지식만으로 답변
        direct_raw = pipeline.direct_chain.invoke({"question": q})
        direct_ans = clean_llm_output(direct_raw)

        # RAG On: 매뉴얼 검색 결과를 근거로 답변 (Query Expansion 없이 원본 질문으로만 검색)
        docs = pipeline.retriever.invoke(q)
        context_str = format_docs_with_pages(docs)
        rag_raw = rag_chain.invoke({"context": context_str, "question": q})
        rag_ans = clean_llm_output(rag_raw)

        rows.append({
            "type": item["type"],
            "question": q,
            "ground_truth": item["ground_truth"],
            "direct_answer": direct_ans,
            "rag_answer": rag_ans,
        })
        print(f"[{LABEL}] {i}/{len(BENCHMARK_DATASET)} 완료 ({time.time() - t0:.0f}s 경과)")

        ## => 매 문항마다 중간 저장 -> 도중에 죽어도 생성 결과를 잃지 않음
        os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    print(f"[{LABEL}] 생성 완료, 원본 답변 저장: {RAW_CSV}")

    ## => 채점 전에 3B를 GPU에서 내려야 BERTScore용 BERT 모델이 올라갈 자리가 생김(8GB VRAM 한계)
    print(f"[{LABEL}] LLM을 GPU에서 해제하는 중...")
    free_gpu(rag_chain, pipeline)
    rag_chain = None
    pipeline = None
    free_gpu()

    print(f"[{LABEL}] BERTScore 계산 중...")
    ## => batch_size를 줄여서 채점 단계의 순간 VRAM 사용량도 낮춤(기본값은 64)
    _, _, f1_direct = score(cands=df["direct_answer"].tolist(), refs=df["ground_truth"].tolist(),
                            lang="ko", verbose=False, batch_size=8)
    _, _, f1_rag = score(cands=df["rag_answer"].tolist(), refs=df["ground_truth"].tolist(),
                         lang="ko", verbose=False, batch_size=8)
    df["direct_f1"] = f1_direct.tolist()
    df["rag_f1"] = f1_rag.tolist()
    df["f1_gap"] = df["rag_f1"] - df["direct_f1"]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n=== {MODEL_NAME} 요약 ===")
    print(f" Direct LLM (RAG Off) F1: {df['direct_f1'].mean():.4f}")
    print(f" Strict RAG (RAG On)  F1: {df['rag_f1'].mean():.4f}")
    print(f" 격차: {df['f1_gap'].mean():+.4f}")
    print("\n--- 유형별 격차 ---")
    print(df.groupby("type")[["direct_f1", "rag_f1", "f1_gap"]].mean().to_string())
    print(f"\n[{LABEL}] 저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
