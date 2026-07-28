"""
Qwen2.5-1.5B 기준으로 chunk_size / overlap / top_k 조합을 그리드서치해서
Strict RAG BERTScore F1이 가장 높은 조합을 찾는다.

팀원이 찾은 "최적 파라미터"는 다른 모델 기준이라 1.5B에는 그대로 전이되지 않을 수 있어서,
1.5B 자체로 새로 탐색한다. chunk_size/overlap이 바뀔 때만 벡터스토어를 재빌드하고,
top_k는 같은 벡터스토어에서 retriever만 바꿔가며 평가해 재빌드 횟수를 최소화한다.

결과는 매 chunk_size 완료 시마다 CSV에 이어서 저장되고(중간에 죽어도 진행 상황 보존),
이미 CSV에 있는 chunk_size는 재실행 시 건너뛴다(이어서 실행 가능).
각 chunk_size는 고유한 임시 디렉터리를 쓰고 실행 중엔 삭제하지 않아서,
Windows에서 이전 Chroma sqlite 파일이 아직 잠겨있어 rmtree가 실패하는 문제를 피한다.
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

import pandas as pd
from bert_score import score
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

from src.vector_store import get_embedding_function, get_retriever
from src.load_pdf import load_and_split_multimodal_pdf
from src.models.llm_factory import LLMFactory
from src.rag_chain import RAG_PROMPT, format_docs_with_pages, clean_llm_output
from src.evaluation import BENCHMARK_DATASET

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODE = "local"

## => 필요에 따라 값 조정 가능. chunk_size가 바뀌면 벡터스토어 재빌드가 필요해서
##    비용이 크므로 3개로 제한, top_k는 재빌드 없이 가벼우니 넉넉하게 3개.
CHUNK_SIZE_GRID = [400, 600, 800]
OVERLAP_RATIO = 1 / 6  # overlap = chunk_size * ratio (기존 600/100 비율과 동일)
TOP_K_GRID = [2, 4, 6]

TUNE_ROOT = os.path.join(PROJECT_ROOT, "vect", "_tune_tmp")
OUT_CSV = os.path.join(PROJECT_ROOT, "eval", "param_tuning_results.csv")


def build_vectorstore_for(chunk_size: int, overlap: int) -> Chroma:
    tune_dir = os.path.join(TUNE_ROOT, f"cs{chunk_size}_ov{overlap}")
    if os.path.exists(tune_dir):
        shutil.rmtree(tune_dir, ignore_errors=True)
    embeddings = get_embedding_function()
    chunks = load_and_split_multimodal_pdf(chunk_size=chunk_size, chunk_overlap=overlap)
    return Chroma.from_documents(chunks, embeddings, persist_directory=tune_dir)


def evaluate_combo(vectorstore: Chroma, llm, top_k: int) -> float:
    retriever = get_retriever(vectorstore, top_k=top_k)
    rag_chain = (
        {
            "context": lambda x: format_docs_with_pages(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answers, ground_truths = [], []
    for item in BENCHMARK_DATASET:
        raw = rag_chain.invoke({"question": item["question"]})
        answers.append(clean_llm_output(raw))
        ground_truths.append(item["ground_truth"])

    _, _, f1 = score(cands=answers, refs=ground_truths, lang="ko", verbose=False, batch_size=8)
    return f1.mean().item()


def load_existing_results() -> pd.DataFrame:
    if os.path.exists(OUT_CSV):
        return pd.read_csv(OUT_CSV)
    return pd.DataFrame(columns=["chunk_size", "overlap", "top_k", "bertscore_f1", "elapsed_sec"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"[Tune] 모델: {MODEL_NAME} ({MODE}) / 평가 문항: {len(BENCHMARK_DATASET)}개")

    df_existing = load_existing_results()
    done_chunk_sizes = set(df_existing["chunk_size"].unique().tolist()) if not df_existing.empty else set()
    if done_chunk_sizes:
        print(f"[Tune] 기존 결과 발견, 이미 완료된 chunk_size={sorted(done_chunk_sizes)}는 건너뜀")

    remaining = [cs for cs in CHUNK_SIZE_GRID if cs not in done_chunk_sizes]
    if not remaining:
        print("[Tune] 모든 chunk_size가 이미 완료되어 있습니다.")
        df_all = df_existing
    else:
        llm = LLMFactory.get_llm(model_name=MODEL_NAME, mode=MODE)
        all_results = df_existing.to_dict("records")

        for chunk_size in remaining:
            overlap = int(chunk_size * OVERLAP_RATIO)
            print(f"\n[Tune] chunk_size={chunk_size} overlap={overlap} -> 벡터스토어 빌드 중...")
            t0 = time.time()
            vs = build_vectorstore_for(chunk_size, overlap)
            print(f"[Tune]   빌드 완료 ({time.time() - t0:.1f}s)")

            for top_k in TOP_K_GRID:
                t0 = time.time()
                f1 = evaluate_combo(vs, llm, top_k)
                elapsed = time.time() - t0
                print(f"[Tune]   top_k={top_k} -> F1={f1:.4f} ({elapsed:.1f}s)")
                all_results.append({
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "top_k": top_k,
                    "bertscore_f1": round(f1, 4),
                    "elapsed_sec": round(elapsed, 1),
                })

            ## => chunk_size 하나 끝날 때마다 바로 저장 -> 중간에 죽어도 진행 상황 보존
            df_all = pd.DataFrame(all_results).sort_values("bertscore_f1", ascending=False)
            os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
            df_all.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
            print(f"[Tune]   중간 저장 완료: {OUT_CSV}")

            ## => 다음 chunk_size로 넘어가기 전에 참조를 끊어서 Windows 파일 잠금 해제 유도
            del vs
            gc.collect()

    shutil.rmtree(TUNE_ROOT, ignore_errors=True)

    print("\n=== 결과 (F1 높은 순) ===")
    print(df_all.to_string(index=False))
    best = df_all.iloc[0]
    print(f"\n[Tune] 최적 조합: chunk_size={int(best['chunk_size'])}, overlap={int(best['overlap'])}, top_k={int(best['top_k'])} (F1={best['bertscore_f1']:.4f})")
    print(f"[Tune] 결과 저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
