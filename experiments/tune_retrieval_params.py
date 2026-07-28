"""
검색 파라미터(top_k, 하이브리드 가중치) + 프롬프트 스타일 튜닝 실험.

목표: F1(BERTScore)을 재학습 없이 올릴 수 있는 설정을 찾는다.
- retriever 파라미터는 RAGPipeline 생성자에서 바로 바꿀 수 있어 재인덱싱 불필요.
- 속도를 위해 Strict RAG(answer_rag)만 채점하고, grounding_check/adaptive/direct는 생략.
- 61문항 전체 벤치마크 사용, 모델은 Qwen2.5-0.5B-Instruct(로컬, 빠름).
"""
import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from bert_score import score
from src.rag_chain import RAGPipeline
from src.vector_store import build_or_load_vectorstore
from src.config import CHROMA_DB_DIR_MARKDOWN_ML, MULTILINGUAL_EMBEDDING_MODEL
from src.evaluation import BENCHMARK_DATASET

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "retrieval_tuning_results.json"

questions = [d["question"] for d in BENCHMARK_DATASET]
ground_truths = [d["ground_truth"] for d in BENCHMARK_DATASET]

print("[Tune] 벡터스토어 로딩...")
vs = build_or_load_vectorstore(persist_directory=CHROMA_DB_DIR_MARKDOWN_ML, embed_model=MULTILINGUAL_EMBEDDING_MODEL)

results = []


def run_config(label, **pipeline_kwargs):
    print(f"\n[Tune] === {label} 실행 중 ===")
    t0 = time.time()
    pipeline = RAGPipeline(model_name=MODEL_NAME, mode="local", vectorstore=vs,
                            enable_grounding_check=False, **pipeline_kwargs)
    answers = []
    for q in questions:
        res = pipeline.answer_rag(q)
        answers.append(res["answer"])
    gen_time = time.time() - t0

    P, R, F1 = score(cands=answers, refs=ground_truths, lang="ko", verbose=False)
    row = {
        "label": label,
        "params": pipeline_kwargs,
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
        "gen_time_sec": gen_time,
    }
    print(f"[Tune] {label}: P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} ({gen_time:.1f}s)")
    results.append(row)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    del pipeline
    return row


# 1) top_k 스윕 (벡터 검색만, 하이브리드 없이)
for k in [2, 4, 6, 8]:
    run_config(f"top_k={k}", top_k=k, use_hybrid_search=False)

# 2) 하이브리드 검색 가중치 스윕 (top_k는 위에서 가장 좋았던 값 재사용 예정이나,
#    일단 기본 top_k=4로 고정해 비교)
for w in [0.3, 0.5, 0.7]:
    run_config(f"hybrid_w={w}", top_k=4, use_hybrid_search=True, hybrid_vector_weight=w)

print("\n[Tune] 전체 결과:")
for r in sorted(results, key=lambda x: -x["f1"]):
    print(f"  {r['label']:15s} F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

print(f"\n[Tune] 결과 저장: {RESULTS_PATH}")
