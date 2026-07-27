### [ 1. 라이브러리 임포트 ] ###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import pandas as pd
from dotenv import load_dotenv

from config import EXPERIMENT_PRESETS, PROJECT_ROOT
from main import run_pipeline
from evaluate import run_evaluation

load_dotenv()

### [ 2. 비교할 프리셋 (요구사항 항목별로 하나씩) ] ###
PRESETS_TO_COMPARE = [
    "baseline", "small_chunk", "large_chunk", "kr_sbert_embed", "chroma_store",
    "chroma_large_chunk", "chroma_top_k_4_similarity", "chroma_top_k_6_similarity",
    "chroma_top_k_4_mmr", "chroma_top_k_6_mmr", "hf_local_llm",
]

# Use this set to isolate the impact of chunking after prompt/retrieval improvements.
IMPROVEMENT_PRESETS = [
    "chroma_mmr_500_strict",
    "chroma_mmr_650_strict",
    "chroma_mmr_800_strict",
]

# Final validation compares only the two non-dominated chunk candidates.
CANDIDATE_PRESETS = [
    "chroma_mmr_500_strict",
    "chroma_mmr_800_strict",
]

# Retrieval-only comparison; it preserves the already reviewed 500-MMR result.
RETRIEVAL_PRESETS = [
    "chroma_similarity_500_strict",
    "chroma_mmr_500_fetch30_strict",
]

HYBRID_PRESETS = [
    "bm25_500_strict",
    "hybrid_rrf_500_strict",
]

RERANKER_PRESETS = [
    "chroma_rerank_500_strict",
]


CURATED_PRESETS = [
    "chroma_mmr_500_curated",
]

EMBEDDING_PRESETS = [
    "bge_m3_chroma_mmr_500_strict",
]
def manual_review_metrics(eval_df):
    """Return review coverage and hallucination rate from O/X manual labels."""
    if "hallucination_flag" not in eval_df.columns:
        return {"manual_reviewed_count": 0, "manual_hallucination_rate": None}
    labels = eval_df["hallucination_flag"].fillna("").astype(str).str.strip().str.upper()
    reviewed = labels.isin(["O", "X"])
    return {"manual_reviewed_count": int(reviewed.sum()),
            "manual_hallucination_rate": float((labels[reviewed] == "O").mean()) if reviewed.any() else None}


def summarize_existing_results(presets):
    """Create the final comparison table after manual labels have been completed."""
    summary = []
    for name in presets:
        path = os.path.join(PROJECT_ROOT, "eval", f"results_{name}.csv")
        if not os.path.exists(path):
            print(f"[skip] ?? ?? ??: {path}")
            continue
        eval_df = pd.read_csv(path)
        row = {"preset": name, "bertscore_f1": eval_df["bertscore_f1"].mean(),
               "response_time_sec": eval_df["response_time_sec"].mean()}
        if "ragas_faithfulness" in eval_df.columns:
            row["faithfulness"] = eval_df["ragas_faithfulness"].mean()
        row.update(manual_review_metrics(eval_df))
        summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(PROJECT_ROOT, "eval", "comparison_summary.csv"), index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    return summary_df


### [ 3. 순서대로 실행 + 요약표 생성 ] ###
def run_all(presets, use_ragas=True):
    summary = []
    for name in presets:
        print(f"\n=== [{name}] 실행 중 ===")
        cfg = EXPERIMENT_PRESETS[name]
        questions_csv = os.path.join(PROJECT_ROOT, "eval", "questions_template.csv")
        output_csv = os.path.join(PROJECT_ROOT, "eval", f"results_{name}.csv")

        chain, retriever, llmModel, embeddings = run_pipeline(cfg)
        eval_df = run_evaluation(
            chain, retriever, llmModel, embeddings, questions_csv, output_csv,
            use_ragas=use_ragas,
            ragas_metrics=cfg.get("ragas_metrics"),
            ragas_max_workers=cfg.get("ragas_max_workers", 1),
        )

        row = {
            "preset": name,
            "bertscore_f1": eval_df["bertscore_f1"].mean(),
            "response_time_sec": eval_df["response_time_sec"].mean(),
        }
        if "ragas_faithfulness" in eval_df.columns:
            row["faithfulness"] = eval_df["ragas_faithfulness"].mean()
        row.update(manual_review_metrics(eval_df))
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(PROJECT_ROOT, "eval", "comparison_summary.csv"),
                       index=False, encoding="utf-8-sig")
    print("\n=== 비교 요약 ===")
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_ragas", action="store_true")
    parser.add_argument("--summarize_only", action="store_true", help="manual labels are completed results only")
    parser.add_argument("--improvement_only", action="store_true", help="run only the controlled strict Chroma experiments")
    parser.add_argument("--candidate_only", action="store_true", help="run final validation for the 500/800 chunk candidates")
    parser.add_argument("--retrieval_only", action="store_true", help="run 500-chunk retrieval ablations only")
    parser.add_argument("--hybrid_only", action="store_true", help="run BM25 and hybrid-RRF experiments only")
    parser.add_argument("--reranker_only", action="store_true", help="run the multilingual reranker experiment only")
    parser.add_argument("--curated_only", action="store_true", help="run the PDF-verified curated corpus experiment only")
    parser.add_argument("--embedding_only", action="store_true", help="run the stronger multilingual embedding experiment only")
    args = parser.parse_args()
    presets = EMBEDDING_PRESETS if args.embedding_only else (CURATED_PRESETS if args.curated_only else (RERANKER_PRESETS if args.reranker_only else (HYBRID_PRESETS if args.hybrid_only else (RETRIEVAL_PRESETS if args.retrieval_only else (CANDIDATE_PRESETS if args.candidate_only else (IMPROVEMENT_PRESETS if args.improvement_only else PRESETS_TO_COMPARE))))))
    if args.summarize_only:
        summarize_existing_results(presets)
    else:
        run_all(presets, use_ragas=not args.no_ragas)