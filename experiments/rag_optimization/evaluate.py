### [ 1. 라이브러리 임포트 ] ###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import pandas as pd
from bert_score import score as bert_score


### [ 2. 평가 데이터 로드 ] ###
def load_eval_set(csv_path: str):
    ## => 컬럼: question, reference_answer, reference_sentence
    return pd.read_csv(csv_path)


### [ 3. 답변 생성 + 응답시간 측정 ] ###
def run_predictions(chain, retriever, eval_df):
    ## => contexts_full은 RAGAS 입력용(list[str] per question), retrieved_context는 사람이 보기용 요약
    predictions, contexts_full, contexts_preview, response_times = [], [], [], []
    for question in eval_df["question"]:
        t0 = time.time()
        answer = chain.invoke(question)
        response_times.append(time.time() - t0)

        docs = retriever.invoke(question)
        full_texts = [d.page_content for d in docs]
        contexts_full.append(full_texts)
        contexts_preview.append(" | ".join(t[:80] for t in full_texts))
        predictions.append(answer)

    eval_df["predicted_answer"] = predictions
    eval_df["retrieved_context"] = contexts_preview
    eval_df["response_time_sec"] = response_times
    return eval_df, contexts_full


### [ 4. BERTScore (필수 평가 지표) ] ###
def add_bertscore(eval_df):
    P, R, F1 = bert_score(
        eval_df["predicted_answer"].tolist(),
        eval_df["reference_answer"].tolist(),
        lang="ko",
    )
    eval_df["bertscore_precision"] = P.tolist()
    eval_df["bertscore_recall"] = R.tolist()
    eval_df["bertscore_f1"] = F1.tolist()
    return eval_df


### [ 5. RAGAS (lab_06과 동일. faithfulness = hallucination을 수치로 확인) ] ###
## => 주의: ragas가 langchain-community 최신 버전과 임포트 충돌을 일으키는 경우가 있음(2026.07 기준 확인됨).
##    실패해도 전체 파이프라인이 죽지 않도록 실패 시 조용히 건너뜀 (BERTScore + 수동 체크는 계속 진행)
def add_ragas_scores(eval_df, contexts_full, llm, embeddings, metric_names=None, max_workers=1):
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall
        from ragas.run_config import RunConfig
    except Exception as e:
        print(f"[RAGAS 건너뜀] import 실패: {e}")
        print("  -> 새 가상환경에서 ragas만 먼저 pip install 해보거나, --no_ragas 옵션으로 계속 진행하세요.")
        return eval_df

    try:
        ragas_data = {
            "question": eval_df["question"].tolist(),
            "answer": eval_df["predicted_answer"].tolist(),
            "contexts": contexts_full,
            "ground_truth": eval_df["reference_answer"].tolist(),
        }
        dataset = Dataset.from_dict(ragas_data)
        metric_map = {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        selected_names = metric_names or list(metric_map)
        selected_metrics = [metric_map[name] for name in selected_names]
        result = ragas_evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=max_workers, max_retries=3, max_wait=30),
        )
        ragas_df = result.to_pandas()
        for col in selected_names:
            if col in ragas_df.columns:
                eval_df[f"ragas_{col}"] = ragas_df[col].values
    except Exception as e:
        print(f"[RAGAS 건너뜀] 실행 중 오류: {e}")
    return eval_df


### [ 6. Hallucination 수동 체크 (RAGAS faithfulness와 교차검증용 빈 컬럼) ] ###
def add_hallucination_column(eval_df):
    """Add manual-review fields without erasing labels supplied in an input CSV.

    Use O for a hallucinated/unsupported answer and X for a grounded answer.
    ``reviewer_note`` should record the unsupported claim or supporting evidence.
    """
    if "hallucination_flag" not in eval_df.columns:
        eval_df["hallucination_flag"] = ""
    if "reviewer_note" not in eval_df.columns:
        eval_df["reviewer_note"] = ""
    return eval_df


def run_evaluation(chain, retriever, llm, embeddings, questions_csv: str,
                    output_csv: str = "eval/results.csv", use_ragas: bool = True,
                    ragas_metrics=None, ragas_max_workers=1):
    eval_df = load_eval_set(questions_csv)
    eval_df, contexts_full = run_predictions(chain, retriever, eval_df)
    eval_df = add_bertscore(eval_df)
    if use_ragas:
        eval_df = add_ragas_scores(eval_df, contexts_full, llm, embeddings, ragas_metrics, ragas_max_workers)
    eval_df = add_hallucination_column(eval_df)
    eval_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    BEST_F1 = eval_df["bertscore_f1"].max()
    msg = f"BEST_F1: {BEST_F1:.4f} / 평균 응답시간: {eval_df['response_time_sec'].mean():.2f}s"
    if use_ragas and "ragas_faithfulness" in eval_df.columns:
        msg += f" / 평균 faithfulness: {eval_df['ragas_faithfulness'].mean():.4f}"
    print(msg)
    return eval_df
