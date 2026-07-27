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
def add_ragas_scores(eval_df, contexts_full, llm, embeddings):
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
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
        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        ragas_df = result.to_pandas()
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in ragas_df.columns:
                eval_df[f"ragas_{col}"] = ragas_df[col].values
    except Exception as e:
        print(f"[RAGAS 건너뜀] 실행 중 오류: {e}")
    return eval_df


### [ 6. Hallucination 수동 체크 (RAGAS faithfulness와 교차검증용 빈 컬럼) ] ###
def add_hallucination_column(eval_df):
    eval_df["hallucination_flag"] = ""   ## => 팀원이 O/X로 직접 채우는 칸
    eval_df["reviewer_note"] = ""
    return eval_df


def run_evaluation(chain, retriever, llm, embeddings, questions_csv: str,
                    output_csv: str = "eval/results.csv", use_ragas: bool = True):
    eval_df = load_eval_set(questions_csv)
    eval_df, contexts_full = run_predictions(chain, retriever, eval_df)
    eval_df = add_bertscore(eval_df)
    if use_ragas:
        eval_df = add_ragas_scores(eval_df, contexts_full, llm, embeddings)
    eval_df = add_hallucination_column(eval_df)
    eval_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    BEST_F1 = eval_df["bertscore_f1"].max()
    msg = f"BEST_F1: {BEST_F1:.4f} / 평균 응답시간: {eval_df['response_time_sec'].mean():.2f}s"
    if use_ragas and "ragas_faithfulness" in eval_df.columns:
        msg += f" / 평균 faithfulness: {eval_df['ragas_faithfulness'].mean():.4f}"
    print(msg)
    return eval_df


### [ 7. RAG 적용 vs 미적용(Direct LLM) 비교 ] ###
## => 교수님 피드백 대응: CATIA 매뉴얼이 LLM 사전학습에 이미 포함됐을 가능성이 높으므로,
##    "RAG를 걸었을 때와 안 걸었을 때 답변이 실제로 달라지는가"를 같은 질문셋으로 직접 대조한다.
##    question_type 컬럼(general/trick 등)이 있으면 유형별로도 집계한다.
def run_comparison(rag_chain, direct_chain, retriever, questions_csv: str,
                    output_csv: str = "eval/results_compare.csv"):
    eval_df = load_eval_set(questions_csv)

    rag_answers, direct_answers, contexts_preview = [], [], []
    rag_times, direct_times = [], []
    for question in eval_df["question"]:
        t0 = time.time()
        rag_answers.append(rag_chain.invoke(question))
        rag_times.append(time.time() - t0)

        t0 = time.time()
        direct_answers.append(direct_chain.invoke(question))
        direct_times.append(time.time() - t0)

        docs = retriever.invoke(question)
        contexts_preview.append(" | ".join(d.page_content[:80] for d in docs))

    eval_df["rag_answer"] = rag_answers
    eval_df["direct_llm_answer"] = direct_answers
    eval_df["retrieved_context"] = contexts_preview
    eval_df["rag_response_time_sec"] = rag_times
    eval_df["direct_response_time_sec"] = direct_times

    _, _, rag_f1 = bert_score(eval_df["rag_answer"].tolist(), eval_df["reference_answer"].tolist(), lang="ko")
    _, _, direct_f1 = bert_score(eval_df["direct_llm_answer"].tolist(), eval_df["reference_answer"].tolist(), lang="ko")
    eval_df["rag_bertscore_f1"] = rag_f1.tolist()
    eval_df["direct_bertscore_f1"] = direct_f1.tolist()
    eval_df["bertscore_f1_gap"] = eval_df["rag_bertscore_f1"] - eval_df["direct_bertscore_f1"]

    ## => 팀원이 직접 눈으로 보고 O/X로 채우는 정성 비교 칸 (RAG가 실제로 더 정확했는지, 환각을 막았는지)
    eval_df["rag_hallucination_flag"] = ""
    eval_df["direct_hallucination_flag"] = ""
    eval_df["reviewer_note"] = ""

    eval_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"RAG 평균 F1: {eval_df['rag_bertscore_f1'].mean():.4f} / Direct LLM 평균 F1: {eval_df['direct_bertscore_f1'].mean():.4f}")
    print(f"평균 F1 격차(RAG - Direct): {eval_df['bertscore_f1_gap'].mean():.4f}")
    if "question_type" in eval_df.columns:
        print("\n[질문 유형별 평균 F1 격차]")
        print(eval_df.groupby("question_type")["bertscore_f1_gap"].mean())
    return eval_df
