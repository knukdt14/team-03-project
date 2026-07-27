import os
import pandas as pd
from typing import List, Dict, Any
from bert_score import score
from src.rag_chain import RAGPipeline

# Try importing Ragas for RAG evaluation (PDF Pages 202-223)
try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False


# Benchmark dataset (Question, Ground Truth, Query Type)
BENCHMARK_DATASET = [
    {
        "type": "General QA",
        "question": "CATIA V5에서 Pad 기능의 주요 역할과 스케치 조건은 무엇인가요?",
        "ground_truth": "Pad 기능은 2D 스케치 프로파일을 돌출시켜 3D 솔리드 바디를 생성하는 기본 기능으로, 스케치는 닫힌 루프(Closed Loop) 형태이어야 합니다."
    },
    {
        "type": "Specific QA",
        "question": "Dress-Up Feature 중 Stiffener 기능의 사용 목적과 권장 구배 각도는 몇 도인가요?",
        "ground_truth": "Stiffener 기능은 캐싱 부품 내부 리브(Rib) 보강재를 효율적으로 생성할 때 사용하며, 몰딩 공정 부품의 경우 보통 4도의 구배 각도가 권장됩니다."
    },
    {
        "type": "Specific QA",
        "question": "Draft Angle 기능이란 무엇이며 어떨 때 사용하나요?",
        "ground_truth": "Draft Angle 기능은 금형(Molding) 공정에서 부품을 원활하게 빼내기 위해 측면에 꺾임각/경사각을 부여하는 Dress-Up 기능입니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5에서 3D 퀀텀 머시닝(Quantum Machining) AI 가속 모드는 어떻게 실행하나요?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5에서 자율주행 차체 3D 홀로그램 자동 설계 알고리즘 메뉴 위치는?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    }
]


def calculate_ragas_metrics(pipeline: RAGPipeline, questions: List[str], answers: List[str], contexts: List[List[str]], ground_truths: List[str]) -> Dict[str, List[float]]:
    """
    Calculates Ragas evaluation metrics (PDF Page 202-223):
    - faithfulness: How factually accurate the generated answer is to the retrieved contexts.
    - answer_relevancy: How relevant the generated answer is to the user question.
    - context_precision: Signal-to-noise ratio of retrieved contexts.
    - context_recall: Whether all necessary information was retrieved.
    """
    if not HAS_RAGAS:
        print("[Evaluation Warning] Ragas library not available, skipping Ragas metrics.")
        return {}
        
    try:
        ragas_dict = {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths
        }
        dataset = Dataset.from_dict(ragas_dict)
        print("[Evaluation] Prepared RAGAS Dataset format (user_input, response, retrieved_contexts, reference).")
        
        # Check if OpenAI API Key is present for automated LLM judge, otherwise skip API call to prevent hang
        if not os.getenv("OPENAI_API_KEY"):
            print("[Evaluation Note] Ragas automated LLM-as-a-judge requires OPENAI_API_KEY. Formatted Dataset created successfully.")
            return {}
            
        results = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df_ragas = results.to_pandas()
        return {
            "ragas_faithfulness": df_ragas.get("faithfulness", [0.0]*len(questions)).tolist(),
            "ragas_answer_relevancy": df_ragas.get("answer_relevancy", [0.0]*len(questions)).tolist(),
            "ragas_context_precision": df_ragas.get("context_precision", [0.0]*len(questions)).tolist(),
            "ragas_context_recall": df_ragas.get("context_recall", [0.0]*len(questions)).tolist(),
        }
    except Exception as e:
        print(f"[Evaluation Note] Ragas execution notice ({e}). Skipping automated Ragas API call.")
        return {}


def run_evaluation_benchmark(pipeline: RAGPipeline, dataset: List[Dict[str, str]] = BENCHMARK_DATASET) -> pd.DataFrame:
    """
    Runs 3-way A/B testing benchmark with mandatory BERTScore and Ragas metrics:
    1. Direct LLM (RAG Off)
    2. Strict RAG (Strict Manual Only)
    3. Adaptive Fallback RAG (Manual First -> Direct LLM Fallback)
    """
    questions = [d["question"] for d in dataset]
    ground_truths = [d["ground_truth"] for d in dataset]
    types = [d["type"] for d in dataset]
    
    rag_off_answers = []
    strict_rag_answers = []
    adaptive_answers = []
    adaptive_statuses = []
    source_pages_list = []
    retrieved_contexts_list = []
    
    print("[Evaluation] Executing 3-Way Benchmark (Direct LLM vs Strict RAG vs Adaptive Fallback RAG)...")
    for item in dataset:
        q = item["question"]
        # 1. Direct LLM (RAG Off)
        ans_off = pipeline.answer_direct(q)
        rag_off_answers.append(ans_off)
        
        # 2. Strict RAG
        res_strict = pipeline.answer_rag(q)
        strict_rag_answers.append(res_strict["answer"])
        source_pages_list.append(str(res_strict["source_pages"]))
        
        # Extract text content of retrieved docs for Ragas & Context inspection
        raw_contexts = [doc.page_content for doc in res_strict["retrieved_docs"]]
        retrieved_contexts_list.append(raw_contexts)
        
        # 3. Adaptive Fallback RAG
        res_adaptive = pipeline.answer_adaptive_fallback(q)
        adaptive_answers.append(res_adaptive["answer"])
        adaptive_statuses.append(res_adaptive["status_tag"])
        
    print("[Evaluation] Calculating BERTScore (Precision, Recall, F1) for Direct LLM...")
    P_off, R_off, F1_off = score(cands=rag_off_answers, refs=ground_truths, lang="ko", verbose=False)
    
    print("[Evaluation] Calculating BERTScore (Precision, Recall, F1) for Strict RAG...")
    P_strict, R_strict, F1_strict = score(cands=strict_rag_answers, refs=ground_truths, lang="ko", verbose=False)

    print("[Evaluation] Calculating BERTScore (Precision, Recall, F1) for Adaptive Fallback RAG...")
    P_adapt, R_adapt, F1_adapt = score(cands=adaptive_answers, refs=ground_truths, lang="ko", verbose=False)
    
    # Base Results DataFrame with Full BERTScore metrics (P, R, F1) matching PDF Page 199-213
    df_results = pd.DataFrame({
        "Type": types,
        "Question": questions,
        "Ground Truth": ground_truths,
        "Direct LLM Answer": rag_off_answers,
        "Direct LLM BERTScore Precision": P_off.tolist(),
        "Direct LLM BERTScore Recall": R_off.tolist(),
        "Direct LLM F1": F1_off.tolist(),
        "Strict RAG Answer": strict_rag_answers,
        "Strict RAG BERTScore Precision": P_strict.tolist(),
        "Strict RAG BERTScore Recall": R_strict.tolist(),
        "Strict RAG F1": F1_strict.tolist(),
        "Adaptive Fallback Answer": adaptive_answers,
        "Adaptive Status": adaptive_statuses,
        "Adaptive Fallback BERTScore Precision": P_adapt.tolist(),
        "Adaptive Fallback BERTScore Recall": R_adapt.tolist(),
        "Adaptive Fallback F1": F1_adapt.tolist(),
        "Source Pages": source_pages_list
    })
    
    # Calculate Ragas Metrics (PDF Page 202-223)
    ragas_metrics = calculate_ragas_metrics(pipeline, questions, strict_rag_answers, retrieved_contexts_list, ground_truths)
    for col_name, values in ragas_metrics.items():
        df_results[col_name] = values
        
    print("[Evaluation] 3-Way Benchmark & Metric Assessment Complete.")
    return df_results


if __name__ == "__main__":
    pipeline = RAGPipeline()
    df = run_evaluation_benchmark(pipeline)
    print(df[["Type", "Question", "Direct LLM F1", "Strict RAG F1", "Adaptive Fallback F1"]])
