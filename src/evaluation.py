import pandas as pd
from typing import List, Dict, Any
from bert_score import score
from src.rag_chain import RAGPipeline


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


def run_evaluation_benchmark(pipeline: RAGPipeline, dataset: List[Dict[str, str]] = BENCHMARK_DATASET) -> pd.DataFrame:
    """
    Runs 3-way A/B testing benchmark:
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
        
        # 3. Adaptive Fallback RAG
        res_adaptive = pipeline.answer_adaptive_fallback(q)
        adaptive_answers.append(res_adaptive["answer"])
        adaptive_statuses.append(res_adaptive["status_tag"])
        
    print("[Evaluation] Calculating BERTScore for Direct LLM...")
    P_off, R_off, F1_off = score(cands=rag_off_answers, refs=ground_truths, lang="ko", verbose=False)
    
    print("[Evaluation] Calculating BERTScore for Strict RAG...")
    P_strict, R_strict, F1_strict = score(cands=strict_rag_answers, refs=ground_truths, lang="ko", verbose=False)

    print("[Evaluation] Calculating BERTScore for Adaptive Fallback RAG...")
    P_adapt, R_adapt, F1_adapt = score(cands=adaptive_answers, refs=ground_truths, lang="ko", verbose=False)
    
    df_results = pd.DataFrame({
        "Type": types,
        "Question": questions,
        "Ground Truth": ground_truths,
        "Direct LLM Answer": rag_off_answers,
        "Direct LLM F1": F1_off.tolist(),
        "Strict RAG Answer": strict_rag_answers,
        "Strict RAG F1": F1_strict.tolist(),
        "Adaptive Fallback Answer": adaptive_answers,
        "Adaptive Status": adaptive_statuses,
        "Adaptive Fallback F1": F1_adapt.tolist(),
        "Source Pages": source_pages_list
    })
    
    print("[Evaluation] 3-Way Benchmark complete.")
    return df_results


if __name__ == "__main__":
    pipeline = RAGPipeline()
    df = run_evaluation_benchmark(pipeline)
    print(df[["Type", "Question", "Direct LLM F1", "Strict RAG F1", "Adaptive Fallback F1"]])
