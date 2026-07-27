import os
import sys
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_chain import RAGPipeline
from src.evaluation import run_evaluation_benchmark

sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("==================================================")
    print("   CATIA Manual RAG & Procedure Verification      ")
    print("       BERTScore & RAGAS Evaluation Benchmark     ")
    print("==================================================")
    
    pipeline = RAGPipeline()
    df_results = run_evaluation_benchmark(pipeline)
    
    # Save results to CSV inside data directory
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "evaluation_results.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] Detailed evaluation results saved to '{csv_path}'")
    
    print("\n--------------------------------------------------")
    print("            3-WAY BENCHMARK SUMMARY               ")
    print("--------------------------------------------------")
    
    cols = list(df_results.columns)
    direct_col = "Direct LLM F1" if "Direct LLM F1" in cols else ("RAG Off BERTScore F1" if "RAG Off BERTScore F1" in cols else cols[4])
    strict_col = "Strict RAG F1" if "Strict RAG F1" in cols else ("RAG On BERTScore F1" if "RAG On BERTScore F1" in cols else cols[6])
    adaptive_col = "Adaptive Fallback F1" if "Adaptive Fallback F1" in cols else ("Adaptive Status F1" if "Adaptive Status F1" in cols else strict_col)
    
    print(f" 1. Direct LLM (RAG Off) BERTScore F1:         {df_results[direct_col].mean():.4f}")
    print(f" 2. Strict RAG (Strict Manual Only) BERTScore F1: {df_results[strict_col].mean():.4f}")
    if adaptive_col in cols:
        print(f" 3. Adaptive Fallback RAG BERTScore F1:         {df_results[adaptive_col].mean():.4f}")
        
    if "ragas_faithfulness" in cols:
        print("\n--------------------------------------------------")
        print("            RAGAS METRICS SUMMARY                ")
        print("--------------------------------------------------")
        print(f" - Ragas Faithfulness:       {df_results['ragas_faithfulness'].mean():.4f}")
        print(f" - Ragas Answer Relevancy:   {df_results['ragas_answer_relevancy'].mean():.4f}")
        print(f" - Ragas Context Precision:  {df_results['ragas_context_precision'].mean():.4f}")
        print(f" - Ragas Context Recall:     {df_results['ragas_context_recall'].mean():.4f}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()
