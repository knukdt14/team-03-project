import pandas as pd
from langchain_upstage import ChatUpstage
from src.vector_store import build_or_load_vectorstore, get_embedding_function, get_retriever
from experiments.rag_optimization.evaluate import add_ragas_scores

INPUT = 'eval/CATIA_Solar_Gap_2_Plus_Qwen_BGE_M3_Results.csv'
OUTPUT = 'eval/CATIA_Solar_Gap_2_Plus_Qwen_BGE_M3_RAGAS_Results.csv'

df = pd.read_csv(INPUT)
vs = build_or_load_vectorstore()
contexts = [[d.page_content for d in get_retriever(vs, top_k=4).invoke(q)] for q in df['question']]
work = pd.DataFrame({'question': df['question'], 'reference_answer': df['reference_answer'], 'predicted_answer': df['rag_answer']})
work = add_ragas_scores(work, contexts, ChatUpstage(model='solar-pro2'), get_embedding_function(), ['faithfulness','context_precision','context_recall'], max_workers=1)
for col in [c for c in work.columns if c.startswith('ragas_')]: df[col] = work[col]
df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(df[[c for c in df if c.startswith('ragas_')]].mean(numeric_only=True))
