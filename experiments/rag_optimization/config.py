### [ 1. 실험 기본 설정 ] ###
## => 이 값들만 바꾸면 다른 조합으로 재실험 가능
## => 요구사항: LLM / Embedding / Prompt / 벡터스토어 / chunk_size / overlap_size / top_k 비교

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# This module lives under experiments/rag_optimization/ in the team repository.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_CONFIG = {
    "pdf_path": os.path.join(PROJECT_ROOT, "data"),
    "corpus_tag": "base",
    "chunk_size": 500,                 ## => 글자 수 기준 (lab_07과 동일)
    "overlap_size": 100,               ## => lab_07 chunk_overlap=100과 동일
    "top_k": 4,
    "embed_provider": "huggingface",   ## => huggingface | openai
    "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  ## => lab_07에서 쓴 다국어 임베딩
    "vectorstore": "faiss",            ## => faiss | chroma | pinecone
    "search_type": "similarity",       ## => similarity | mmr
    "fetch_k": None,                   ## => candidate pool size before MMR diversification
    "query_expansion_mode": "none",   ## => none | catia_ko_en
    "answer_postprocess": "none",     ## => none | final_answer_only
    "retrieval_mode": "vector",         ## => vector | bm25 | hybrid_rrf
    "bm25_k": 6,                         ## => number of lexical retrieval candidates
    "reranker_model": None,               ## => CrossEncoder model for second-stage reranking
    "candidate_k": None,                  ## => vector candidates before reranking
    "llm_provider": "upstage",         ## => huggingface | openai | anthropic | upstage -- Upstage 키부터 발급받으므로 기본값
    "llm_model": "solar-pro2",         ## => 2026.07 기준 Upstage 최신 모델 별칭 (Solar Pro 3로 자동 연결됨)
    "prompt_style": "default",
    "max_new_tokens": None,         ## => default | cot | cite_source
    "score_threshold": None,          ## => reject retrieval below this relevance score
    "ragas_metrics": None,             ## => None means all available RAGAS metrics
    "ragas_max_workers": 1,            ## => serialize API calls to avoid rate limits
}

## => 발표 비교표에 그대로 옮겨 쓸 수 있는 실험 프리셋
EXPERIMENT_PRESETS = {
    "baseline":       DEFAULT_CONFIG,   ## => Upstage 키만 있으면 바로 실행 가능
    "small_chunk":    {**DEFAULT_CONFIG, "chunk_size": 300, "overlap_size": 50},
    "large_chunk":    {**DEFAULT_CONFIG, "chunk_size": 800, "overlap_size": 150},
    "chroma_large_chunk": {
        **DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 800, "overlap_size": 150,
        "prompt_style": "grounded_concise_cite",
    },
    "top_k_high":     {**DEFAULT_CONFIG, "top_k": 6},
    "mmr_search":     {**DEFAULT_CONFIG, "search_type": "mmr"},
    "cot_prompt":     {**DEFAULT_CONFIG, "prompt_style": "cot"},
    "kr_sbert_embed": {**DEFAULT_CONFIG, "embed_model": "snunlp/KR-SBERT-V40K-klueNLI-augSTS"},  ## => lab_04의 한국어 특화 임베딩과 비교
    "openai_embed":   {**DEFAULT_CONFIG, "embed_provider": "openai", "embed_model": "text-embedding-3-small"},  ## => lab_06과 동일, OPENAI_API_KEY 필요
    "chroma_store":   {**DEFAULT_CONFIG, "vectorstore": "chroma"},
    "chroma_top_k_4_similarity": {**DEFAULT_CONFIG, "vectorstore": "chroma", "top_k": 4, "search_type": "similarity", "prompt_style": "grounded_concise_cite"},
    "chroma_top_k_6_similarity": {**DEFAULT_CONFIG, "vectorstore": "chroma", "top_k": 6, "search_type": "similarity", "prompt_style": "grounded_concise_cite"},
    "chroma_top_k_4_mmr":        {**DEFAULT_CONFIG, "vectorstore": "chroma", "top_k": 4, "search_type": "mmr", "prompt_style": "grounded_concise_cite"},
    "chroma_top_k_6_mmr":        {**DEFAULT_CONFIG, "vectorstore": "chroma", "top_k": 6, "search_type": "mmr", "prompt_style": "grounded_concise_cite"},
    # Controlled improvement set: same model/store/search/prompt; only chunking changes.
    "chroma_mmr_500_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "mmr", "prompt_style": "grounded_strict_cite", "ragas_metrics": ["faithfulness"], "ragas_max_workers": 1},
    "chroma_mmr_650_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 650, "overlap_size": 125, "top_k": 6, "search_type": "mmr", "prompt_style": "grounded_strict_cite"},
    "chroma_mmr_800_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 800, "overlap_size": 150, "top_k": 6, "search_type": "mmr", "prompt_style": "grounded_strict_cite", "ragas_metrics": ["faithfulness"], "ragas_max_workers": 1},
    # Retrieval-focused ablation: hold chunking/prompt fixed and change only retrieval.
    "chroma_similarity_500_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "similarity", "prompt_style": "grounded_strict_cite"},
    "chroma_mmr_500_fetch30_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "mmr", "fetch_k": 30, "prompt_style": "grounded_strict_cite", "ragas_metrics": ["faithfulness"], "ragas_max_workers": 1},
    # Hybrid retrieval experiment: fixed chunk/prompt; only retrieval strategy changes.
    "bm25_500_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "retrieval_mode": "bm25", "bm25_k": 6, "prompt_style": "grounded_strict_cite"},
    "hybrid_rrf_500_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "mmr", "retrieval_mode": "hybrid_rrf", "bm25_k": 6, "prompt_style": "grounded_strict_cite"},
    # Reranker experiment: retrieve broadly, then select the six most question-relevant chunks.
    "chroma_rerank_500_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "mmr", "fetch_k": 40, "candidate_k": 20, "reranker_model": "BAAI/bge-reranker-v2-m3", "prompt_style": "grounded_strict_cite"},
    "chroma_mmr_500_curated": {**DEFAULT_CONFIG, "corpus_tag": "curated_v1", "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "mmr", "prompt_style": "grounded_strict_cite"},
    # Stronger multilingual embedding ablation; all retrieval settings remain fixed.
    "bge_m3_chroma_mmr_500_strict": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "mmr", "embed_model": "BAAI/bge-m3", "prompt_style": "grounded_strict_cite", "ragas_metrics": ["faithfulness"], "ragas_max_workers": 1},
    "bge_m3_chroma_mmr_500_bilingual": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "mmr", "embed_model": "BAAI/bge-m3", "query_expansion_mode": "catia_ko_en", "prompt_style": "grounded_strict_cite", "ragas_metrics": ["faithfulness"], "ragas_max_workers": 1},
    "bge_m3_chroma_mmr_500_normalized": {**DEFAULT_CONFIG, "vectorstore": "chroma", "chunk_size": 500, "overlap_size": 100, "top_k": 6, "search_type": "mmr", "embed_model": "BAAI/bge-m3", "answer_postprocess": "final_answer_only", "prompt_style": "grounded_strict_cite", "ragas_metrics": ["faithfulness"], "ragas_max_workers": 1},
    "pinecone_store": {**DEFAULT_CONFIG, "vectorstore": "pinecone"},  ## => lab_07의 Pinecone 실습과 동일, PINECONE_API_KEY 필요
    "openai_llm":     {**DEFAULT_CONFIG, "llm_provider": "openai", "llm_model": "gpt-5.4-nano"},  ## => lab_06/07과 동일, OPENAI_API_KEY 필요
    "claude_llm":     {**DEFAULT_CONFIG, "llm_provider": "anthropic", "llm_model": "claude-haiku-4-5-20251001"},  ## => ANTHROPIC_API_KEY 필요
    "hf_local_llm":   {**DEFAULT_CONFIG, "llm_provider": "huggingface", "llm_model": "Qwen/Qwen2.5-3B-Instruct", "max_new_tokens": 128, "prompt_style": "grounded_concise_cite"}
}
