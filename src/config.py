### [ 1. 실험 기본 설정 ] ###
## => 이 값들만 바꾸면 다른 조합으로 재실험 가능
## => 요구사항: LLM / Embedding / Prompt / 벡터스토어 / chunk_size / overlap_size / top_k 비교

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEFAULT_CONFIG = {
    "pdf_path": "data",                ## => data/ 폴더에 CATIA 교육자료 PDF 여러 개를 넣어두면 전부 로드됨
    "chunk_size": 500,                 ## => 글자 수 기준 (lab_07과 동일)
    "overlap_size": 100,               ## => lab_07 chunk_overlap=100과 동일
    "top_k": 4,
    "embed_provider": "huggingface",   ## => huggingface | openai
    "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  ## => lab_07에서 쓴 다국어 임베딩
    "vectorstore": "faiss",            ## => faiss | chroma | pinecone
    "search_type": "similarity",       ## => similarity | mmr
    "llm_provider": "upstage",         ## => huggingface | openai | anthropic | upstage -- Upstage 키부터 발급받으므로 기본값
    "llm_model": "solar-pro2",         ## => 2026.07 기준 Upstage 최신 모델 별칭 (Solar Pro 3로 자동 연결됨)
    "prompt_style": "default",         ## => default | cot | cite_source
}

## => 발표 비교표에 그대로 옮겨 쓸 수 있는 실험 프리셋
EXPERIMENT_PRESETS = {
    "baseline":       DEFAULT_CONFIG,   ## => Upstage 키만 있으면 바로 실행 가능
    "small_chunk":    {**DEFAULT_CONFIG, "chunk_size": 300, "overlap_size": 50},
    "large_chunk":    {**DEFAULT_CONFIG, "chunk_size": 800, "overlap_size": 150},
    "top_k_high":     {**DEFAULT_CONFIG, "top_k": 6},
    "mmr_search":     {**DEFAULT_CONFIG, "search_type": "mmr"},
    "cot_prompt":     {**DEFAULT_CONFIG, "prompt_style": "cot"},
    "kr_sbert_embed": {**DEFAULT_CONFIG, "embed_model": "snunlp/KR-SBERT-V40K-klueNLI-augSTS"},  ## => lab_04의 한국어 특화 임베딩과 비교
    "openai_embed":   {**DEFAULT_CONFIG, "embed_provider": "openai", "embed_model": "text-embedding-3-small"},  ## => lab_06과 동일, OPENAI_API_KEY 필요
    "chroma_store":   {**DEFAULT_CONFIG, "vectorstore": "chroma"},
    "pinecone_store": {**DEFAULT_CONFIG, "vectorstore": "pinecone"},  ## => lab_07의 Pinecone 실습과 동일, PINECONE_API_KEY 필요
    "openai_llm":     {**DEFAULT_CONFIG, "llm_provider": "openai", "llm_model": "gpt-5.4-nano"},  ## => lab_06/07과 동일, OPENAI_API_KEY 필요
    "claude_llm":     {**DEFAULT_CONFIG, "llm_provider": "anthropic", "llm_model": "claude-haiku-4-5-20251001"},  ## => ANTHROPIC_API_KEY 필요
    "hf_local_llm":   {**DEFAULT_CONFIG, "llm_provider": "huggingface", "llm_model": "Qwen/Qwen2.5-7B-Instruct"},  ## => API 키 불필요, 로컬 GPU 사용
}
