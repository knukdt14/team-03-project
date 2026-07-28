import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Disable duplicate OpenMP runtime warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load environment variables (.env) from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Paths matching GitHub Baseline Structure
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))  # Source PDF files folder
VECT_DIR = os.getenv("VECT_DIR", str(PROJECT_ROOT / "vect"))  # Vector DB root folder
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(PROJECT_ROOT / "vect" / "chroma_db_multimodal")) # Vector DB path
CHROMA_DB_DIR_MARKDOWN = os.getenv("CHROMA_DB_DIR_MARKDOWN", str(PROJECT_ROOT / "vect" / "chroma_db_markdown"))  # PDF->Markdown 로더용 별도 Vector DB (기존과 나란히 비교하기 위해 분리)
CHROMA_DB_DIR_MARKDOWN_ML = os.getenv("CHROMA_DB_DIR_MARKDOWN_ML", str(PROJECT_ROOT / "vect" / "chroma_db_markdown_multilingual"))  # Markdown 로더 + 다국어 임베딩 조합 (한국어 질의 검색 병목 대응)

# 기존 all-MiniLM-L6-v2는 사실상 영어 전용이라 한국어 질의 검색에 약함 (EXPERIMENTS.md 참고).
# 다국어(한국어 포함) 임베딩으로 교체 실험용 상수.
MULTILINGUAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Legacy alias compatibility
DOCS_DIR = DATA_DIR

# Teammate Git Baseline Configuration
DEFAULT_CONFIG = {
    "pdf_path": "data",  # PDF raw files in data/
    "vectorstore_path": "vect",  # VectorDB index in vect/
    "chunk_size": 600,
    "overlap_size": 100,
    "top_k": 4,
    "embed_provider": "huggingface",
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vectorstore": "chroma",
    "search_type": "similarity",
    "llm_provider": "huggingface",
    "llm_model": "Qwen/Qwen2.5-3B-Instruct",
    "llm_mode": "local",  # options: "local" (PyTorch GPU download), "api" (HF Serverless)
    "prompt_style": "default",
}

# Experiment Presets for Team Benchmarking
EXPERIMENT_PRESETS = {
    "baseline": DEFAULT_CONFIG,
    "qwen_3b_local": {**DEFAULT_CONFIG, "llm_model": "Qwen/Qwen2.5-3B-Instruct", "llm_mode": "local"},
    "qwen_1.5b_local": {**DEFAULT_CONFIG, "llm_model": "Qwen/Qwen2.5-1.5B-Instruct", "llm_mode": "local"},
    "qwen_0.5b_local": {**DEFAULT_CONFIG, "llm_model": "Qwen/Qwen2.5-0.5B-Instruct", "llm_mode": "local"},  ## => 더 작고 CATIA 지식이 얕은 모델 -> RAG 적용 전/후 차이가 더 뚜렷하게 드러남
    "qwen_7b_api": {**DEFAULT_CONFIG, "llm_model": "Qwen/Qwen2.5-7B-Instruct", "llm_mode": "api"},
    "llama_8b_api": {**DEFAULT_CONFIG, "llm_model": "meta-llama/Llama-3.1-8B-Instruct", "llm_mode": "api"},
}

# Global Fallback Defaults
DEFAULT_LLM_MODEL = DEFAULT_CONFIG["llm_model"]
LOCAL_LLM_MODEL = DEFAULT_CONFIG["llm_model"]
# Keep each embedding model paired with its own Chroma database.  Changing an
# embedding model while reusing the old vectors produces invalid retrieval.
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_CONFIG["embed_model"])
LLM_MODE = DEFAULT_CONFIG["llm_mode"]
CHUNK_SIZE = DEFAULT_CONFIG["chunk_size"]
CHUNK_OVERLAP = DEFAULT_CONFIG["overlap_size"]
DEFAULT_TOP_K = DEFAULT_CONFIG["top_k"]
