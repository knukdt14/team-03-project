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
    "qwen_7b_api": {**DEFAULT_CONFIG, "llm_model": "Qwen/Qwen2.5-7B-Instruct", "llm_mode": "api"},
    "llama_8b_api": {**DEFAULT_CONFIG, "llm_model": "meta-llama/Llama-3.1-8B-Instruct", "llm_mode": "api"},
}

# Global Fallback Defaults
DEFAULT_LLM_MODEL = DEFAULT_CONFIG["llm_model"]
LOCAL_LLM_MODEL = DEFAULT_CONFIG["llm_model"]
LOCAL_EMBEDDING_MODEL = DEFAULT_CONFIG["embed_model"]
LLM_MODE = DEFAULT_CONFIG["llm_mode"]
CHUNK_SIZE = DEFAULT_CONFIG["chunk_size"]
CHUNK_OVERLAP = DEFAULT_CONFIG["overlap_size"]
DEFAULT_TOP_K = DEFAULT_CONFIG["top_k"]
