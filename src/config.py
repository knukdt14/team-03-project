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

## => 2026-07-28: 기본값이 실측으로 이미 폐기된 영어 전용 임베딩(all-MiniLM-L6-v2)을 계속 가리키고
##    있던 문제 수정. 지금까지 보고서/PPT의 좋은 결과(F1 0.76+)는 전부 LOCAL_EMBEDDING_MODEL 환경
##    변수를 터미널에서 수동으로 BAAI/bge-m3로 오버라이드했을 때만 나온 것 -- .env.example/README
##    어디에도 이 설정이 문서화돼 있지 않아서, 새로 clone해서 그대로 실행하면 검증된 성능이 전부
##    사라지고(심하면 기존 BGE-M3 벡터스토어와 차원이 안 맞아 즉시 크래시) 재현이 안 됐음.
##    실제로 검증된 조합(wooryeol-branch 40문항 실험, F1 0.7460->0.7637)을 기본값으로 반영.
DEFAULT_CONFIG = {
    "pdf_path": "data",  # PDF raw files in data/
    "vectorstore_path": "vect",  # VectorDB index in vect/
    "chunk_size": 500,
    "overlap_size": 100,
    "top_k": 6,
    "embed_provider": "huggingface",
    "embed_model": "BAAI/bge-m3",
    "vectorstore": "chroma",
    "search_type": "mmr",
    "llm_provider": "huggingface",
    "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
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
DEFAULT_SEARCH_TYPE = DEFAULT_CONFIG["search_type"]

## => (실험 결과 비활성화됨) Adaptive Fallback RAG가 "문서를 찾긴 찾았지만 실제로는 무관한" 상황에서
##    근거 없는 답을 만들어내는 문제(GitHub 이슈 #16) 대응으로 검색 거리 기반 신뢰도 체크를 시도했으나,
##    실측 결과 이 임베딩 모델은 트릭 질문(거리 0.43~0.64)이 오히려 정상 질문(거리 0.74~0.95)보다
##    가깝게 나와 판별 기준으로 못 씀 (역효과 확인). rag_chain.py에서 거리 값 자체는 진단용으로
##    계속 기록하되, float('inf')로 자동 폴백 트리거는 사실상 비활성화. 다른 방식(키워드 중첩 등) 필요.
RETRIEVAL_DISTANCE_THRESHOLD = float('inf')
