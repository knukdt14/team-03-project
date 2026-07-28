import os
import sys
import ssl
from pathlib import Path

# Monkey-patch Windows SSL cert store bug (_ssl.c:4057)
_orig_load_default_certs = ssl.SSLContext.load_default_certs
def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _orig_load_default_certs(self, purpose)
    except Exception:
        pass
ssl.SSLContext.load_default_certs = _safe_load_default_certs

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHROMA_DB_DIR_MARKDOWN_ML, DATA_DIR, MULTILINGUAL_EMBEDDING_MODEL
from src.vector_store import build_or_load_vectorstore

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("==================================================")
    print("   Building Markdown + Multilingual Embedding      ")
    print("   Vector Store in vect/                           ")
    print("==================================================")
    print(f"[Info] Source PDF Directory: {DATA_DIR}")
    print(f"[Info] Target VectorDB Directory: {CHROMA_DB_DIR_MARKDOWN_ML}")
    print(f"[Info] Embedding Model: {MULTILINGUAL_EMBEDDING_MODEL}")

    vs = build_or_load_vectorstore(
        persist_directory=CHROMA_DB_DIR_MARKDOWN_ML,
        force_rebuild=True,
        loader="markdown",
        embed_model=MULTILINGUAL_EMBEDDING_MODEL,
    )
    print("[Success] Markdown + Multilingual Vector Store Build Complete in vect/!")

if __name__ == "__main__":
    main()
