import os
import sys
import ssl

# Fix Windows SSL Certificate Store error (_ssl.c:4057)
_orig_load_default_certs = ssl.SSLContext.load_default_certs
def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _orig_load_default_certs(self, purpose)
    except Exception:
        pass
ssl.SSLContext.load_default_certs = _safe_load_default_certs

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DB_DIR, LOCAL_EMBEDDING_MODEL, DEFAULT_TOP_K
from src.multimodal_loader import load_and_split_multimodal_pdf
from src.load_pdf import load_and_split_markdown_pdf

# Import Local Hugging Face Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_function(model_name: str = LOCAL_EMBEDDING_MODEL):
    """
    Returns Local HuggingFace Embeddings for Multilingual support (No OpenAI dependency).
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"[VectorStore] Initializing Local Embeddings '{model_name}' on device '{device}'...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )


def build_or_load_vectorstore(
    persist_directory: str = CHROMA_DB_DIR,
    pdf_path: Optional[str] = None,
    force_rebuild: bool = False,
    loader: str = "multimodal",  ## => "multimodal"(기존, 이미지 개수 태그) | "markdown"(pymupdf4llm, 헤더/표 구조 보존)
    embed_model: Optional[str] = None,  ## => None이면 LOCAL_EMBEDDING_MODEL(기본, 영어 위주). 다국어로 바꾸려면 config.MULTILINGUAL_EMBEDDING_MODEL 등을 전달
) -> Chroma:
    """
    Builds or loads Chroma vector store for CATIA multimodal manual.
    """
    embeddings = get_embedding_function(embed_model) if embed_model else get_embedding_function()

    # Check if vectorstore exists
    if not force_rebuild and os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"[VectorStore] Loading existing Chroma database from: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vectorstore

    print(f"[VectorStore] Building new Chroma database (loader={loader}) at: {persist_directory}")
    if loader == "markdown":
        chunks = load_and_split_markdown_pdf(pdf_path) if pdf_path else load_and_split_markdown_pdf()
    else:
        chunks = load_and_split_multimodal_pdf(pdf_path) if pdf_path else load_and_split_multimodal_pdf()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"[VectorStore] Vector store successfully created and persisted ({len(chunks)} chunks).")
    return vectorstore


def get_retriever(vectorstore: Chroma, top_k: int = DEFAULT_TOP_K, search_type: str = "similarity"):
    """
    Returns retriever from vectorstore.
    """
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": top_k}
    )


if __name__ == "__main__":
    vs = build_or_load_vectorstore()
    retriever = get_retriever(vs, top_k=2)
    docs = retriever.invoke("동심원 구속 조건")
    print(f"Retrieved {len(docs)} docs.")
