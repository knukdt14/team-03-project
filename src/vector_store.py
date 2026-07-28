import os
import sys
import ssl
import torch

# Fix Windows SSL Certificate Store error (_ssl.c:4057)
_orig_load_default_certs = ssl.SSLContext.load_default_certs
def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _orig_load_default_certs(self, purpose)
    except Exception:
        pass
ssl.SSLContext.load_default_certs = _safe_load_default_certs

import shutil
from typing import List, Optional
from langchain_core.documents import Document
# Use the dedicated integration package.  The legacy community wrapper is
# deprecated and can be incompatible with current Chroma HNSW persistence.
from langchain_chroma import Chroma
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VectorStore] Initializing Local Embeddings '{model_name}' on {device}...")
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

    if force_rebuild and os.path.exists(persist_directory):
        ## => Chroma.from_documents는 기존 persist_directory에 append하므로, force_rebuild 시 먼저 비워야 중복 저장을 막을 수 있음
        print(f"[VectorStore] force_rebuild=True -> 기존 컬렉션 삭제: {persist_directory}")
        shutil.rmtree(persist_directory)

    print(f"[VectorStore] Building new Chroma database (loader={loader}) at: {persist_directory}")
    if loader == "markdown":
        chunks = load_and_split_markdown_pdf(pdf_path) if pdf_path else load_and_split_markdown_pdf()
    else:
        chunks = load_and_split_multimodal_pdf(pdf_path) if pdf_path else load_and_split_multimodal_pdf()

    # Adding all multimodal chunks in one upsert can corrupt the HNSW
    # compaction step on Windows.  Persist smaller deterministic batches.
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        ids = [f"chunk-{start + offset}" for offset in range(len(batch))]
        vectorstore.add_documents(documents=batch, ids=ids)
        print(f"[VectorStore] Indexed {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")
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


def _get_all_documents_from_chroma(vectorstore: Chroma) -> List[Document]:
    """Chroma 컬렉션에 저장된 모든 청크를 Document 리스트로 복원 (BM25 인덱싱용, PDF 재파싱 불필요)."""
    raw = vectorstore.get(include=["documents", "metadatas"])
    return [Document(page_content=t, metadata=m or {}) for t, m in zip(raw["documents"], raw["metadatas"])]


def get_hybrid_retriever(vectorstore: Chroma, top_k: int = DEFAULT_TOP_K, vector_weight: float = 0.5):
    """
    벡터 검색(의미 유사도) + BM25(키워드 매칭)를 결합한 하이브리드 리트리버.

    청킹/임베딩 튜닝만으로는 안 풀리는 근본적 검색 실패 케이스(GitHub 이슈 #16) 대응:
    ".CATPart 확장자는?" 같은 질문은 정답 청크에 ".CATPart"라는 정확한 문자열이 있는데도
    다국어 임베딩 벡터 검색으로는 top-k에 안 들어오는 경우가 있었음. BM25는 임베딩 없이
    정확한 단어/문자열 매칭으로 찾기 때문에, 이런 "정확한 키워드는 일치하는데 의미 벡터로는
    안 잡히는" 케이스를 보완할 수 있음.

    vector_weight: 0~1 사이. 1이면 벡터 검색만, 0이면 BM25만. 기본 0.5(동일 비중).
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers.ensemble import EnsembleRetriever

    documents = _get_all_documents_from_chroma(vectorstore)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1 - vector_weight, vector_weight],
    )


if __name__ == "__main__":
    vs = build_or_load_vectorstore()
    retriever = get_retriever(vs, top_k=2)
    docs = retriever.invoke("동심원 구속 조건")
    print(f"Retrieved {len(docs)} docs.")
