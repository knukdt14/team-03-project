import os
import sys
import ssl

# Disable duplicate OpenMP runtime warnings & patch Windows SSL cert bug
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

_orig_load_default_certs = ssl.SSLContext.load_default_certs
def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _orig_load_default_certs(self, purpose)
    except Exception:
        pass
ssl.SSLContext.load_default_certs = _safe_load_default_certs

# Import vectorstore integrations
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from src.config import CHROMA_DB_DIR, LOCAL_EMBEDDING_MODEL


### [ 2. 임베딩 모델 (실험 변수: embed_provider, embed_model) ] ###
def get_embeddings(provider: str = "huggingface", model_name: str = LOCAL_EMBEDDING_MODEL):
    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif provider == "openai":
        return OpenAIEmbeddings(model=model_name)
    raise ValueError(f"알 수 없는 embed_provider: {provider}")


### [ 3. 벡터스토어 생성/저장/로드 (실험 변수: vectorstore) ] ###
def build_vectorstore(chunks, embeddings, store_type: str = "chroma", save_path: str = CHROMA_DB_DIR):
    if store_type == "faiss":
        vecStore = FAISS.from_documents(chunks, embeddings)
        vecStore.save_local(save_path)
        return vecStore
    elif store_type == "chroma":
        return Chroma.from_documents(chunks, embeddings, persist_directory=save_path)
    raise ValueError(f"알 수 없는 vectorstore: {store_type}")


def load_vectorstore(embeddings, store_type: str = "chroma", save_path: str = CHROMA_DB_DIR):
    if store_type == "faiss":
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    elif store_type == "chroma":
        return Chroma(persist_directory=save_path, embedding_function=embeddings)
    raise ValueError(f"알 수 없는 vectorstore: {store_type}")
