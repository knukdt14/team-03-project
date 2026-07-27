### [ 1. 라이브러리 임포트 ] ###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma   ## => lab_06/07과 동일하게 standalone 패키지 사용


### [ 2. 임베딩 모델 (실험 변수: embed_provider, embed_model) ] ###
def get_embeddings(provider: str = "huggingface", model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    if provider == "huggingface":
        ## => normalize_embeddings=True 필수: 정규화 안 하면 벡터 크기(norm)가 청크마다 제각각이라
        ##    FAISS 기본 L2 거리 검색이 "의미"가 아니라 "벡터 길이" 위주로 순위를 매겨버려
        ##    실제로 관련 있는 청크도 검색 결과에서 밀려남 (실행 검증 중 발견, EXPERIMENTS.md 참고)
        return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
    elif provider == "openai":
        return OpenAIEmbeddings(model=model_name)
    raise ValueError(f"알 수 없는 embed_provider: {provider}")


### [ 3. 벡터스토어 생성/저장/로드 (실험 변수: vectorstore) ] ###
def build_vectorstore(chunks, embeddings, store_type: str = "faiss", save_path: str = "vecstore_index"):
    if store_type == "faiss":
        vecStore = FAISS.from_documents(chunks, embeddings)
        vecStore.save_local(save_path)
        return vecStore
    elif store_type == "chroma":
        return Chroma.from_documents(chunks, embeddings, persist_directory=save_path)
    elif store_type == "pinecone":
        return _build_pinecone(chunks, embeddings)
    raise ValueError(f"알 수 없는 vectorstore: {store_type}")


def load_vectorstore(embeddings, store_type: str = "faiss", save_path: str = "vecstore_index"):
    if store_type == "faiss":
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    elif store_type == "chroma":
        return Chroma(persist_directory=save_path, embedding_function=embeddings)
    elif store_type == "pinecone":
        return _build_pinecone(None, embeddings)  ## => Pinecone은 클라우드에 이미 저장돼있어 재사용 시 재삽입 없이 연결만 함
    raise ValueError(f"알 수 없는 vectorstore: {store_type}")


### [ 3-1. Pinecone (lab_07 실습과 동일, PINECONE_API_KEY 필요) ] ###
def _build_pinecone(chunks, embeddings, index_name: str = "catia-rag-demo"):
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    pc = Pinecone()
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            dimension=384,   ## => paraphrase-multilingual-MiniLM-L12-v2 임베딩 차원에 맞춤 (모델 바꾸면 같이 변경)
            metric="cosine",
        )
    if chunks:
        return PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)
