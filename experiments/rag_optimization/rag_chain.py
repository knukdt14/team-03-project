### [ 1. 라이브러리 임포트 ] ###
import os
import re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


### [ 2. LLM 선택 (실험 변수: llm_provider, llm_model) ] ###
def get_llm(provider: str = "upstage", model_name: str = "solar-pro2", max_new_tokens=None):
    ## => gpt-5.4-nano는 lab_06/07에서 실제 쓴 모델명. 최신 모델명은 제공사 문서에서 확인
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=0)
    elif provider == "upstage":
        from langchain_upstage import ChatUpstage
        return ChatUpstage(model=model_name)
    
    elif provider == "huggingface":
        import torch
        from langchain_huggingface import HuggingFacePipeline
        return HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            # FP16 reduces Qwen 3B's weight memory from roughly 12 GB to 6 GB,
            # helping it fit in the RTX 4070 Laptop GPU's 8 GB VRAM.
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
            pipeline_kwargs={"max_new_tokens": max_new_tokens or 512, "temperature": 0.01},
        )
    raise ValueError(f"알 수 없는 llm_provider: {provider}")


### [ 3. 프롬프트 전략 (실험 변수: prompt_style) ] ###
## => "default"는 lab_06/07과 동일한 [문맥]/[질문]/[답변] 형식
PROMPT_STYLES = {
    'default': '다음 문맥만 이용하여 질문에 답하세요.\n문맥에서 답을 찾을 수 없으면 "문서에서 답을 찾을 수 없습니다."라고 답하세요.\n[문맥] {context}\n[질문] {question}\n[답변]',
    'cot': '다음 문맥만 이용하여 단계적으로 근거를 정리한 뒤 질문에 답하세요.\n문맥에서 답을 찾을 수 없으면 "문서에서 답을 찾을 수 없습니다."라고 답하세요.\n[문맥] {context}\n[질문] {question}\n[생각 과정과 답변]',
    'cite_source': '다음 문맥 내용만 근거로 답하고, 답변 마지막 줄에 근거 문장을 그대로 적으세요.\n문맥에서 답을 찾을 수 없으면 "문서에서 답을 찾을 수 없습니다."라고 답하세요.\n[문맥] {context}\n[질문] {question}\n[답변]',
    'grounded_concise_cite': "아래 [문서]에 포함된 정보만 근거로 답변하세요. 문서에 근거가 없거나 불충분하면 반드시 '문서에서 답을 찾을 수 없습니다.'라고 답하고 추측하거나 일반 지식을 사용하지 마세요.\n답변은 2~4개의 간결한 문장으로 작성하세요. 마지막 문장에는 '출처:'를 붙여, 답변의 근거가 된 문서 내용 또는 문서명을 짧게 표시하세요.\n[문서] {context}\n[질문] {question}\n[답변]",
    'grounded_strict_cite': "아래 [문서]에 답의 직접 근거가 없으면 반드시 다음 한 문장만 그대로 답하세요.\n문서에서 답을 찾을 수 없습니다.\n문서에 답이 있을 때만 문서 근거로 1~2문장으로 간결하게 답하세요. 추측, 일반 지식, 추가 설명은 금지합니다.\n답할 수 있는 경우에만 마지막 줄에 '출처:'를 붙이세요.\n[문서] {context}\n[질문] {question}\n[답변]",
}

def build_prompt(prompt_style: str = "default"):
    return ChatPromptTemplate.from_template(PROMPT_STYLES.get(prompt_style, PROMPT_STYLES["default"]))


### [ 4. RAG 체인 구성 (실험 변수: top_k, search_type) ] ###
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


_REFUSAL = "문서에서 답을 찾을 수 없습니다."
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?\s*[\[\(]?\s*(?:최종\s*답변|final\s*answer|정답|답변)\s*[\]\)]?(?:\*\*)?\s*:?\s*(.+)",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalize_final_answer(text: str) -> str:
    """Keep an explicitly marked final answer and normalize safe refusals.

    The model occasionally emits draft analysis before a final response. This
    postprocessor is intentionally conservative: it only extracts an explicit
    final-answer section or collapses an answer that begins with the fixed
    refusal sentence.
    """
    cleaned = str(text).strip()
    matches = list(_FINAL_ANSWER_PATTERN.finditer(cleaned))
    if matches:
        cleaned = matches[-1].group(1).strip().lstrip("* \t\r\n")
    if cleaned.startswith(_REFUSAL):
        return _REFUSAL
    return cleaned



def build_reranker(model_name: str):
    """Load a multilingual cross-encoder for second-stage document reranking."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _rerank_documents(question, docs, reranker, top_k: int):
    if not docs:
        return []
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

def build_bm25_retriever(chunks, k: int = 6):
    """Create a lexical retriever from the same chunks used by the vector store."""
    from langchain_community.retrievers import BM25Retriever
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def _document_key(doc):
    metadata = getattr(doc, "metadata", {})
    return (metadata.get("source"), metadata.get("page"), metadata.get("chunk_id"), doc.page_content)


def _rrf_merge(vector_docs, bm25_docs, top_k: int, rrf_k: int = 60):
    """Fuse ranked vector and keyword results using reciprocal-rank fusion."""
    scores, documents = {}, {}
    for ranked_docs in (vector_docs, bm25_docs):
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _document_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            documents[key] = doc
    return [documents[key] for key in sorted(scores, key=scores.get, reverse=True)[:top_k]]


def _rrf_merge_many(ranked_lists, top_k: int, rrf_k: int = 60):
    """Fuse any number of ranked retrieval lists while removing duplicate chunks."""
    scores, documents = {}, {}
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _document_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            documents[key] = doc
    return [documents[key] for key in sorted(scores, key=scores.get, reverse=True)[:top_k]]


_KO_EN_QUERY_PHRASES = {
    "솔리드 파트": "CATIA solid part file extension CATPart",
    "파트 파일": "CATIA part file extension CATPart",
    "어셈블리 정보": "CATIA assembly information file extension CATProduct",
    "어셈블리를 만드는": "CATIA assembly creation steps CATProduct Product Structure Constraints",
    "도면": "CATIA drawing file extension CATDrawing",
    "보라색": "CATIA sketch purple over-constrained constraint",
    "Boolean": "CATIA Boolean operation Insert New Body additional Body",
    "외부 형상": "CATIA import geometry IGES STEP AP203 DXF DWG",
    "구조 해석": "CATIA Generative Part Structural Analysis limited mesh control solid geometry",
    "Knowledge Advisor": "CATIA Knowledge Advisor parameters formulas relations",
    "Constraints": "CATIA Assembly Design Constraints Toolbar Coincidence Contact Offset Angular Anchor Fix Together",
    "기존 컴포넌트": "CATIA insert existing component assembly",
    "확장자": "CATIA file extension",
}


def build_catia_ko_en_query(question: str):
    """Create a low-latency English retrieval expansion for common CATIA terms.

    This intentionally uses a deterministic glossary instead of an extra LLM call,
    so the retrieval experiment isolates cross-lingual search from generation cost.
    """
    if not any("가" <= char <= "힣" for char in question):
        return None
    phrases = [english for korean, english in _KO_EN_QUERY_PHRASES.items() if korean.lower() in question.lower()]
    if not phrases:
        return None
    return " ".join(dict.fromkeys(phrases))


def build_rag_chain(vecStore, llmModel, top_k: int = 4, search_type: str = "similarity",
                    prompt_style: str = "default", score_threshold=None, fetch_k=None,
                    retrieval_mode: str = "vector", bm25_retriever=None,
                    reranker=None, candidate_k=None, query_expansion_mode: str = "none",
                    answer_postprocess: str = "none"):
    """Build vector, lexical, or RRF-hybrid retrieval before answer generation."""
    retrieval_k = candidate_k if reranker is not None and candidate_k is not None else top_k
    search_kwargs = {"k": retrieval_k}
    if search_type == "mmr" and fetch_k is not None:
        search_kwargs["fetch_k"] = max(fetch_k, retrieval_k)
    vector_retriever = vecStore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def retrieve_vector(question):
        if score_threshold is not None:
            best = vecStore.similarity_search_with_relevance_scores(question, k=1)
            if not best or best[0][1] < score_threshold:
                return []
        return vector_retriever.invoke(question)

    def retrieve_vector_bilingual(question):
        original_docs = retrieve_vector(question)
        english_query = build_catia_ko_en_query(question)
        if not english_query:
            return original_docs
        english_docs = retrieve_vector(english_query)
        return _rrf_merge_many([original_docs, english_docs], top_k)

    vector_lookup = retrieve_vector_bilingual if query_expansion_mode == "catia_ko_en" else retrieve_vector

    if retrieval_mode == "bm25":
        if bm25_retriever is None:
            raise ValueError("BM25 retrieval requires a BM25 retriever")
        retriever = bm25_retriever
    elif retrieval_mode == "hybrid_rrf":
        if bm25_retriever is None:
            raise ValueError("Hybrid retrieval requires a BM25 retriever")
        retriever = RunnableLambda(
            lambda question: _rrf_merge(vector_lookup(question), bm25_retriever.invoke(question), top_k)
        )
    else:
        retriever = RunnableLambda(vector_lookup) if score_threshold is not None or query_expansion_mode != "none" else vector_retriever

    if reranker is not None:
        first_stage_retriever = retriever
        retriever = RunnableLambda(
            lambda question: _rerank_documents(question, first_stage_retriever.invoke(question), reranker, top_k)
        )

    prompt = build_prompt(prompt_style)
    output_parser = StrOutputParser()
    if answer_postprocess == "final_answer_only":
        output_parser = output_parser | RunnableLambda(normalize_final_answer)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llmModel
        | output_parser
    )
    return chain, retriever
