import os
import sys
import re
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LOCAL_LLM_MODEL, LLM_MODE, RETRIEVAL_DISTANCE_THRESHOLD
from src.vector_store import build_or_load_vectorstore, get_retriever, get_hybrid_retriever
from src.models.llm_factory import LLMFactory


def clean_llm_output(text: str) -> str:
    """
    Cleans Qwen/ChatML special tokens (<|im_start|>, <|im_end|>) and prompt echo from output.
    """
    if not isinstance(text, str):
        return str(text)
        
    # Split on assistant role marker if present
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    elif "assistant\n" in text:
        text = text.split("assistant\n")[-1]
        
    # Strip ChatML special tags
    text = re.sub(r"<\|im_start\|>.*", "", text)
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return text


def _sanitize_expansion(text: str) -> str:
    """Query Expansion 출력 검증: 소형 모델이 지시를 어기고 문장/설명을 뱉으면 버리고,
    키워드 형태면 최대 5개·키워드당 20자로 제한해서 원본 질문 신호가 희석되지 않게 한다."""
    text = text.strip().strip(".").strip()
    if not text:
        return ""
    if len(text) > 100 or text.count(".") > 1:  ## => 문장형 출력(장황함) 걸러냄
        return ""
    keywords = [k.strip() for k in re.split(r"[,，]", text) if k.strip()]
    keywords = [k for k in keywords if k and len(k) <= 20][:5]
    return ", ".join(keywords)


# Query Expansion Prompt
## => zero-shot 지시만으론 소형 모델(1.5B 등)이 형식을 잘 안 지켜서 few-shot 예시 2개 추가.
##    출력은 expand_query()의 _sanitize_expansion()에서 한 번 더 검증/정제함.
EXPAND_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "당신은 CATIA V5 매뉴얼 검색 최적화 AI입니다. 질문에 있는 CATIA 기능/용어의 동의어와 영문 기능명을 콤마(,)로 구분해 5개 이내로만 출력하세요. 설명, 문장, 마침표는 쓰지 마세요."),
    ("user", "질문: 동심원의 구속 조건이 뭐야\n확장 키워드:"),
    ("assistant", "Concentricity, 동심원, 구속조건, Constraint"),
    ("user", "질문: Pad 기능 사용법 알려줘\n확장 키워드:"),
    ("assistant", "Pad, 패드, 돌출, 스케치, Sketch Based Feature"),
    ("user", "질문: {question}\n확장 키워드:")
])


# Direct LLM Prompt
DIRECT_LLM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절하고 정확한 CATIA V5 전문 AI 서비스입니다."),
    ("user", "{question}")
])


# Multimodal RAG Prompt
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "당신은 CATIA V5 전문 엔지니어입니다. 아래 매뉴얼 문맥만을 참고하여 답변하세요. 매뉴얼에 정보가 없으면 '제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다.'라고 답변하세요."),
    ("user", "[매뉴얼 문맥]\n{context}\n\n[질문]\n{question}")
])


# Procedure Prompt
PROCEDURE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "당신은 CATIA 작업 절차 정합성 평가 전문가입니다. 사용자의 작업 절차와 매뉴얼 표준을 비교하여 한국어로 피드백 리포트를 생성하세요."),
    ("user", "[참조 매뉴얼 표준]\n{context}\n\n[사용자 작업 절차]\n{user_procedure}")
])


STRICT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a CATIA manual answer extractor. Answer only from the supplied manual context. First identify the exact tool, feature, toolbar, or command name that is written in the context; copy that name exactly and do not replace it with a similar term. Then give its directly stated purpose in one short sentence. Do not use outside knowledge, infer missing steps, add examples, or add a list. If the context does not explicitly support the answer, reply exactly: 'The provided manual does not contain this information.' Your entire answer must be at most 30 words."),
    ("user", "[Manual context]\n{context}\n\n[Question]\n{question}\n\n[Answer]")
])

GROUNDING_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict CATIA manual fact checker. The proposed answer is valid only when its exact tool or feature name and stated purpose are explicitly supported by the manual context. Similar tools do not count. Reply with exactly one word: SUPPORTED or UNSUPPORTED."),
    ("user", "[Manual context]\n{context}\n\n[Question]\n{question}\n\n[Proposed answer]\n{answer}\n\n[Verdict]")
])

UNSUPPORTED_ANSWER = "The provided manual does not contain this information."


def format_docs_with_pages(docs) -> str:
    """Formats retrieved documents with page numbers and source filenames."""
    formatted = []
    for doc in docs:
        filename = doc.metadata.get("source_file", "CATIA Manual")
        page = doc.metadata.get("page", 0) + 1
        formatted.append(f"--- [{filename} - Page {page}] ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def extract_source_citations(docs) -> List[str]:
    """Extracts unique source citations in 'Filename (P.X)' format."""
    citations = []
    for d in docs:
        fname = d.metadata.get("source_file", "CATIA Manual")
        page = d.metadata.get("page", 0) + 1
        citations.append(f"{fname} (P.{page})")
    return sorted(list(dict.fromkeys(citations)))


class RAGPipeline:
    def __init__(self, model_name: str = LOCAL_LLM_MODEL, mode: str = LLM_MODE, vectorstore=None, top_k: int = 4,
                 use_hybrid_search: bool = False, hybrid_vector_weight: float = 0.5,
                 use_query_expansion: bool = False, enable_grounding_check: bool = True):
        ## => use_hybrid_search=True: 벡터 검색 + BM25 키워드 검색 결합 (GitHub 이슈 #16의
        ##    "청킹/임베딩 튜닝만으로 안 풀리는 근본적 검색 실패 케이스" 대응, vector_store.get_hybrid_retriever 참고)
        ## => enable_grounding_check=True: 답변 생성 후 GROUNDING_CHECK_PROMPT로 실제 근거 여부를
        ##    한 번 더 LLM에게 확인시킴 (이슈 #16의 "소형 모델 환각/폴백 미작동" 대응 - 거리 점수 기반
        ##    방식은 실측 결과 역효과라 폐기, 이 방식으로 대체)
        self.model_name = model_name
        self.mode = mode
        self.llm = LLMFactory.get_llm(model_name=model_name, mode=mode)
        self.vectorstore = vectorstore if vectorstore else build_or_load_vectorstore()
        self.use_hybrid_search = use_hybrid_search
        if use_hybrid_search:
            self.retriever = get_hybrid_retriever(self.vectorstore, top_k=top_k, vector_weight=hybrid_vector_weight)
        else:
            self.retriever = get_retriever(self.vectorstore, top_k=top_k)
        self.use_query_expansion = use_query_expansion
        self.enable_grounding_check = enable_grounding_check

        self.expand_chain = EXPAND_QUERY_PROMPT | self.llm | StrOutputParser()
        self.direct_chain = DIRECT_LLM_PROMPT | self.llm | StrOutputParser()
        self.grounding_chain = GROUNDING_CHECK_PROMPT | self.llm | StrOutputParser()
        
        self.rag_chain = (
            {
                "context": lambda x: format_docs_with_pages(self.retriever.invoke(x["expanded_query"])),
                "question": lambda x: x["question"]
            }
            | STRICT_RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        self.procedure_chain = (
            {
                "context": lambda x: format_docs_with_pages(self.retriever.invoke(self.expand_query(x["user_procedure"]))),
                "user_procedure": lambda x: x["user_procedure"]
            }
            | PROCEDURE_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def expand_query(self, question: str) -> str:
        try:
            raw = self.expand_chain.invoke({"question": question})
            expanded = _sanitize_expansion(clean_llm_output(raw))
            if not expanded:
                return question
            return f"{question} {expanded}"
        except Exception:
            return question

    def answer_direct(self, question: str) -> str:
        try:
            raw = self.direct_chain.invoke({"question": question})
            return clean_llm_output(raw)
        except Exception as e:
            return f"⚠️ [Direct LLM 응답 오류]: {e}"

    def _top_retrieval_distance(self, query: str) -> Optional[float]:
        """검색된 문서가 실제로 질문과 관련 있는지 확인하기 위한 최상위 거리 점수 (진단용으로만 사용).
        Chroma의 similarity_search_with_score()는 (Document, distance) 튜플을 반환하며,
        정규화 임베딩 기준 값이 낮을수록 유사함. 벡터스토어가 이 API를 지원하지 않으면 None.
        주의: 실측 결과 이 거리 점수만으로는 트릭 질문/정상 질문을 구분할 수 없음이 확인되어
        (config.RETRIEVAL_DISTANCE_THRESHOLD 참고), 아래 is_answer_grounded()가 실제 환각 방지 로직임."""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=1)
            if not results:
                return None
            return results[0][1]
        except Exception:
            return None

    def is_answer_grounded(self, question: str, answer: str, context: str) -> bool:
        """Accept only answers explicitly supported by the retrieved manual."""
        if not answer or answer.strip() == UNSUPPORTED_ANSWER:
            return False
        try:
            verdict = clean_llm_output(self.grounding_chain.invoke({
                "question": question, "answer": answer, "context": context
            })).strip().upper()
            return verdict == "SUPPORTED"
        except Exception:
            return False

    def answer_rag(self, question: str) -> Dict[str, Any]:
        expanded_q = self.expand_query(question) if self.use_query_expansion else question
        docs = self.retriever.invoke(expanded_q)
        context_str = format_docs_with_pages(docs)
        top_distance = self._top_retrieval_distance(expanded_q)

        try:
            raw_answer = self.rag_chain.invoke({"question": question, "expanded_query": expanded_q})
            answer = clean_llm_output(raw_answer)
        except Exception as e:
            answer = f"⚠️ [RAG 응답 생성 오류]: {e}"

        grounded = True
        if self.enable_grounding_check:
            grounded = self.is_answer_grounded(question, answer, context_str)
            if not grounded:
                answer = UNSUPPORTED_ANSWER

        citations = extract_source_citations(docs)

        return {
            "question": question,
            "expanded_query": expanded_q,
            "answer": answer,
            "retrieved_docs": docs,
            "context": context_str,
            "source_pages": citations,
            "top_retrieval_distance": top_distance,
            "grounded": grounded,
            "model_name": self.model_name
        }

    def answer_adaptive_fallback(self, question: str) -> Dict[str, Any]:
        rag_res = self.answer_rag(question)
        rag_ans = rag_res["answer"]
        
        unanswerable_keywords = [
            "관련 내용이 명시되어 있지 않습니다",
            "매뉴얼에 관련 내용이 없습니다",
            "매뉴얼에서 관련 정보를 찾을 수 없습니다",
            "명시되어 있지 않습니다",
            UNSUPPORTED_ANSWER,  ## => STRICT_RAG_PROMPT/grounding_check가 근거 없다고 판단하면 이 영문 문구를 씀
        ]

        keyword_missing = any(kw in rag_ans for kw in unanswerable_keywords)

        ## => 이슈 #16 대응(2차): grounding_check(is_answer_grounded)가 이미 "실제 근거 있음"을
        ##    LLM으로 재검증한 결과를 최우선으로 신뢰함 - 답변 텍스트가 어떻든 grounded=False면 폴백.
        not_grounded = rag_res.get("grounded") is False

        ## => 이슈 #16 대응(1차 시도, 참고용): 검색 거리 점수 기반 신뢰도 체크는 실측 결과 트릭/정상
        ##    질문을 구분 못 해 역효과였음이 확인되어 RETRIEVAL_DISTANCE_THRESHOLD=inf로 사실상 비활성화됨
        ##    (항상 False). 값 자체는 진단용으로만 남겨둠.
        top_distance = rag_res.get("top_retrieval_distance")
        weak_retrieval = top_distance is not None and top_distance > RETRIEVAL_DISTANCE_THRESHOLD
        is_missing = keyword_missing or weak_retrieval or not_grounded

        if is_missing:
            direct_ans = self.answer_direct(question)
            if keyword_missing:
                reason = "매뉴얼에 직접적인 근거가 없어"
            elif not_grounded:
                reason = "생성된 답변이 검색된 매뉴얼 내용으로 실제 뒷받침되지 않아(근거 검증 실패)"
            else:
                reason = "검색된 문서가 질문과 관련성이 낮아(유사도 신뢰도 미달)"
            fallback_ans = f"⚠️ **[매뉴얼 미포함 - Direct LLM 사전학습 지식 폴백 답변]**\n{reason} 일반 LLM 사전학습 지식을 바탕으로 답변합니다:\n\n{direct_ans}"
            return {
                "question": question,
                "answer": fallback_ans,
                "retrieved_docs": rag_res["retrieved_docs"],
                "context": rag_res["context"],
                "source_pages": rag_res["source_pages"],
                "used_fallback": True,
                "weak_retrieval": weak_retrieval,
                "top_retrieval_distance": top_distance,
                "status_tag": "Direct LLM Fallback (매뉴얼 미포함)",
                "model_name": self.model_name
            }
        else:
            return {
                "question": question,
                "answer": f"✅ **[매뉴얼 근거 답변]**\n{rag_ans}",
                "retrieved_docs": rag_res["retrieved_docs"],
                "context": rag_res["context"],
                "source_pages": rag_res["source_pages"],
                "used_fallback": False,
                "weak_retrieval": False,
                "top_retrieval_distance": top_distance,
                "status_tag": "RAG Manual Answer (매뉴얼 근거)",
                "model_name": self.model_name
            }

    def verify_procedure(self, user_procedure: str) -> Dict[str, Any]:
        expanded_proc = self.expand_query(user_procedure) if self.use_query_expansion else user_procedure
        docs = self.retriever.invoke(expanded_proc)
        context_str = format_docs_with_pages(docs)
        
        try:
            raw_feedback = self.procedure_chain.invoke({"user_procedure": user_procedure})
            feedback = clean_llm_output(raw_feedback)
        except Exception as e:
            feedback = f"⚠️ [정합성 검증 응답 오류]: {e}"
            
        citations = extract_source_citations(docs)
        
        return {
            "user_procedure": user_procedure,
            "feedback": feedback,
            "retrieved_docs": docs,
            "source_pages": citations,
            "model_name": self.model_name
        }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    pipeline = RAGPipeline(model_name="Qwen/Qwen2.5-3B-Instruct", mode="api")
    query = "동심원의 구속 조건"
    print("=== Testing clean output for '동심원의 구속 조건' ===")
    res = pipeline.answer_rag(query)
    print("Answer:\n", res["answer"])
