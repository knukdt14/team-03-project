import os
import sys
import re
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LOCAL_LLM_MODEL, LLM_MODE
from src.vector_store import build_or_load_vectorstore, get_retriever
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


# Query Expansion Prompt
EXPAND_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "당신은 CATIA V5 전문 검색 최적화 AI입니다. 질문의 동의어, 영문 기능명(Concentricity, Coincidence, Pad, Pocket 등), 한글 용어를 조합한 키워드만 단답형으로 출력하세요."),
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
    def __init__(self, model_name: str = LOCAL_LLM_MODEL, mode: str = LLM_MODE, vectorstore=None, top_k: int = 4):
        self.model_name = model_name
        self.mode = mode
        self.llm = LLMFactory.get_llm(model_name=model_name, mode=mode)
        self.vectorstore = vectorstore if vectorstore else build_or_load_vectorstore()
        self.retriever = get_retriever(self.vectorstore, top_k=top_k)
        
        self.expand_chain = EXPAND_QUERY_PROMPT | self.llm | StrOutputParser()
        self.direct_chain = DIRECT_LLM_PROMPT | self.llm | StrOutputParser()
        
        self.rag_chain = (
            {
                "context": lambda x: format_docs_with_pages(self.retriever.invoke(x["expanded_query"])),
                "question": lambda x: x["question"]
            }
            | RAG_PROMPT
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
            expanded = clean_llm_output(raw)
            return f"{question} {expanded}"
        except Exception:
            return question

    def answer_direct(self, question: str) -> str:
        try:
            raw = self.direct_chain.invoke({"question": question})
            return clean_llm_output(raw)
        except Exception as e:
            return f"⚠️ [Direct LLM 응답 오류]: {e}"

    def answer_rag(self, question: str) -> Dict[str, Any]:
        expanded_q = self.expand_query(question)
        docs = self.retriever.invoke(expanded_q)
        context_str = format_docs_with_pages(docs)
        
        try:
            raw_answer = self.rag_chain.invoke({"question": question, "expanded_query": expanded_q})
            answer = clean_llm_output(raw_answer)
        except Exception as e:
            answer = f"⚠️ [RAG 응답 생성 오류]: {e}"
            
        citations = extract_source_citations(docs)
        
        return {
            "question": question,
            "expanded_query": expanded_q,
            "answer": answer,
            "retrieved_docs": docs,
            "context": context_str,
            "source_pages": citations,
            "model_name": self.model_name
        }

    def answer_adaptive_fallback(self, question: str) -> Dict[str, Any]:
        rag_res = self.answer_rag(question)
        rag_ans = rag_res["answer"]
        
        unanswerable_keywords = [
            "관련 내용이 명시되어 있지 않습니다",
            "매뉴얼에 관련 내용이 없습니다",
            "매뉴얼에서 관련 정보를 찾을 수 없습니다",
            "명시되어 있지 않습니다"
        ]
        
        is_missing = any(kw in rag_ans for kw in unanswerable_keywords)
        
        if is_missing:
            direct_ans = self.answer_direct(question)
            fallback_ans = f"⚠️ **[매뉴얼 미포함 - Direct LLM 사전학습 지식 폴백 답변]**\n매뉴얼에 직접적인 근거가 없어 일반 LLM 사전학습 지식을 바탕으로 답변합니다:\n\n{direct_ans}"
            return {
                "question": question,
                "answer": fallback_ans,
                "retrieved_docs": rag_res["retrieved_docs"],
                "context": rag_res["context"],
                "source_pages": rag_res["source_pages"],
                "used_fallback": True,
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
                "status_tag": "RAG Manual Answer (매뉴얼 근거)",
                "model_name": self.model_name
            }

    def verify_procedure(self, user_procedure: str) -> Dict[str, Any]:
        expanded_proc = self.expand_query(user_procedure)
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
