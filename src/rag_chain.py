### [ 1. 라이브러리 임포트 ] ###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


### [ 2. LLM 선택 (실험 변수: llm_provider, llm_model) ] ###
def get_llm(provider: str = "upstage", model_name: str = "solar-pro2"):
    ## => gpt-5.4-nano는 lab_06/07에서 실제 쓴 모델명. 최신 모델명은 제공사 문서에서 확인
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=0)
    elif provider == "upstage":
        ## => Upstage Solar: OpenAI 호환 엔드포인트라 ChatOpenAI 재사용
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            base_url="https://api.upstage.ai/v1/solar",
            api_key=os.environ.get("UPSTAGE_API_KEY"),
            temperature=0,
        )
    elif provider == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        return HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device=0,   ## => GPU(RTX 4070) 사용, CPU면 -1로 변경
            pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.01},
        )
    raise ValueError(f"알 수 없는 llm_provider: {provider}")


### [ 3. 프롬프트 전략 (실험 변수: prompt_style) ] ###
## => "default"는 lab_06/07과 동일한 [문맥]/[질문]/[답변] 형식
PROMPT_STYLES = {
    "default": (
        "다음 문맥만 이용하여 질문에 답하세요.\n"
        "문맥에서 답을 찾을 수 없으면 \"문서에서 답을 찾을 수 없습니다.\"라고 답하세요.\n"
        "[문맥] {context}\n[질문] {question}\n[답변]"
    ),
    "cot": (
        "다음 문맥만 이용하여 단계적으로 근거를 정리한 뒤 질문에 답하세요.\n"
        "문맥에서 답을 찾을 수 없으면 \"문서에서 답을 찾을 수 없습니다.\"라고 답하세요.\n"
        "[문맥] {context}\n[질문] {question}\n[생각 과정과 답변]"
    ),
    "cite_source": (
        "다음 문맥 내용만 근거로 답하고, 답변 마지막 줄에 근거 문장을 그대로 적으세요.\n"
        "문맥에서 답을 찾을 수 없으면 \"문서에서 답을 찾을 수 없습니다.\"라고 답하세요.\n"
        "[문맥] {context}\n[질문] {question}\n[답변]"
    ),
}


def build_prompt(prompt_style: str = "default"):
    return ChatPromptTemplate.from_template(PROMPT_STYLES.get(prompt_style, PROMPT_STYLES["default"]))


### [ 4. RAG 체인 구성 (실험 변수: top_k, search_type) ] ###
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain(vecStore, llmModel, top_k: int = 4, search_type: str = "similarity", prompt_style: str = "default"):
    retriever = vecStore.as_retriever(search_type=search_type, search_kwargs={"k": top_k})
    prompt = build_prompt(prompt_style)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llmModel
        | StrOutputParser()
    )
    return chain, retriever


### [ 5. Direct LLM 체인 (RAG 미적용 비교군) ] ###
## => 검색 없이 LLM 사전학습 지식만으로 답변. CATIA 매뉴얼이 이미 사전학습에 포함됐을 가능성이 높다는
##    지적에 대응해, RAG 적용 전/후 답변 차이를 직접 대조하기 위한 비교군 체인
DIRECT_LLM_PROMPT = "다음 질문에 답하세요. 확실히 모르면 모른다고 답하세요.\n[질문] {question}\n[답변]"


def build_direct_llm_chain(llmModel):
    prompt = ChatPromptTemplate.from_template(DIRECT_LLM_PROMPT)
    chain = prompt | llmModel | StrOutputParser()
    return chain
