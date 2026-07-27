import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from src.rag_chain import RAGPipeline
from src.evaluation import BENCHMARK_DATASET, run_evaluation_benchmark

# Page Configuration
st.set_page_config(
    page_title="CATIA 멀티모달 RAG & 작업 절차 정합성 검증 시스템",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic design
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .badge-rag-strict {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-rag-adaptive {
        background-color: #3B82F6;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-rag-off {
        background-color: #EF4444;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline(model_name: str, mode: str):
    """Cache Multimodal RAG Pipeline initialization by model name and execution mode."""
    return RAGPipeline(model_name=model_name, mode=mode)


def main():
    st.markdown('<div class="main-header">🛠️ CATIA 멀티모달 RAG & 작업 절차 정합성 검증 시스템</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">오픈소스 소형 LLM(Qwen2.5) + 멀티모달 도면 파싱 기반 CATIA 전용 도움말 챗봇</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/catia.png", width=70)
    st.sidebar.title("📌 프로젝트 정보")
    st.sidebar.info("""
    **KDT 14기 3팀 프로젝트**
    - **통합 멀티모달 매뉴얼**: 32개 PDF 문헌 (2,418페이지 / 5,278개 멀티모달 청크)
    - **핵심 기술**: PyMuPDF 도면/캡처 태깅, Query Expansion, Open LLM (Qwen2.5), Local HF Stack
    """)
    
    execution_mode_label = st.sidebar.radio(
        "⚙️ 모델 구동 방식 선택:",
        [
            "🏠 로컬 다운로드 구동 (Local HF Transformers)",
            "☁️ 클라우드 API 구동 (HuggingFace Serverless API)"
        ]
    )
    execution_mode = "local" if "로컬" in execution_mode_label else "api"

    selected_model_label = st.sidebar.selectbox(
        "🤖 오픈소스 LLM 모델 선택:",
        [
            "Qwen/Qwen2.5-3B-Instruct (권장 3B)",
            "Qwen/Qwen2.5-1.5B-Instruct (초경량 1.5B)",
            "Qwen/Qwen2.5-7B-Instruct (7B)",
            "meta-llama/Llama-3.1-8B-Instruct (8B)"
        ]
    )

    model_map = {
        "Qwen/Qwen2.5-3B-Instruct (권장 3B)": "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct (초경량 1.5B)": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct (7B)": "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct (8B)": "meta-llama/Llama-3.1-8B-Instruct"
    }
    selected_model_name = model_map[selected_model_label]
    
    top_k = st.sidebar.slider("검색 문서 수 (Top-K)", min_value=1, max_value=10, value=4)

    # Load Pipeline
    with st.spinner(f"[{selected_model_label}] 멀티모달 파이프라인 및 벡터 DB 로딩 중..."):
        pipeline = load_rag_pipeline(selected_model_name, execution_mode)
        pipeline.retriever.search_kwargs["k"] = top_k

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📖 멀티모달 질의응답 (QA)", "⚙️ 작업 절차 정합성 검증", "📊 3-Way 모델 비교 A/B 벤치마크"])

    # TAB 1: QA
    with tab1:
        st.subheader("📖 CATIA 멀티모달 매뉴얼 한국어 질의응답")
        st.write(f"**현재 구동 모델**: `{selected_model_name}` | **방식**: `{execution_mode_label}`")

        # Preset Questions
        preset_q = st.selectbox(
            "💡 샘플 질문 선택:",
            [
                "직접 입력",
                "동심원의 구속 조건",
                "CATIA V5에서 Pad 기능의 역할과 스케치 조건은 무엇인가요?",
                "Stiffener 기능의 사용 목적과 권장 구배 각도는 몇 도인가요?",
                "Draft Angle 기능이란 무엇이며 어떨 때 사용하나요?",
                "CATIA V5에서 3D 퀀텀 머시닝(Quantum Machining) AI 가속 모드는 어떻게 하나요?",
                "CATIA V5에서 자율주행 차체 3D 홀로그램 자동 설계 알고리즘 메뉴 위치는?"
            ]
        )

        user_query = st.text_input(
            "질문을 입력하세요:",
            value="" if preset_q == "직접 입력" else preset_q,
            placeholder="예: CATIA Pad 기능에서 First Limit 설정 방법은?"
        )

        mode = st.radio(
            "실행 방식 선택:",
            [
                "🎯 Adaptive Fallback RAG (매뉴얼 우선 + 미포함 시 Direct LLM 폴백)",
                "🔒 Strict RAG (매뉴얼 엄격 전용 - 없으면 '매뉴얼에 없음' 응답)",
                "⚡ Direct LLM (RAG Off - 사전학습 지식만)"
            ],
            horizontal=False
        )

        if st.button("질문하기 🚀", key="btn_qa"):
            if not user_query.strip():
                st.warning("질문을 입력해주세요.")
            else:
                st.divider()
                
                # 1. 질문 (User Question)
                st.markdown("### ❓ 질문")
                st.info(user_query)

                if "Adaptive Fallback RAG" in mode:
                    with st.spinner("멀티모달 매뉴얼 검색 및 적응형 폴백 분석 중..."):
                        res = pipeline.answer_adaptive_fallback(user_query)
                    
                    # 2. 답변 (Answer)
                    st.markdown("### 🤖 답변")
                    st.markdown(f'<span class="badge-rag-adaptive">Adaptive Fallback RAG ({res["status_tag"]})</span>', unsafe_allow_html=True)
                    if res["used_fallback"]:
                        st.warning(res["answer"])
                    else:
                        st.success(res["answer"])
                    
                    # 3. 참고 문서 내용 (Context Chunks & Citations)
                    st.markdown("### 📚 참고 문서 내용 (Context Chunks)")
                    st.markdown(f"**📌 참조 출처 문헌 및 페이지**: `{', '.join(res['source_pages'])}`")
                    
                    for i, doc in enumerate(res["retrieved_docs"]):
                        fname = doc.metadata.get('source_file', 'CATIA Manual')
                        page_num = doc.metadata.get('page', 0) + 1
                        with st.expander(f"📄 참고 문서 Chunk {i+1} : [{fname} - Page {page_num}]"):
                            st.text(doc.page_content)

                elif "Strict RAG" in mode:
                    with st.spinner("멀티모달 매뉴얼 엄격 검색 및 답변 생성 중..."):
                        res = pipeline.answer_rag(user_query)
                    
                    # 2. 답변 (Answer)
                    st.markdown("### 🤖 답변")
                    st.markdown(f'<span class="badge-rag-strict">Strict RAG (매뉴얼 엄격 전용)</span>', unsafe_allow_html=True)
                    st.success(res["answer"])
                    
                    # 3. 참고 문서 내용 (Context Chunks & Citations)
                    st.markdown("### 📚 참고 문서 내용 (Context Chunks)")
                    st.markdown(f"**📌 참조 출처 문헌 및 페이지**: `{', '.join(res['source_pages'])}`")
                    
                    for i, doc in enumerate(res["retrieved_docs"]):
                        fname = doc.metadata.get('source_file', 'CATIA Manual')
                        page_num = doc.metadata.get('page', 0) + 1
                        with st.expander(f"📄 참고 문서 Chunk {i+1} : [{fname} - Page {page_num}]"):
                            st.text(doc.page_content)

                else:
                    with st.spinner("Direct LLM 답변 생성 중..."):
                        ans_direct = pipeline.answer_direct(user_query)
                    
                    # 2. 답변 (Answer)
                    st.markdown("### 🤖 답변")
                    st.markdown(f'<span class="badge-rag-off">Direct LLM (RAG Off)</span>', unsafe_allow_html=True)
                    st.info(ans_direct)
                    
                    # 3. 참고 문서 내용 (Context Chunks - None for Direct LLM)
                    st.markdown("### 📚 참고 문서 내용")
                    st.caption("⚡ Direct LLM 모드는 RAG를 사용하지 않으므로 참조 문서가 없습니다.")

    # TAB 2: Procedure Verification
    with tab2:
        st.subheader("⚙️ 작업 절차 정합성 검증 (Procedure Verification)")
        st.write("사용자가 수행하려는 작업 절차를 입력하면 매뉴얼 표준과 1:1 비교하여 정합성 및 오류/누락을 피드백합니다.")

        preset_proc = st.selectbox(
            "💡 샘플 작업 절차 선택:",
            [
                "직접 입력",
                "1. 스케치 작성 -> 2. Pad 돌출 생성 -> 3. Casing 내부에 Stiffener 리브 생성 -> 4. 4도 Draft 경사각 적용",
                "1. Pad 생성 -> 2. 스케치 작성 -> 3. 바로 해석 실행 (스케치 및 재질 설정 없이)",
                "1. 2D 프로파일 생성 -> 2. Pocket 홈 파기 -> 3. Fillet 라운딩 처리"
            ]
        )

        user_proc_text = st.text_area(
            "검증할 작업 절차를 입력하세요:",
            value="" if preset_proc == "직접 입력" else preset_proc,
            height=120,
            placeholder="1. 2D 스케치 작성\n2. Pad 기능으로 3D 솔리드 생성\n3. Stiffener로 내부에 리브 생성"
        )

        if st.button("정합성 검증 실행 🔍", key="btn_proc"):
            if not user_proc_text.strip():
                st.warning("작업 절차를 입력해주세요.")
            else:
                st.divider()
                st.markdown("### ❓ 검증 대상 작업 절차")
                st.info(user_proc_text)
                
                with st.spinner("매뉴얼 표준과 정합성 비교 분석 중..."):
                    proc_res = pipeline.verify_procedure(user_proc_text)
                
                st.markdown("### 🤖 검증 리포트 및 피드백")
                st.markdown(proc_res["feedback"])
                
                st.markdown("### 📚 참고 문서 내용 (Context Chunks)")
                st.markdown(f"**📌 참조 출처 문헌 및 페이지**: `{', '.join(proc_res['source_pages'])}`")
                for i, doc in enumerate(proc_res["retrieved_docs"]):
                    fname = doc.metadata.get('source_file', 'CATIA Manual')
                    page_num = doc.metadata.get('page', 0) + 1
                    with st.expander(f"📄 참고 문서 Chunk {i+1} : [{fname} - Page {page_num}]"):
                        st.text(doc.page_content)

    # TAB 3: A/B Testing Benchmark
    with tab3:
        st.subheader("📊 3-Way 모델 비교 A/B 테스트 (Direct LLM vs Strict RAG vs Adaptive Fallback RAG)")
        csv_file_path = os.path.join(PROJECT_ROOT, "data", "evaluation_results.csv")
        
        btn_eval = st.button("3-Way 벤치마크 평가 다시 실행 🧪", key="btn_eval")
        
        df_eval = None
        if btn_eval:
            with st.spinner("벤치마크 데이터셋 5개 항목에 대해 3개 모델 대조 평가 측정 중..."):
                df_eval = run_evaluation_benchmark(pipeline, BENCHMARK_DATASET)
                st.success("벤치마크 평가 완료!")
        elif os.path.exists(csv_file_path):
            try:
                df_eval = pd.read_csv(csv_file_path)
            except Exception:
                df_eval = None

        if df_eval is not None:
            cols = list(df_eval.columns)
            direct_col = "Direct LLM F1" if "Direct LLM F1" in cols else ("RAG Off BERTScore F1" if "RAG Off BERTScore F1" in cols else cols[4])
            strict_col = "Strict RAG F1" if "Strict RAG F1" in cols else ("RAG On BERTScore F1" if "RAG On BERTScore F1" in cols else cols[6])
            adaptive_col = "Adaptive Fallback F1" if "Adaptive Fallback F1" in cols else ("Adaptive Status F1" if "Adaptive Status F1" in cols else strict_col)

            direct_f1 = df_eval[direct_col].mean()
            strict_f1 = df_eval[strict_col].mean()
            adaptive_f1 = df_eval[adaptive_col].mean() if adaptive_col in cols else strict_f1
            
            col1, col2, col3 = st.columns(3)
            col1.metric("1. Direct LLM F1", f"{direct_f1:.4f}")
            col2.metric("2. Strict RAG F1", f"{strict_f1:.4f}", delta=f"+{(strict_f1 - direct_f1):.4f}")
            col3.metric("3. Adaptive Fallback F1", f"{adaptive_f1:.4f}", delta=f"+{(adaptive_f1 - direct_f1):.4f}")
            
            st.markdown("### 📋 상세 3개 모델 대조 결과 표")
            raw_display_cols = [c for c in [
                "Type", "Question",
                "Direct LLM Answer", direct_col,
                "Strict RAG Answer", strict_col,
                "Adaptive Fallback Answer", "Adaptive Status", adaptive_col,
                "Source Pages"
            ] if c in df_eval.columns]
            
            display_cols = list(dict.fromkeys(raw_display_cols))
            st.dataframe(df_eval[display_cols])
        else:
            st.info("버튼을 눌러 3-Way 벤치마크 평가를 실행하세요.")


if __name__ == "__main__":
    main()
