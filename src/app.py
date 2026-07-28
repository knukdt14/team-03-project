import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from pypdf import PdfReader
from src.rag_chain import RAGPipeline
from src.evaluation import BENCHMARK_DATASET, run_evaluation_benchmark
from src.config import DATA_DIR

# Page Configuration
st.set_page_config(
    page_title="CATIA 멀티모달 RAG & 작업 절차 정합성 검증 시스템",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic design -- 밝은 화이트 톤 + 네이비블루 포인트
st.markdown("""
<style>
    /* ---------- 헤더 ---------- */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.3rem;
        padding-bottom: 0.7rem;
        border-bottom: 3px solid #1E3A8A;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-top: 0.8rem;
        margin-bottom: 1.8rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* ---------- 배지 ---------- */
    .badge-rag-strict, .badge-rag-adaptive, .badge-rag-off {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
        margin-bottom: 0.7rem;
    }
    .badge-rag-strict { background-color: #10B981; color: white; }
    .badge-rag-adaptive { background-color: #1E3A8A; color: white; }
    .badge-rag-off { background-color: #EF4444; color: white; }

    /* ---------- 사이드바 ---------- */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] h1 {
        color: #1E3A8A;
        font-size: 1.3rem;
    }

    /* ---------- 버튼 ---------- */
    .stButton > button, .stFormSubmitButton > button {
        background-color: #1E3A8A;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.6rem;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #2748A8;
        color: white;
    }

    /* ---------- 탭 ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }

    /* ---------- 참고 문서 expander ---------- */
    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        margin-bottom: 0.6rem;
        overflow: hidden;
    }

    /* ---------- 지표 카드 (3-Way 벤치마크) ---------- */
    [data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline(model_name: str, mode: str):
    """Cache Multimodal RAG Pipeline initialization by model name and execution mode."""
    return RAGPipeline(model_name=model_name, mode=mode)


@st.cache_resource
def get_corpus_stats(_pipeline):
    """PDF/페이지/청크 수를 실제 데이터에서 집계 -- 청킹 로직이 바뀔 때마다 사이드바 문구를
    손으로 고쳐줄 필요 없도록 하드코딩 대신 동적으로 계산."""
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    total_pages = 0
    for f in pdf_files:
        try:
            total_pages += len(PdfReader(os.path.join(DATA_DIR, f)).pages)
        except Exception:
            pass
    try:
        chunk_count = _pipeline.vectorstore._collection.count()
    except Exception:
        chunk_count = None
    return len(pdf_files), total_pages, chunk_count


def render_chunk_images(doc):
    """이 청크의 문단에 배정된 CAD 도면/스크린샷 이미지를 화면에 표시한다(bbox 근접 매칭 기준)."""
    raw_paths = doc.metadata.get("image_paths", "")
    if not raw_paths:
        return
    image_paths = [p for p in raw_paths.split(";") if p and os.path.exists(p)]
    for path in image_paths:
        ## => 리사이즈 불가능한 이미지 하나 때문에 전체 페이지가 죽지 않도록 개별 처리
        try:
            st.image(path, width=300)
        except Exception:
            continue


def render_context_chunks(docs):
    """검색된 청크를 (파일명, 페이지) 기준으로 묶어서 표시한다 -- 문단 단위 청킹이라 같은 페이지에서
    여러 청크가 뽑히는 경우가 흔해져서, 페이지 헤더를 중복 표시하지 않고 하나로 묶어서 보여줌."""
    groups = []
    group_index = {}
    for doc in docs:
        fname = doc.metadata.get('source_file', 'CATIA Manual')
        page_num = doc.metadata.get('page', 0) + 1
        key = (fname, page_num)
        if key not in group_index:
            group_index[key] = len(groups)
            groups.append((fname, page_num, []))
        groups[group_index[key]][2].append(doc)

    for i, (fname, page_num, group_docs) in enumerate(groups):
        label = f"📄 참고 문서 {i+1} : [{fname} - Page {page_num}]"
        if len(group_docs) > 1:
            label += f" ({len(group_docs)}개 문단 병합)"
        with st.expander(label):
            for j, doc in enumerate(group_docs):
                if j > 0:
                    st.divider()
                st.text(doc.page_content)
                render_chunk_images(doc)


def main():
    st.markdown('<div class="main-header">🛠️ CATIA 멀티모달 RAG & 작업 절차 정합성 검증 시스템</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">오픈소스 소형 LLM(Qwen2.5) + 멀티모달 도면 파싱 기반 CATIA 전용 도움말 챗봇</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/catia.png", width=70)
    st.sidebar.title("📌 프로젝트 정보")

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

    n_pdfs, n_pages, n_chunks = get_corpus_stats(pipeline)
    chunk_display = f"{n_chunks:,}개" if n_chunks is not None else "알 수 없음"
    st.sidebar.info(f"""
    **KDT 14기 3팀 프로젝트**
    - **통합 멀티모달 매뉴얼**: {n_pdfs}개 PDF 문헌 ({n_pages:,}페이지 / {chunk_display} 멀티모달 청크)
    - **핵심 기술**: PyMuPDF 도면/캡처 태깅, Query Expansion, Open LLM (Qwen2.5), Local HF Stack
    """)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📖 멀티모달 질의응답 (QA)", "⚙️ 작업 절차 정합성 검증", "📊 3-Way 모델 비교 A/B 벤치마크"])

    # TAB 1: QA
    with tab1:
        st.subheader("📖 CATIA 멀티모달 매뉴얼 한국어 질의응답")
        st.write(f"**현재 구동 모델**: `{selected_model_name}` | **방식**: `{execution_mode_label}`")

        # Preset Questions (폼 밖에 둬서 선택하자마자 바로 입력창에 반영되게 함)
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

        ## => st.form으로 감싸면 입력창에 타이핑만 하고 Enter를 안 눌러도, 제출 버튼 클릭 시
        ##    한 번에 값이 커밋되어 전달됨 (타이핑 후 바로 클릭했을 때 값이 안 먹히던 문제 해결)
        with st.form("qa_form"):
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

            submitted = st.form_submit_button("질문하기 🚀")

        if submitted:
            if not user_query.strip():
                st.warning("질문을 입력해주세요.")
            else:
                t0 = time.time()
                if "Adaptive Fallback RAG" in mode:
                    with st.spinner("멀티모달 매뉴얼 검색 및 적응형 폴백 분석 중..."):
                        res = pipeline.answer_adaptive_fallback(user_query)
                    st.session_state["qa_result"] = {"type": "adaptive", "query": user_query, "res": res, "elapsed": time.time() - t0}
                elif "Strict RAG" in mode:
                    with st.spinner("멀티모달 매뉴얼 엄격 검색 및 답변 생성 중..."):
                        res = pipeline.answer_rag(user_query)
                    st.session_state["qa_result"] = {"type": "strict", "query": user_query, "res": res, "elapsed": time.time() - t0}
                else:
                    with st.spinner("Direct LLM 답변 생성 중..."):
                        ans_direct = pipeline.answer_direct(user_query)
                    st.session_state["qa_result"] = {"type": "direct", "query": user_query, "ans": ans_direct, "elapsed": time.time() - t0}

        ## => session_state에서 렌더링 -> 사이드바 Top-K 등 다른 위젯을 건드려서 스크립트가
        ##    다시 실행돼도 방금 받은 답변이 사라지지 않고 그대로 남아있음
        if "qa_result" in st.session_state:
            r = st.session_state["qa_result"]
            st.divider()

            # 1. 질문 (User Question)
            st.markdown("### ❓ 질문")
            st.info(r["query"])

            # 2. 답변 (Answer)
            st.markdown("### 🤖 답변")
            st.caption(f"⏱️ 응답 시간: {r['elapsed']:.2f}초")

            if r["type"] == "adaptive":
                res = r["res"]
                st.markdown(f'<span class="badge-rag-adaptive">Adaptive Fallback RAG ({res["status_tag"]})</span>', unsafe_allow_html=True)
                if res["used_fallback"]:
                    st.warning(res["answer"])
                else:
                    st.success(res["answer"])

                st.markdown("### 📚 참고 문서 내용 (Context Chunks)")
                st.markdown(f"**📌 참조 출처 문헌 및 페이지**: `{', '.join(res['source_pages'])}`")
                render_context_chunks(res["retrieved_docs"])

            elif r["type"] == "strict":
                res = r["res"]
                st.markdown(f'<span class="badge-rag-strict">Strict RAG (매뉴얼 엄격 전용)</span>', unsafe_allow_html=True)
                st.success(res["answer"])

                st.markdown("### 📚 참고 문서 내용 (Context Chunks)")
                st.markdown(f"**📌 참조 출처 문헌 및 페이지**: `{', '.join(res['source_pages'])}`")
                render_context_chunks(res["retrieved_docs"])

            else:
                st.markdown(f'<span class="badge-rag-off">Direct LLM (RAG Off)</span>', unsafe_allow_html=True)
                st.info(r["ans"])

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

        with st.form("proc_form"):
            user_proc_text = st.text_area(
                "검증할 작업 절차를 입력하세요:",
                value="" if preset_proc == "직접 입력" else preset_proc,
                height=120,
                placeholder="1. 2D 스케치 작성\n2. Pad 기능으로 3D 솔리드 생성\n3. Stiffener로 내부에 리브 생성"
            )
            proc_submitted = st.form_submit_button("정합성 검증 실행 🔍")

        if proc_submitted:
            if not user_proc_text.strip():
                st.warning("작업 절차를 입력해주세요.")
            else:
                t0 = time.time()
                with st.spinner("매뉴얼 표준과 정합성 비교 분석 중..."):
                    proc_res = pipeline.verify_procedure(user_proc_text)
                st.session_state["proc_result"] = {"query": user_proc_text, "res": proc_res, "elapsed": time.time() - t0}

        if "proc_result" in st.session_state:
            r = st.session_state["proc_result"]
            proc_res = r["res"]
            st.divider()
            st.markdown("### ❓ 검증 대상 작업 절차")
            st.info(r["query"])

            st.markdown("### 🤖 검증 리포트 및 피드백")
            st.caption(f"⏱️ 응답 시간: {r['elapsed']:.2f}초")
            st.markdown(proc_res["feedback"])

            st.markdown("### 📚 참고 문서 내용 (Context Chunks)")
            st.markdown(f"**📌 참조 출처 문헌 및 페이지**: `{', '.join(proc_res['source_pages'])}`")
            render_context_chunks(proc_res["retrieved_docs"])

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
