import os
import sys
import time
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from pypdf import PdfReader
from bert_score import score as compute_bert_score

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

# Custom CSS for modern dark navy theme & high contrast readability
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

    html, body, .stMarkdown, p, label, button, input, select, textarea {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
        font-size: 1.05rem;
        color: #E2E8F0;
    }

    /* Preserve Streamlit Material Icons / Symbols font family */
    span[data-testid="stIcon"], 
    [class*="material-symbols"], 
    [class*="Material"],
    [data-testid="collapsedControl"] *,
    header[data-testid="stHeader"] span,
    button[data-testid="baseButton-headerNoPadding"] * {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    /* Overall App Background - Deep Dark Navy */
    .stApp {
        background-color: #090D16 !important;
    }

    /* Fixed Top Header Bar Override */
    header[data-testid="stHeader"], [data-testid="stHeader"] {
        background-color: #090D16 !important;
        border-bottom: 1px solid #1E293B !important;
    }
    header[data-testid="stHeader"] *, [data-testid="stHeader"] * {
        color: #E2E8F0 !important;
    }

    /* Hero Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #1D4ED8 100%);
        padding: 2.3rem 2.6rem;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), 0 0 25px rgba(37, 99, 235, 0.18);
        margin-bottom: 2rem;
        border: 1px solid #2563EB;
    }
    .hero-title {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.15rem !important;
        color: #93C5FD !important;
        font-weight: 500;
        margin-top: 0.3rem;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        font-size: 1.05rem !important;
        font-weight: 600;
        color: #CBD5E1 !important;
    }
    .sidebar-info-card {
        background: #1E293B;
        border-left: 4px solid #3B82F6;
        border-radius: 12px;
        padding: 1.1rem;
        margin-top: 1rem;
        font-size: 0.98rem;
        line-height: 1.65;
        color: #CBD5E1 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .sidebar-info-card strong {
        color: #60A5FA !important;
    }

    /* Form Container Styling */
    [data-testid="stForm"] {
        background: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 18px !important;
        padding: 2rem !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 1.5rem;
    }

    /* Input & Selectbox & Radio Labels */
    .stTextInput label, .stSelectbox label, .stRadio label, .stSlider label {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #F1F5F9 !important;
        margin-bottom: 0.4rem !important;
    }

    /* Selectbox - Page-Matched Theme Colors */
    [data-testid="stSelectbox"] > div,
    div[data-baseweb="select"] {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }

    /* Inner nested elements - no border, transparent bg */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div,
    div[role="combobox"] {
        background-color: transparent !important;
        border: none !important;
    }

    /* Selected text - bright white for readability on dark bg */
    [data-testid="stSelectbox"] span,
    [data-testid="stSelectbox"] p,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] p,
    div[data-baseweb="select"] input {
        color: #FFFFFF !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 1.05rem !important;
    }

    /* Dropdown chevron arrow - subtle slate color */
    div[data-baseweb="select"] svg,
    [data-testid="stSelectbox"] svg {
        color: #94A3B8 !important;
        fill: #94A3B8 !important;
        width: 1.2rem !important;
        height: 1.2rem !important;
    }

    /* Text Input - Dark Background, White Text */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="textarea"],
    .stTextInput > div > div,
    .stTextArea > div > div {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    .stTextInput input,
    div[data-baseweb="textarea"] textarea,
    .stTextArea textarea {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 1.05rem !important;
    }

    /* Selectbox Options Dropdown Menu & Popovers */
    div[data-baseweb="popover"], 
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"], 
    div[data-baseweb="menu"] *,
    ul[data-baseweb="menu"], 
    ul[data-baseweb="menu"] *,
    div[role="listbox"],
    div[role="listbox"] * {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }

    li[data-baseweb="option"], div[role="option"] {
        background-color: #1E293B !important;
        color: #CBD5E1 !important;
        font-size: 1.05rem !important;
    }

    li[data-baseweb="option"]:hover, 
    li[aria-selected="true"], 
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }

    /* Radio Group Styling */
    div[data-aria-label="radio"], div[role="radiogroup"] {
        background-color: #1E293B !important;
        padding: 0.8rem 1.2rem !important;
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
    }
    div[role="radiogroup"] label, div[role="radiogroup"] label * {
        color: #F1F5F9 !important;
        font-size: 1.05rem !important;
    }

    /* Alert Boxes (st.info, st.success, st.warning) */
    div[data-testid="stAlert"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.4rem !important;
        font-size: 1.08rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stAlert"] * {
        color: #F8FAFC !important;
        font-size: 1.08rem !important;
    }
    /* Info Alert Accent */
    div[data-testid="stAlert"][kind="info"] {
        border-left: 5px solid #3B82F6 !important;
        background-color: #1E293B !important;
    }
    /* Success Alert Accent */
    div[data-testid="stAlert"][kind="success"] {
        border-left: 5px solid #10B981 !important;
        background-color: #064E3B !important;
    }

    /* Button Styling */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #60A5FA !important;
        border-radius: 10px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
        margin-top: 0.5rem;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(59, 130, 246, 0.6) !important;
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
    }

    /* Metrics Styling */
    [data-testid="stMetric"] {
        background: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.3rem !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
        min-height: 135px !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
    [data-testid="stMetricValue"] div {
        font-size: 2.0rem !important;
        font-weight: 800 !important;
        color: #38BDF8 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        border-bottom: 2px solid #1E293B !important;
        padding-bottom: 0px;
        margin-bottom: 1.8rem;
    }
    .stTabs [data-baseweb="tab-list"] button, .stTabs [data-baseweb="tab"] {
        border-top-left-radius: 16px !important;
        border-top-right-radius: 16px !important;
        border-bottom-left-radius: 0px !important;
        border-bottom-right-radius: 0px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 1.7rem !important;
        color: #94A3B8 !important;
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-bottom: none !important;
    }
    .stTabs [aria-selected="true"] {
        border-top-left-radius: 16px !important;
        border-top-right-radius: 16px !important;
        border-bottom-left-radius: 0px !important;
        border-bottom-right-radius: 0px !important;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        border: 1px solid #3B82F6 !important;
        border-bottom: none !important;
        box-shadow: 0 -4px 14px rgba(37, 99, 235, 0.25) !important;
    }

    /* Expander Styling */
    [data-testid="stExpander"] {
        background: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
        margin-bottom: 0.9rem !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #1E293B !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }
    [data-testid="stExpander"] * {
        color: #E2E8F0 !important;
    }

    /* Badges */
    .badge-rag-strict {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF !important;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        box-shadow: 0 3px 10px rgba(16, 185, 129, 0.4);
        margin-bottom: 0.7rem;
    }
    .badge-rag-off {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: #FFFFFF !important;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        box-shadow: 0 3px 10px rgba(239, 68, 68, 0.4);
        margin-bottom: 0.7rem;
    }

    /* Dataframe in Tab 2 */
    [data-testid="stDataFrame"] {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
    }
    [data-testid="stDataFrame"] * {
        color: #E2E8F0 !important;
    }

    /* Code Blocks */
    pre, code, div[data-testid="stText"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Headings font sizes & colors */
    h1 { font-size: 2.2rem !important; font-weight: 800 !important; color: #FFFFFF !important; }
    h2 { font-size: 1.75rem !important; font-weight: 750 !important; color: #F8FAFC !important; }
    h3 { font-size: 1.45rem !important; font-weight: 700 !important; color: #F1F5F9 !important; margin-top: 1rem !important; }
    h4 { font-size: 1.25rem !important; font-weight: 650 !important; color: #E2E8F0 !important; }
    h5 { font-size: 1.15rem !important; font-weight: 650 !important; color: #CBD5E1 !important; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_rag_pipeline(model_name: str, mode: str):
    """Cache Multimodal RAG Pipeline initialization by model name and execution mode."""
    return RAGPipeline(model_name=model_name, mode=mode)


@st.cache_data
def load_auto_questions_df():
    """Load CATIA_RAG_High_Gap_Auto_Questions.csv (38 questions) dataset."""
    csv_paths = [
        PROJECT_ROOT / "eval" / "CATIA_RAG_High_Gap_Auto_Questions.csv",
        PROJECT_ROOT / "eval" / "CATIA_Solar_Gap_2_Plus_Qwen_BGE_M3_Results.csv",
        PROJECT_ROOT / "eval" / "CATIA_Solar_Gap_2_Plus_Evaluation_Results.csv"
    ]
    for path in csv_paths:
        if path.exists():
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
                return df
            except Exception:
                try:
                    df = pd.read_csv(path, encoding="utf-8")
                    return df
                except Exception:
                    pass
    return None


@st.cache_resource
def get_corpus_stats(_pipeline):
    """PDF/페이지/청크 수를 동적으로 계산."""
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


def calculate_single_bertscore(candidate: str, reference: str) -> dict:
    """Calculate BERTScore (Precision, Recall, F1) for a single candidate answer against reference."""
    if not candidate or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        P, R, F1 = compute_bert_score(cands=[str(candidate)], refs=[str(reference)], lang="ko", verbose=False)
        return {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F1[0])
        }
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def render_chunk_images(doc):
    """이 청크의 문단에 배정된 CAD 도면/스크린샷 이미지를 화면에 표시한다."""
    raw_paths = doc.metadata.get("image_paths", "")
    if not raw_paths:
        return
    image_paths = [p for p in raw_paths.split(";") if p and os.path.exists(p)]
    for path in image_paths:
        try:
            st.image(path, width=300)
        except Exception:
            continue


def render_context_chunks(docs):
    """검색된 청크를 (파일명, 페이지) 기준으로 묶어서 표시한다."""
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
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🛠️ CATIA 멀티모달 RAG & 성능 비교 평가 시스템</div>
        <div class="hero-subtitle">Qwen 2.5 오픈소스 LLM (0.5B / 1.5B / 3B) + BGE-M3 멀티모달 매뉴얼 RAG & 정량/정성 비교 평가</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Header
    st.sidebar.markdown('<h2 style="color: #FFFFFF; font-size: 1.5rem; font-weight: 800; margin-bottom: 1.2rem;">🛠️ 프로젝트 설정</h2>', unsafe_allow_html=True)

    execution_mode_label = st.sidebar.radio(
        "⚙️ 모델 구동 방식 선택:",
        [
            "🏠 로컬 다운로드 구동 (Local HF PyTorch)",
            "☁️ 클라우드 API 구동 (HuggingFace Serverless API)"
        ]
    )
    execution_mode = "local" if "로컬" in execution_mode_label else "api"

    # Qwen 2.5 브랜드 고정 & 파라미터별 선택 (0.5B, 1.5B, 3B)
    selected_param_label = st.sidebar.selectbox(
        "🤖 Qwen 2.5 모델 파라미터 선택:",
        [
            "Qwen 2.5 0.5B (기본 / 초경량 0.5B)",
            "Qwen 2.5 1.5B (1.5B)",
            "Qwen 2.5 3B (권장 3B)",
            "Qwen 2.5 7B (7B, API 크레딧 필요)",
            "Llama 3.1 8B (8B, API 크레딧 필요)"
        ]
    )

    param_model_map = {
        "Qwen 2.5 0.5B (기본 / 초경량 0.5B)": "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen 2.5 1.5B (1.5B)": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen 2.5 3B (권장 3B)": "Qwen/Qwen2.5-3B-Instruct",
        "Qwen 2.5 7B (7B, API 크레딧 필요)": "Qwen/Qwen2.5-7B-Instruct",
        "Llama 3.1 8B (8B, API 크레딧 필요)": "meta-llama/Llama-3.1-8B-Instruct"
    }
    selected_model_name = param_model_map[selected_param_label]
    
    top_k = st.sidebar.slider("검색 문서 수 (Top-K)", min_value=1, max_value=10, value=3)

    # Load Pipeline
    with st.spinner(f"[{selected_param_label}] 파이프라인 및 벡터 DB 로딩 중..."):
        pipeline = load_rag_pipeline(selected_model_name, execution_mode)
        pipeline.retriever.search_kwargs["k"] = top_k

    n_pdfs, n_pages, n_chunks = get_corpus_stats(pipeline)
    chunk_display = f"{n_chunks:,}개" if n_chunks is not None else "알 수 없음"
    st.sidebar.markdown(f"""
    <div class="sidebar-info-card">
        <strong style="font-size: 1.05rem; color: #1E3A8A;">KDT 14기 3팀 프로젝트</strong><br>
        • <strong>베이스라인 모델</strong>: <code>{selected_model_name}</code><br>
        • <strong>통합 멀티모달 매뉴얼</strong>: {n_pdfs}개 PDF ({n_pages:,}p / {chunk_display} 청크)<br>
        • <strong>핵심 기술</strong>: PyMuPDF 도면/캡처 태깅, BGE-M3 Vector Store, Open LLM (Qwen 2.5), BERTScore & Solar Evaluation
    </div>
    """, unsafe_allow_html=True)

    # Load Auto Questions Dataset CSV (38 questions)
    df_auto = load_auto_questions_df()

    # Tabs (Only 2 Tabs: QA & Benchmark Dataset)
    tab1, tab2 = st.tabs(["📖 RAG vs Direct LLM 답변 & 평가 (QA)", "📊 CATIA High-Gap 벤치마크 데이터셋"])

    # TAB 1: QA & Evaluation
    with tab1:
        st.subheader("📖 RAG 적용 LLM vs 순수 LLM 답변 생성 및 BERTScore/정성 평가 대조")
        st.write(f"**현재 구동 모델**: `{selected_model_name}` | **방식**: `{execution_mode_label}`")

        # Preset Questions from CATIA_RAG_High_Gap_Auto_Questions.csv (38 items)
        preset_options = ["직접 입력"]
        qa_dict_by_q = {}
        if df_auto is not None and "question" in df_auto.columns:
            for idx, row in df_auto.iterrows():
                q_text = str(row["question"]).strip()
                ref_text = str(row.get("reference_answer", "")).strip()
                qid = str(row.get("id", f"Q{idx+1:02d}"))
                opt_label = f"[{qid}] {q_text[:50]}..." if len(q_text) > 50 else f"[{qid}] {q_text}"
                preset_options.append(opt_label)
                qa_dict_by_q[opt_label] = {
                    "question": q_text,
                    "reference_answer": ref_text,
                    "row": row
                }

        selected_preset = st.selectbox(
            "💡 벤치마크 고격차 평가 질문 선택 (CATIA_RAG_High_Gap_Auto_Questions.csv 전체 38개 문항):",
            preset_options
        )

        default_q_val = ""
        default_ref_val = ""
        if selected_preset != "직접 입력" and selected_preset in qa_dict_by_q:
            default_q_val = qa_dict_by_q[selected_preset]["question"]
            default_ref_val = qa_dict_by_q[selected_preset]["reference_answer"]

        with st.form("qa_form"):
            user_query = st.text_input(
                "질문을 입력하세요:",
                value=default_q_val,
                placeholder="예: CATIA V5에서 Specification Tree를 숨기거나 다시 표시하는 토글 단축키는?"
            )

            ref_answer = st.text_input(
                "표준 정답 (Ground Truth / Reference Answer):",
                value=default_ref_val,
                placeholder="BERTScore 및 정성 평가 기준 정답 (예: F3)"
            )

            mode_option = st.radio(
                "실행 모델 모드 선택 (2가지):",
                [
                    "⚔️ RAG 적용 LLM vs 순수 LLM 1:1 대조 및 평가 모드 (추천)",
                    "🎯 RAG 적용 LLM 전용 (문서 검색 기반)",
                    "⚡ 순수 LLM 전용 (Direct LLM / RAG Off)"
                ],
                horizontal=False
            )

            submitted = st.form_submit_button("답변 생성 및 평가 실행 🚀")

        if submitted:
            if not user_query.strip():
                st.warning("질문을 입력해주세요.")
            else:
                t0 = time.time()
                if "1:1 대조" in mode_option:
                    with st.spinner("RAG 적용 LLM 및 순수 LLM 답변 생성 중..."):
                        ans_direct = pipeline.answer_direct(user_query)
                        t_dir = time.time() - t0
                        
                        t1 = time.time()
                        res_rag = pipeline.answer_rag(user_query)
                        t_rag = time.time() - t1

                        st.session_state["qa_result"] = {
                            "type": "compare",
                            "query": user_query,
                            "ref_answer": ref_answer,
                            "ans_direct": ans_direct,
                            "res_rag": res_rag,
                            "t_dir": t_dir,
                            "t_rag": t_rag
                        }
                elif "RAG 적용" in mode_option:
                    with st.spinner("RAG 적용 LLM 답변 생성 중..."):
                        res_rag = pipeline.answer_rag(user_query)
                        t_rag = time.time() - t0
                        st.session_state["qa_result"] = {
                            "type": "rag_only",
                            "query": user_query,
                            "ref_answer": ref_answer,
                            "res_rag": res_rag,
                            "t_rag": t_rag
                        }
                else:
                    with st.spinner("순수 LLM (Direct) 답변 생성 중..."):
                        ans_direct = pipeline.answer_direct(user_query)
                        t_dir = time.time() - t0
                        st.session_state["qa_result"] = {
                            "type": "direct_only",
                            "query": user_query,
                            "ref_answer": ref_answer,
                            "ans_direct": ans_direct,
                            "t_dir": t_dir
                        }

        if "qa_result" in st.session_state:
            r = st.session_state["qa_result"]
            st.divider()

            # 1. 질문 & 표준 정답
            st.markdown("### ❓ 질문 및 표준 정답 (Ground Truth)")
            st.info(f"**질문**: {r['query']}")
            if r["ref_answer"]:
                st.success(f"**표준 정답 (Ground Truth)**: `{r['ref_answer']}`")

            # 2. 답변 대조 및 지표 출력 (높이 완벽 정렬: min-height 125px 고정!)
            st.markdown("### 🤖 모델 답변 및 평가 지표 (Evaluation Metrics)")

            if r["type"] == "compare":
                m_dir = calculate_single_bertscore(r["ans_direct"], r["ref_answer"]) if r["ref_answer"] else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                m_rag = calculate_single_bertscore(r["res_rag"]["answer"], r["ref_answer"]) if r["ref_answer"] else {"precision": 0.0, "recall": 0.0, "f1": 0.0}

                col_dir, col_rag = st.columns(2)

                # Direct LLM Column (Left)
                with col_dir:
                    st.markdown('<span class="badge-rag-off">⚡ 순수 LLM (Direct LLM / RAG Off)</span>', unsafe_allow_html=True)
                    st.caption(f"⏱️ 응답 시간: {r['t_dir']:.2f}초")

                    if r["ref_answer"]:
                        st.markdown("#### 📊 순수 LLM 평가 지표")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("BERT F1", f"{m_dir['f1']:.4f}")
                        m2.metric("Precision", f"{m_dir['precision']:.4f}")
                        m3.metric("Recall", f"{m_dir['recall']:.4f}")

                    st.markdown("#### 📝 생성 답변")
                    st.info(r["ans_direct"])

                # RAG LLM Column (Right)
                with col_rag:
                    st.markdown('<span class="badge-rag-strict">🎯 RAG 적용 LLM (문서 검색 기반)</span>', unsafe_allow_html=True)
                    st.caption(f"⏱️ 응답 시간: {r['t_rag']:.2f}초")

                    if r["ref_answer"]:
                        delta_f1 = m_rag['f1'] - m_dir['f1']
                        delta_p = m_rag['precision'] - m_dir['precision']
                        delta_r = m_rag['recall'] - m_dir['recall']

                        st.markdown("#### 📊 RAG 적용 LLM 평가 지표")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("BERT F1", f"{m_rag['f1']:.4f}", delta=f"{delta_f1:+.4f}")
                        m2.metric("Precision", f"{m_rag['precision']:.4f}", delta=f"{delta_p:+.4f}")
                        m3.metric("Recall", f"{m_rag['recall']:.4f}", delta=f"{delta_r:+.4f}")

                    st.markdown("#### 📝 생성 답변")
                    st.success(r["res_rag"]["answer"])

                st.markdown("### 📚 RAG 참조 출처 및 검색 문맥 (Context Chunks)")
                st.markdown(f"**📌 출처 페이지**: `{', '.join(r['res_rag']['source_pages'])}`")
                render_context_chunks(r["res_rag"]["retrieved_docs"])

            elif r["type"] == "rag_only":
                st.markdown('<span class="badge-rag-strict">🎯 RAG 적용 LLM (문서 검색 기반)</span>', unsafe_allow_html=True)
                st.caption(f"⏱️ 응답 시간: {r['t_rag']:.2f}초")

                if r["ref_answer"]:
                    m_rag = calculate_single_bertscore(r["res_rag"]["answer"], r["ref_answer"])
                    st.markdown("#### 📊 RAG 적용 LLM 평가 지표")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("BERT F1", f"{m_rag['f1']:.4f}")
                    m2.metric("Precision", f"{m_rag['precision']:.4f}")
                    m3.metric("Recall", f"{m_rag['recall']:.4f}")

                st.markdown("#### 📝 생성 답변")
                st.success(r["res_rag"]["answer"])

                st.markdown("### 📚 RAG 참조 출처 및 검색 문맥 (Context Chunks)")
                st.markdown(f"**📌 출처 페이지**: `{', '.join(r['res_rag']['source_pages'])}`")
                render_context_chunks(r["res_rag"]["retrieved_docs"])

            else:
                st.markdown('<span class="badge-rag-off">⚡ 순수 LLM (Direct LLM / RAG Off)</span>', unsafe_allow_html=True)
                st.caption(f"⏱️ 응답 시간: {r['t_dir']:.2f}초")

                if r["ref_answer"]:
                    m_dir = calculate_single_bertscore(r["ans_direct"], r["ref_answer"])
                    st.markdown("#### 📊 순수 LLM 평가 지표")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("BERT F1", f"{m_dir['f1']:.4f}")
                    m2.metric("Precision", f"{m_dir['precision']:.4f}")
                    m3.metric("Recall", f"{m_dir['recall']:.4f}")

                st.markdown("#### 📝 생성 답변")
                st.info(r["ans_direct"])

    # TAB 2: CATIA High-Gap 벤치마크 데이터셋 (CATIA_RAG_High_Gap_Auto_Questions.csv 전체 38개 문항)
    with tab2:
        st.subheader("📊 CATIA High-Gap 전체 벤치마크 데이터셋 & 종합 평가 지표")
        
        if df_auto is not None:
            st.write(f"**전체 평가 데이터셋 (CATIA_RAG_High_Gap_Auto_Questions.csv)**: 총 `{len(df_auto)}개 문항`")
            
            # Overall Dataset Metrics Calculation
            solar_dir_mean = df_auto["solar_direct_score"].mean() if "solar_direct_score" in df_auto.columns else 0.0
            solar_rag_mean = df_auto["solar_rag_score"].mean() if "solar_rag_score" in df_auto.columns else 0.0
            solar_gap_mean = df_auto["solar_score_gap"].mean() if "solar_score_gap" in df_auto.columns else (solar_rag_mean - solar_dir_mean)

            bert_dir_mean = df_auto["direct_f1"].mean() if "direct_f1" in df_auto.columns else 0.0
            bert_rag_mean = df_auto["rag_f1"].mean() if "rag_f1" in df_auto.columns else 0.0
            bert_gap_mean = df_auto["bert_f1_gap"].mean() if "bert_f1_gap" in df_auto.columns else (bert_rag_mean - bert_dir_mean)

            st.markdown("#### 📈 전체 데이터셋 (38문항) 종합 평가지표 요약")
            
            # Row 1: Upstage Solar LLM-as-a-Judge (1~5점)
            st.markdown("##### ⭐ Upstage Solar Pro (1~5점 정성 평가)")
            s1, s2, s3 = st.columns(3)
            s1.metric("Direct LLM Solar 평균", f"{solar_dir_mean:.2f} / 5.0")
            s2.metric("RAG LLM Solar 평균", f"{solar_rag_mean:.2f} / 5.0", delta=f"{solar_gap_mean:+.2f}점")
            s3.metric("Solar Score Improvement", f"{solar_gap_mean:+.2f}점")

            # Row 2: BERTScore F1
            st.markdown("##### 📊 BERTScore F1 (문장 의미 유사도)")
            b1, b2, b3 = st.columns(3)
            b1.metric("Direct LLM BERT F1", f"{bert_dir_mean:.4f}")
            b2.metric("RAG LLM BERT F1", f"{bert_rag_mean:.4f}", delta=f"{bert_gap_mean:+.4f}")
            b3.metric("BERT F1 Improvement", f"{bert_gap_mean:+.4f}")

            st.divider()
            st.markdown("#### 📋 벤치마크 전체 38개 문항 데이터셋 표")
            st.dataframe(df_auto)
        else:
            st.info("CATIA_RAG_High_Gap_Auto_Questions.csv 데이터셋 파일이 존재하지 않습니다.")


if __name__ == "__main__":
    main()
