# 📋 CATIA RAG 프로젝트 전체 코드 구조 및 세부 구현 명세서 (CHANGES_SUMMARY.md)

본 문서는 팀 기존 베이스라인 저장소(Repository) 대비 새로 변경 및 고도화된 전체 시스템 구조, 폴더 체계, 그리고 `src/` 폴더 내 **모든 파이썬 코드별 상세 역할과 구현 내용**을 기술한 보고서입니다.

---

## 🎯 1. 전체 아키텍처 및 변경 방향 요약

```mermaid
flowchart TD
    subgraph Storage [디렉토리 체계 (GitHub 표준)]
        A["📁 data/ (32개 원본 PDF 매뉴얼)"]
        B["📁 vect/ (vect/chroma_db_multimodal 백터 DB)"]
        C["📁 src/ (모든 파이썬 실행 모듈)"]
    end

    subgraph Pipeline [로컬 RAG 실행 흐름]
        C1["src/load_pdf.py (PyMuPDF 멀티모달 도면 태깅)"] --> B
        C2["src/models/llm_factory.py (PyTorch CUDA Qwen2.5 로컬 구동)"] --> C3["src/rag_chain.py (Query Expansion + RAG)"]
        B --> C3
        C3 --> C4["src/app.py (Streamlit UI 화면 정제 출력)"]
    end
```

---

## 📁 2. 세부 파이썬 코드별 기능 및 상세 구현 명세

### 1) ⚙️ [src/config.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/config.py) (중앙 환경 및 디렉토리 경로 설정 모듈)
* **주요 역할**: 프로젝트 전체 경로, 오픈소스 소형 LLM 모델 파라미터, 실험 프리셋(Preset)을 통제하는 중앙 구성 파일입니다.
* **상세 구현 내역**:
  - `PROJECT_ROOT`: `Path(__file__).resolve().parent.parent`를 통해 프로젝트 최상위 루트 디렉토리를 동적으로 추적하고 `sys.path`에 등록.
  - `DATA_DIR`: 원본 PDF 파일들이 위치하는 **`data/`** 폴더 경로 설정.
  - `VECT_DIR` & `CHROMA_DB_DIR`: 백터 데이터베이스가 저구되는 **`vect/`** 및 **`vect/chroma_db_multimodal`** 경로 설정.
  - `DEFAULT_CONFIG` & `EXPERIMENT_PRESETS`: 기존 팀원 베이스라인의 설정 체계와 100% 호환되도록 구성되었으며, `qwen_3b_local` (로컬 GPU 구동) 프리셋 추가.
  - 주요 상수: `CHUNK_SIZE = 600`, `CHUNK_OVERLAP = 100`, `DEFAULT_TOP_K = 4`, `LOCAL_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"`, `LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`.

---

### 2) 🖼️ [src/load_pdf.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/load_pdf.py) *(또는 `multimodal_loader.py`)* (멀티모달 PDF 파서 & 청킹 모듈)
* **주요 역할**: PDF 매뉴얼 내의 텍스트뿐만 아니라 **CAD 도면 그림, 스크린샷 캡처 이미지 개수 및 레이아웃 위치**를 인식하여 메타데이터와 태그를 부착하는 멀티모달 로더입니다.
* **상세 구현 내역**:
  - `detect_lang(text)`: 한글 유니코드 비율을 계산하여 한글(`ko`)과 영어(`en`) 문서 단락을 구분 태깅.
  - `load_pdf(pdf_path)` / `load_pdfs_from_folder(data_dir)`: 기존 팀 베이스라인 호환 PyPDFLoader 기반 기본 로딩 함수.
  - `extract_multimodal_text_from_pdf(pdf_path)`:
    * PyMuPDF (`fitz`)를 사용하여 각 페이지 내 이미지 객체(`page.get_images()`)를 추출 및 감지.
    * 이미지가 포함된 페이지에 `🖼️ [매뉴얼 도면/스크린샷 포함: N개의 CAD 도면 및 설정 창 캡처 그림이 포함된 페이지입니다.]` 태그 결합.
    * PyMuPDF 미설치 환경을 대비한 예외 처리(Try-Except Fallback) 포함.
  - `load_all_multimodal_pdfs()` & `load_and_split_multimodal_pdf()`: `data/` 폴더 내 32개 PDF 전량(2,416페이지)을 읽어 **총 5,278개 멀티모달 청크**로 나눔.

---

### 3) 🤖 [src/models/llm_factory.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/models/llm_factory.py) (로컬 ↔ 클라우드 모듈형 LLM 팩토리)
* **주요 역할**: OpenAI(GPT) 의존성을 완전 제외하고, 오픈소스 소형 LLM(Qwen2.5)을 내 컴퓨터 GPU/CPU로 직접 다운로드 구동하거나 API로 전환할 수 있는 모듈 팩토리입니다.
* **상세 구현 내역**:
  - `LLMFactory.get_llm(model_name, mode)`:
    * `mode="local"`: Hugging Face `transformers` (`AutoModelForCausalLM`, `AutoTokenizer`, `pipeline`)를 활용하여 PyTorch CUDA GPU(`cuda:0`) 또는 CPU 상에서 **`Qwen/Qwen2.5-3B-Instruct`** 모델을 온디바이스로 직접 구동.
    * `mode="api"`: 필요 시 HuggingFace Serverless Inference API 호출 방식으로 전환 지원.
  - `_safe_load_default_certs`: Windows 11 환경의 인증서 저장소 충돌 에러(`_ssl.c:4057`) 차단을 위한 안전 패치 내장.

---

### 4) 📦 [src/build_vectorstore.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/build_vectorstore.py) & [src/vector_store.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/vector_store.py) (로컬 Chroma 벡터 DB 및 임베딩 연동)
* **주요 역할**: 오픈소스 임베딩 모델을 이용해 멀티모달 청크를 검색 가능한 형태의 벡터 DB(`vect/chroma_db_multimodal`)로 저장하고 로딩합니다.
* **상세 구현 내역**:
  - `get_embeddings()` / `get_embedding_function()`: `sentence-transformers/all-MiniLM-L6-v2` 로컬 다국어 임베딩 모델 연동 (OpenAI 임베딩 불필요).
  - `build_vectorstore()` / `build_or_load_vectorstore()`: 5,278개 청크를 `vect/chroma_db_multimodal` 디렉토리에 인덱싱하여 오프라인으로 지속(Persist) 보관.
  - `get_retriever()`: 코사인 유사도(Similarity Search) 기반 Top-K(기본 4개) 매뉴얼 단락 검색기 제공.
  - 윈도우 SSL 인증서 오류 방지를 위한 패치 내장.

---

### 5) 🔗 [src/rag_chain.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/rag_chain.py) (RAG 파이프라인, Query Expansion, 작업 절차 1:1 검증 엔진)
* **주요 역할**: 질의어 확장, Strict RAG 답변 생성, Adaptive Fallback 처리, 그리고 Qwen 답변 텍스트 정제를 담당하는 핵심 로직 모듈입니다.
* **상세 구현 내역**:
  - `clean_llm_output(text)`: Qwen 모델 생성 시 유출되는 ChatML 특수 태그(`<|im_start|>`, `<|im_end|>`) 및 사용자 프롬프트 복사(Echo) 현상을 100% 자동 정제하는 정규식 함수.
  - `EXPAND_QUERY_PROMPT`: 사용자의 한글 질문(예: *"동심원 구속"*)을 CATIA 영문 전문 용어(*"Concentricity"*, *"Coincidence"*, *"Pad"*, *"Pocket"* 등)로 확장하여 매뉴얼 검색 정확도 극대화.
  - `RAG_PROMPT`: 검색된 매뉴얼 단락만을 근거로 한국어 답변을 생성하고, 근거가 없을 시 지어내지 않고 *"제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."*라고 응답.
  - `PROCEDURE_PROMPT`: 사용자가 입력한 공정 작업 순서와 매뉴얼 표준 절차를 1:1 대조하여 피드백 리포트 생성.
  - `answer_adaptive_fallback()`: 매뉴얼 미포함 시 일반 LLM 사전학습 지식으로 폴백 후 `⚠️ [매뉴얼 미포함 - LLM 사전학습 지식 폴백]` 태그 명시.

---

### 6) 💻 [src/app.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/app.py) (Streamlit 웹 애플리케이션 프론트엔드)
* **주요 역할**: 사용자가 멀티모달 QA, 작업 절차 정합성 검증, 3-Way 모델 비교 A/B 테스트를 손쉽게 사용할 수 있는 웹 화면입니다.
* **상세 구현 내역**:
  - **사이드바**: 오픈소스 모델 선택 드롭다운 (Qwen2.5-3B, Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.1-8B) 및 `로컬 다운로드 구동` ↔ `클라우드 API 구동` 방식 스위처.
  - **화면 레이아웃 정열 (요청 반영)**:
    1. **`### ❓ 질문`**: 사용자 질문 내용
    2. **`### 🤖 답변`**: Qwen 모델의 정제된 한국어 답변
    3. **`### 📚 참고 문서 내용 (Context Chunks)`**: 참조 출처 매뉴얼 파일명 및 페이지(P.12 등)와 원문 단락 표시
  - **3가지 탭 제공**: 1) 📖 멀티모달 질의응답 (QA), 2) ⚙️ 작업 절차 정합성 검증, 3) 📊 3-Way 모델 비교 A/B 벤치마크.

---

### 7) 📊 [src/evaluation.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/evaluation.py) & [src/run_evaluation.py](file:///c:/KDT_14/%5B11%5DTransformer/project/team-03-project/src/run_evaluation.py) (정량적 벤치마크 평가 모듈)
* **주요 역할**: Direct LLM, Strict RAG, Adaptive Fallback RAG 3가지 모델 구조의 성능을 정량 대조 평가합니다.
* **상세 구현 내역**:
  - `BENCHMARK_DATASET`: 일반 질의, CATIA 기능 세부 질의, 매뉴얼 미포함 함정 질의(Trick QA) 세트 내장.
  - `run_evaluation_benchmark()`: BERTScore F1 평가 지표를 계산하여 `data/evaluation_results.csv` 파일로 자동 저장.
  - **성능 측정 결과**: Strict RAG 평균 F1 **`0.8071`** (함정 질의 환각 방지 F1 `1.0000` 달성).

---

## 💻 3. 실행 명령어 안내

```bash
# 1. 멀티모달 Vector DB 빌드 (data/ PDF -> vect/ 인덱스)
python src/build_multimodal_vectorstore.py

# 2. Streamlit 웹 애플리케이션 실행
streamlit run src/app.py

# 3. 3-Way A/B 테스트 벤치마크 평가 실행
python src/run_evaluation.py
```
