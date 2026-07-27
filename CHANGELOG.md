# Changelog

## 2026-07-27 (3)

### Fixed — 참고 이미지가 질문 내용과 무관하게 뜨던 문제
- **문제**: 멀티모달 QA 답변 시 "참고 문서" 이미지가 실제로는 질문과 무관했음. 기존엔 페이지 전체 텍스트+이미지를 통째로 하나의 청크로 묶어서, 텍스트가 여러 청크로 쪼개져도 이미지 리스트는 모든 청크에 그대로 복사됐음 (예: "동심원(Concentricity)" 질문에 "Coincidence" 등 다른 제약조건 이미지까지 같이 뜸)
- `src/load_pdf.py`: `extract_multimodal_text_from_pdf()`를 페이지 단위 → **문단(텍스트 블록) 단위**로 재작성
  - `extract_images_from_page()`: `fitz.Pixmap` 기반으로 이미지 추출 (기존 `extract_image()` 대비 SMask/알파 합성 지원)
  - `_assign_images_to_nearest_block()`: 각 이미지를 좌표(bbox)상 가장 가까운 문단에 매칭
  - `_group_blocks_into_chunks()`: 문단을 chunk_size 기준으로 그룹핑하되, 새 문단이 자기만의 이미지를 데려오면 주제 전환으로 보고 청크 분리 (슬라이드형 페이지는 전체가 chunk_size보다 작아 글자 수만으로 안 갈라지는 문제 대응)
- `src/app.py`: `render_chunk_images()`(이미지 렌더링), `render_context_chunks()`(같은 파일+페이지 청크를 하나의 expander로 병합 표시) 추가, 3개 탭 모두 적용

### Fixed — 부가 이미지 품질 버그 (같은 작업 중 발견)
- **검은 이미지**: PyMuPDF `extract_image()`가 SMask(알파 마스크)를 합성하지 않아 마스크 기반 아이콘/로고가 검은 사각형으로 렌더링됨 → `fitz.Pixmap(base, mask)` 합성 방식으로 교체
- **리사이즈 크래시**: 얇은 구분선 이미지(1px 높이 등)가 Streamlit 리사이즈 시 `height and width must be > 0` 에러 유발 → 8px 이하 이미지 추출 단계에서 제외, `app.py`에서도 이미지별 개별 try/except로 방어
- **텍스트 중복 이미지**: 일부 PDF가 제목/캡션 텍스트를 이미지로 별도 렌더링해서 겹쳐놓음 (예: "Concentricity" 캡션 옆에 "Concentricity" 글자 이미지가 따로 존재) → `_is_text_duplicate_image()`: 이미지가 실제 텍스트 블록과 10% 이상 겹치면 제외 (진짜 도면은 겹침 0%, 가짜는 12~96% 겹치는 것으로 확인됨)
- **워터마크**: 페이지 하단 배경 워터마크(예: "3D Modeling...")가 알파(불투명도) 0에 가깝게 설정되어 사실상 안 보이도록 디자인된 것 → 합성 후 평균 알파 10% 미만이면 제외
- **벡터DB 중복 저장**: `build_or_load_vectorstore(force_rebuild=True)`가 기존 Chroma 컬렉션을 안 지우고 append만 해서 같은 청크가 신/구 메타데이터로 중복 저장됨 → force_rebuild 시 기존 디렉터리 삭제하도록 `src/vector_store.py` 수정

### Changed — 로컬 GPU 사용
- `torch`/`torchvision`을 CPU 전용 빌드에서 CUDA 12.8(cu128, RTX 30/40/50시리즈 지원) GPU 빌드로 교체
- `accelerate` 추가 (`device_map` 기반 로컬 모델 GPU 로딩에 필요)
- `requirements.txt`에 `--extra-index-url https://download.pytorch.org/whl/cu128` + 버전 고정

## 2026-07-27 (2)

### Added
- `.gitignore`에 `vect/`, `vecstore_cache/`, `data/evaluation_results.csv`, `eval/results*.csv`, `Claude.md` 항목 추가

### Changed
- `requirements.txt`에 누락돼 있던 의존성(`torch`, `transformers`, `PyMuPDF`, `chromadb`, `PyPDF2`, `langchain-core` 등) 추가 — main 브랜치 대비 누락되어 실제 코드(`RAGPipeline`, `vector_store.py`)가 요구하는 패키지와 안 맞던 문제 해결

### Removed
- `vect/`(Chroma 벡터DB, 44MB+)와 `data/evaluation_results.csv`를 git 추적에서 제거 — 코드로 재생성 가능한 생성물이라 실행할 때마다 불필요한 diff가 쌓이던 문제 해결
- `Claude.md`를 git 추적에서 제거 — 개인용 작업 규칙 파일이라 팀 저장소에 공유될 필요 없음

## 2026-07-27 (1)

### Removed
- `src/main.py`, `src/evaluate.py` 삭제 — 팀원 브랜치 병합(`c56dd49`)으로 `src/rag_chain.py`가 `RAGPipeline` 클래스 기반 아키텍처로 전면 교체되면서, 기존 함수 기반 API(`get_llm`, `build_rag_chain`, `build_direct_llm_chain`)를 참조하던 두 파일이 `ImportError`로 깨진 상태가 됨. 새 진입점(`src/app.py`, `src/run_evaluation.py`)이 이를 대체하므로 정리.

### 참고
- 이전에 추가했던 RAG Off(Direct LLM) 대조군 기능은 병합된 `RAGPipeline.answer_direct()` / `answer_adaptive_fallback()`으로 이미 더 발전된 형태(쿼리 확장, adaptive fallback, 절차 검증, 출처 인용 포함)로 구현되어 있어 중복 제거됨.
