# Changelog

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
