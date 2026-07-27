# Changelog

## 2026-07-27 (2)

### Removed
- `src/main.py`, `src/evaluate.py` 삭제 — 팀원 브랜치 병합(`c56dd49`)으로 `src/rag_chain.py`가 `RAGPipeline` 클래스 기반 아키텍처로 전면 교체되면서, 기존 함수 기반 API(`get_llm`, `build_rag_chain`, `build_direct_llm_chain`)를 참조하던 두 파일이 `ImportError`로 깨진 상태가 됨. 새 진입점(`src/app.py`, `src/run_evaluation.py`)이 이를 대체하므로 정리.

### 참고
- 이전에 추가했던 RAG Off(Direct LLM) 대조군 기능은 병합된 `RAGPipeline.answer_direct()` / `answer_adaptive_fallback()`으로 이미 더 발전된 형태(쿼리 확장, adaptive fallback, 절차 검증, 출처 인용 포함)로 구현되어 있어 중복 제거됨.

## 2026-07-27 (1)

### Added
- RAG Off (Direct LLM) 모드 추가 — 검색 없이 LLM 사전지식만으로 답변하는 대조군 체인 (이후 팀원 병합으로 대체/제거됨, 위 항목 참고)
