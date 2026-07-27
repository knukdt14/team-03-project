# 실험 기록

> 이 프로젝트는 **최종 수치보다 실험 과정과 문제 해결 과정을 중점적으로 평가**합니다.
> 프리셋을 바꿔가며 실험할 때마다 아래 표에 한 줄씩 추가해 주세요. 발표자료·보고서의 핵심 근거 자료로 그대로 쓰입니다.

## 기록 양식

| 날짜 | 담당자 | 실행 명령 | 바꾼 변수 | 결과 요약 | 발생한 문제 | 해결 방법 |
|---|---|---|---|---|---|---|
| 2026-07-27 | 조현우 | `python main.py --mode compare --preset baseline` | RAG 적용 vs Direct LLM(RAG 미적용) 비교 | (실행 후 `eval/results_compare.csv`의 `bertscore_f1_gap`, `question_type`별 격차 기록) | | |
| 2026-07-27 | 조현우 | `python main.py --preset baseline` (chat 모드로 단일 질문 테스트) | 최초 엔드투엔드 실행 검증 | "솔리드 파트는 어떤 확장자로 저장되나요?"에 "문서에서 답을 찾을 수 없습니다" 오답. 정답 문장("Parts are stored in files with the extension .CATPart")은 `CATIA V5 Lectures.pdf` 13페이지에 실제로 존재함을 확인 → **데이터 문제가 아니라 검색(retrieval) 품질 문제** | 1) `pdf_path`가 상대경로("data")라 `src/`에서 실행 시 실 데이터 폴더(루트의 data/)를 못 찾음 → FileNotFoundError. 2) Windows 콘솔 기본 인코딩(cp949)에서 PDF/LLM 출력에 유니코드 특수문자 섞이면 print()가 UnicodeEncodeError로 죽음. 3) `top_k=4`(baseline), `top_k=6`(top_k_high), `mmr_search` 모두 정답 청크를 못 찾음(한국어 질문 ↔ 영어 문서 교차언어 검색 약함으로 추정). 4) `bge-m3` 임베딩으로 교차언어 검색 개선을 시도했으나 `torch 2.5.1` 환경에서 `transformers`가 안전 취약점(CVE-2025-32434)으로 로드 거부(`torch>=2.6` 요구) | 1), 2)는 코드 수정으로 해결(`config.py`에서 절대경로 계산, `main.py`에서 stdout UTF-8 재설정). 3), 4)는 미해결 — **다음 시도 후보**: chunk_size를 줄여 해당 문장이 온전한 청크로 분리되게 하기, top_k를 더 크게(8~10), 쿼리 자체를 영어로 번역해서 검색, 또는 torch를 2.6+로 올릴 수 있는 별도 가상환경에서 bge-m3 재시도 |

## 사용 방법

- **프리셋 비교(RAG 설정값 변경)**: `python main.py --preset <프리셋명>` → `eval/results.csv` 생성 → BERTScore/RAGAS 비교
- **RAG 유무 비교(교수님 피드백 대응)**: `python main.py --mode compare --preset <프리셋명>` → `eval/results_compare.csv` 생성
  - `rag_answer` vs `direct_llm_answer`를 나란히 비교
  - `bertscore_f1_gap`(RAG F1 − Direct F1)이 클수록 RAG 효과가 큰 질문
  - `question_type == trick`인 행에서 `direct_llm_answer`가 그럴듯하게 지어냈는데 `rag_answer`는 "문서에서 답을 찾을 수 없습니다"로 응답했다면, 이게 바로 환각 저감의 직접 증거
  - `rag_hallucination_flag` / `direct_hallucination_flag` / `reviewer_note` 칸은 직접 보고 O/X + 메모로 채워주세요

## 각자 남길 것

- 어떤 프리셋을 시도했는지
- 예상과 다르게 나온 부분 (예: 특정 임베딩에서 유독 느림, 특정 프롬프트에서 환각이 오히려 늘어남 등)
- 원인 추정과 해결/우회 방법 (예: RAGAS import 충돌 → `--no_ragas`로 진행)
