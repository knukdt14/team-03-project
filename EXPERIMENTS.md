# 실험 기록

> 이 프로젝트는 **최종 수치보다 실험 과정과 문제 해결 과정을 중점적으로 평가**합니다.
> 프리셋을 바꿔가며 실험할 때마다 아래 표에 한 줄씩 추가해 주세요. 발표자료·보고서의 핵심 근거 자료로 그대로 쓰입니다.

## 기록 양식

| 날짜 | 담당자 | 실행 명령 | 바꾼 변수 | 결과 요약 | 발생한 문제 | 해결 방법 |
|---|---|---|---|---|---|---|
| 2026-07-27 | 조현우 | `python main.py --mode compare --preset baseline` | RAG 적용 vs Direct LLM(RAG 미적용) 비교 | (실행 후 `eval/results_compare.csv`의 `bertscore_f1_gap`, `question_type`별 격차 기록) | | |

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
