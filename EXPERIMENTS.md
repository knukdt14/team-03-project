# 실험 기록

> 이 프로젝트는 **최종 수치보다 실험 과정과 문제 해결 과정을 중점적으로 평가**합니다.
> 프리셋을 바꿔가며 실험할 때마다 아래 표에 한 줄씩 추가해 주세요. 발표자료·보고서의 핵심 근거 자료로 그대로 쓰입니다.

## 기록 양식

| 날짜 | 담당자 | 실행 명령 | 바꾼 변수 | 결과 요약 | 발생한 문제 | 해결 방법 |
|---|---|---|---|---|---|---|
| 2026-07-27 | 조현우 | `python main.py --mode compare --preset baseline` | RAG 적용 vs Direct LLM(RAG 미적용) 비교 | (실행 후 `eval/results_compare.csv`의 `bertscore_f1_gap`, `question_type`별 격차 기록) | | |
| 2026-07-27 | 조현우 | `python -m src.build_markdown_vectorstore` (신규 `loader="markdown"` 옵션, `pymupdf4llm` 기반) | 기존 `multimodal` 로더(순수 텍스트 + "🖼️ N개 이미지 포함" 태그) → PDF를 제목(#/##/###)·표·리스트 구조를 보존한 Markdown으로 추출 후, `MarkdownHeaderTextSplitter`로 헤더 경계부터 나누고 `chunk_size`로 재분할 | 32개 PDF, 2402페이지 → 5245청크로 `vect/chroma_db_markdown`에 별도 저장(기존 `chroma_db_multimodal`과 나란히 비교 가능). 영어 쿼리("What file extension are parts stored in?")로 테스트 시 마크다운 버전은 **top-1으로 정답 청크 히트**, 기존 버전은 이미지 태그만 있는 사실상 빈 페이지("🖼️ 4개 CAD 도면 포함...")가 최상위로 잡히는 노이즈 확인 | 1) 기존 `multimodal` 로더의 이미지 개수 태그 청크가 정보량이 적어서 임베딩 공간에서 다른 질문들과 두루뭉술하게 가까워지는 "노이즈 자석" 역할을 하고 있었음(여러 무관한 질문에서 동일한 태그-only 청크가 최상위로 잡힘). 2) 한국어 질문으로는 마크다운으로 바꿔도, 심지어 다국어 임베딩(`paraphrase-multilingual-MiniLM-L12-v2`)으로 바꿔도 특정 질문(".CATPart 확장자")은 여전히 top-4 실패 → **임베딩 모델(`all-MiniLM-L6-v2`, 사실상 영어 전용)이 마크다운보다 더 근본적인 병목**으로 보임 | 1)은 마크다운 로더가 근본적으로 해결(이미지 태그 대신 실제 문서 구조 보존, 태그는 필요시 헤더 밖에 별도 메타데이터로 분리하는 걸 다음 개선으로 제안). 2)는 미해결 — `rag_chain.py`의 `expand_query()`(LLM으로 한국어 질문을 CATIA 영문 용어로 확장 후 검색)가 이미 이 문제의 완화책으로 구현되어 있어서, **raw 임베딩 검색이 아니라 실제 RAGPipeline(쿼리 확장 포함)으로 재평가할 필요 있음** — 다음 실험으로 남김 |
| 2026-07-27 | 조현우 | 팀 공식 `run_evaluation_benchmark()`로 기존(multimodal) vs 마크다운 벡터스토어를 Qwen2.5-3B-Instruct 기준 3-Way(Direct/Strict RAG/Adaptive) 정식 비교 | 로더만 교체, LLM/청킹 파라미터는 동일 | **Strict RAG 평균 F1: 기존 0.8071 → 마크다운 0.8105 (+0.0034, 전 카테고리 퇴보 없음)**. General QA·Draft Angle 질문에서 개선, 나머지는 동일. 결과 CSV: `report/bench_old_multimodal_summary.csv`, `report/bench_markdown.csv` | 한 프로세스 안에서 Qwen 모델을 두 번(기존→마크다운) 연달아 로드하니 8GB GPU의 VRAM이 부족해져 일부 레이어가 CPU로 offload되며 극도로 느려짐(1시간+ 걸려도 안 끝남) | 프로세스를 kill하고, 두 벤치마크를 **완전히 분리된 프로세스**로 각각 실행 → 정상 속도(수 분)로 완료. **교훈: 로컬 GPU 모델을 여러 설정으로 비교할 때는 반드시 프로세스를 분리할 것** (같은 스크립트 안에서 `del model; torch.cuda.empty_cache()`로도 가능하지만 분리 실행이 더 안전) |

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
