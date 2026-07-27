# 실험 기록

> 이 프로젝트는 **최종 수치보다 실험 과정과 문제 해결 과정을 중점적으로 평가**합니다.
> 프리셋을 바꿔가며 실험할 때마다 아래 표에 한 줄씩 추가해 주세요. 발표자료·보고서의 핵심 근거 자료로 그대로 쓰입니다.

## 기록 양식

| 날짜 | 담당자 | 실행 명령 | 바꾼 변수 | 결과 요약 | 발생한 문제 | 해결 방법 |
|---|---|---|---|---|---|---|
| 2026-07-27 | 조현우 | `python main.py --mode compare --preset baseline` | RAG 적용 vs Direct LLM(RAG 미적용) 비교 | (실행 후 `eval/results_compare.csv`의 `bertscore_f1_gap`, `question_type`별 격차 기록) | | |
| 2026-07-27 | 조현우 | `python main.py --preset baseline` (chat 모드로 단일 질문 테스트) | 최초 엔드투엔드 실행 검증 | "솔리드 파트는 어떤 확장자로 저장되나요?"에 "문서에서 답을 찾을 수 없습니다" 오답. 정답 문장("Parts are stored in files with the extension .CATPart")은 `CATIA V5 Lectures.pdf` 13페이지에 실제로 존재함을 확인 → **데이터 문제가 아니라 검색(retrieval) 품질 문제** | 1) `pdf_path`가 상대경로("data")라 `src/`에서 실행 시 실 데이터 폴더(루트의 data/)를 못 찾음 → FileNotFoundError. 2) Windows 콘솔 기본 인코딩(cp949)에서 PDF/LLM 출력에 유니코드 특수문자 섞이면 print()가 UnicodeEncodeError로 죽음. 3) `top_k=4`(baseline), `top_k=6`(top_k_high), `mmr_search` 모두 정답 청크를 못 찾음(top-20 안에도 없었음). 4) `bge-m3` 임베딩으로 교차언어 검색 개선을 시도했으나 `torch 2.5.1` 환경에서 `transformers`가 안전 취약점(CVE-2025-32434)으로 로드 거부(`torch>=2.6` 요구) | 1), 2)는 코드 수정으로 해결. 3)은 근본 원인 두 가지를 찾음(아래 행 참고). 4)는 미해결 — torch 업그레이드는 다른 수업 과제가 공유하는 `DL_PY311` 환경을 건드릴 수 있어 보류 |
| 2026-07-27 | 조현우 | FAISS `similarity_search_with_score` 직접 호출 + 임베딩 벡터 norm 수동 계산으로 원인 추적 | 검색 실패 근본 원인 진단 | **버그 1 (심각, 수정함)**: `get_embeddings()`가 `HuggingFaceEmbeddings`를 정규화 없이 생성 → 청크마다 임베딩 벡터 크기(norm)가 3.1~4.9로 제각각인데 FAISS는 기본 raw L2 거리로 검색 → 순위가 "의미 유사도"가 아니라 "벡터 길이"로 왜곡됨. `encode_kwargs={"normalize_embeddings": True}` 추가로 수정, 기존 `vecstore_cache/`는 캐시 키에 이 설정이 반영 안 되므로 삭제 후 재빌드 필요. **원인 2 (튜닝 이슈, 팀 실험 과제)**: 정규화 수정 후에도 타겟 청크가 top-10 밖. 실제 청크 원문을 보니 500자 청크 안에 "Part Design 개요/스케치 툴/파라메트릭 기능/.CATPart 확장자" 등 여러 내용이 섞여 있어 특정 사실에 대한 임베딩 신호가 희석됨(직접 계산한 cos=0.33, top-10 컷오프는 cos≈0.4 근처로 추정) | 임베딩 정규화는 전체 프로젝트/모든 프리셋에 영향을 주는 근본 버그라 즉시 수정(`build_vectorstore.py`). 청크 크기 이슈는 **팀 실험 과제로 남김** — `small_chunk`(chunk_size=300) 프리셋으로 이 청크가 더 작게 쪼개지는지, `top_k`를 8~10으로 올리면 잡히는지 시도해보는 게 다음 스텝 |
| 2026-07-27 | 조현우 | `python main.py --preset small_chunk` (chunk_size=300, 정규화 수정 반영 후) | 청크 크기를 줄이면 해결되는지 검증 | 여전히 오답. top-4에 타겟 청크 안 잡힘 → **단순 청크 크기 문제가 아니라, 이 임베딩 모델(`paraphrase-multilingual-MiniLM-L12-v2`)이 이런 짧은 팩트형 질문의 의미를 잘 못 잡는 것으로 보임** | (미해결, 팀 실험 필요) | **다음 팀원이 시도해볼 것**: (1) `kr_sbert_embed`/`openai_embed` 프리셋으로 임베딩 모델 자체를 바꿔서 비교, (2) `top_k`를 8~10으로 올려서 순위 밖 청크까지 확인, (3) 질문 10개 전체를 `--mode eval`로 돌려서 이 케이스가 예외인지 전반적 패턴인지 확인 — baseline 대비 각 프리셋의 BERTScore 변화를 `EXPERIMENTS.md`에 계속 기록해주세요 |

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
