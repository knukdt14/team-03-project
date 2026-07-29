# 지금까지 사용한 모델 & 현재 작업 상황

## 1. 사용한 LLM (답변 생성 모델)

| 모델 | 방식 | 어디서 썼나 | 비고 |
|---|---|---|---|
| **Upstage Solar Pro 2** (`solar-pro2`) | 클라우드 API | 팀 초기 베이스라인(FAISS 기반 구 아키텍처)에서 RAG vs Direct LLM 비교 실험 | 이미 CATIA를 잘 알아서 RAG 효과가 잘 안 보이는 문제가 있었음 (교수님 피드백의 발단) |
| **Qwen2.5-3B-Instruct** | 로컬 GPU(RTX 4070) 다운로드 실행 | 팀이 새로 채택한 현재 기본 모델(`qwen_3b_local` 프리셋). 지금 돌리고 있는 3-Way 벤치마크(Direct/Strict RAG/Adaptive)에 사용 중 | CATIA 지식이 상대적으로 얕아서 RAG 효과가 뚜렷하게 드러남 |
| Qwen2.5-1.5B-Instruct (`qwen_1.5b_local`) | 로컬 GPU | 설정만 있고 아직 미테스트 | 더 작고 빠름, 다음 실험 후보 |
| Qwen2.5-7B-Instruct (`qwen_7b_api`) | HF API | 설정만 있고 아직 미테스트 | 다운로드 없이 API로 바로 시도 가능 |
| Llama-3.1-8B-Instruct (`llama_8b_api`) | HF API | 설정만 있고 아직 미테스트 | 다운로드 없이 API로 바로 시도 가능 |

## 2. 사용한 임베딩 모델 (검색용)

| 모델 | 언어 특성 | 상태 |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 다국어(한국어 포함) | 구 아키텍처에서 사용, 정규화 버그 발견 후 수정함 |
| `all-MiniLM-L6-v2` | **사실상 영어 전용** | 현재 팀 기본값. 한국어 질문 검색에 약하다는 걸 이번에 확인함(핵심 발견) |
| `BAAI/bge-m3` | 다국어 | 시도했으나 `torch` 버전 문제로 로드 실패 (`torch>=2.6` 필요, 현재 `2.5.1`) |
| `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | 한국어 특화 | 프리셋만 있고 아직 미테스트 |

## 3. 지금 하고 있는 작업 — PDF → Markdown 변환

**목표**: 기존 로더(순수 텍스트 + "🖼️ N개 이미지 포함" 태그만 붙이는 방식)가 문서 구조(제목/표/리스트)를 다 날려버려서 청킹 품질이 떨어지는 문제를 해결.

**한 일**:
1. `pymupdf4llm`으로 PDF를 헤더(`#`/`##`/`###`)·표 구조를 보존한 Markdown으로 추출하는 로더 추가 (`src/load_pdf.py`)
2. 헤더 경계부터 나누는 `MarkdownHeaderTextSplitter` 적용
3. 기존 벡터DB(`vect/chroma_db_multimodal`)와 별도로 `vect/chroma_db_markdown`을 만들어서 나란히 비교 가능하게 구성
4. 팀 공식 3-Way 벤치마크(`run_evaluation_benchmark`)로 두 벡터스토어를 Qwen2.5-3B 기준으로 정량 비교 **(현재 실행 중)**

**과정에서 같이 발견/수정한 것**:
- `accelerate` 패키지 누락 → 로컬 GPU 모델 로딩이 조용히 실패하던 버그
- 기존 이미지-태그 청크가 검색 결과에 노이즈로 잡히는 문제
- 쿼리 확장(`expand_query`)이 가끔 엉뚱한 용어(SolidWorks `.sldprt`)를 붙이는 사례
- 임베딩 모델의 영어 편중 문제 (위 표 참고)

**다음 확인할 것**: 마크다운 벡터스토어의 벤치마크 결과가 나오면 기존(0.8071 F1)과 비교해서 실제 개선 여부 확정.
