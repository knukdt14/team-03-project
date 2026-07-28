# RAG 성능 튜닝 실험 리포트 (임베딩/로더/모델 크기)

**CATIA 매뉴얼 RAG 시스템에서 "PDF→Markdown 변환"과 "임베딩 모델 교체"가 성능에 미치는 영향을 정량 측정한 기록입니다.**

`Qwen2.5-0.5B` / `1.5B` / `3B` / `7B(API)` · 5문항 3-Way 벤치마크(`run_evaluation_benchmark`) · BERTScore F1 · RTX 4070 8GB

---

## 최종 결론 요약

### 실측 최고 성능: **F1 = 0.8228**

```yaml
모델:        Qwen/Qwen2.5-3B-Instruct (local GPU)
RAG:         Strict RAG ON
로더:        Markdown (pymupdf4llm, 헤더/표 구조 보존)
임베딩:      paraphrase-multilingual-MiniLM-L12-v2 (다국어)
```

> 기존 기본값(멀티모달 로더 + `all-MiniLM-L6-v2` 영어 전용 임베딩) 대비 **+0.0157**, 0.5B 기준으로는 **+0.05**.

### 레버별 효과 순위

| 순위 | 레버 | 효과 크기 | 판정 |
|:---:|---|---|:---:|
| 1 | **모델 크기 (0.5B→3B)** | **+0.171** | 🟢 압도적 |
| 2 | **임베딩 모델 (영어전용→다국어)** | **+0.012 ~ +0.050** (모델이 작을수록 큼) | 🟢 큼 |
| 3 | **PDF→Markdown 로더** | +0.003 ~ +0.009 | 🟡 작지만 일관됨 |
| 4 | API 모델(7B/Llama8B) | 측정 불가 | 🔴 결제 오류로 무효 |

**핵심 3줄**
1. **모델 크기가 압도적** — 0.5B(0.65)→1.5B(0.72)→3B(0.82)로 로컬 모델 안에서는 크기가 성능을 가장 크게 좌우함
2. **다국어 임베딩이 그다음** — 특히 **약한 모델(0.5B)일수록 효과가 큼**(+0.050), 강한 모델(3B)에서는 효과가 작아짐(+0.012). 검색 품질이 나쁜 걸 모델 크기로 어느 정도 상쇄한다는 뜻
3. **PDF→Markdown 구조 보존은 작지만 공짜로 얻는 개선** — 전 카테고리 퇴보 없이 꾸준히 소폭 개선, 게다가 이미지 태그 노이즈 문제도 부수적으로 해결

---

## 목차

1. [실험 환경 및 방법](#1-실험-환경-및-방법)
2. [임베딩 모델 비교 (전 모델 공통)](#2-임베딩-모델-비교-전-모델-공통)
3. [모델 크기 비교 (0.5B → 1.5B → 3B)](#3-모델-크기-비교-05b--15b--3b)
4. [PDF→Markdown 로더 효과](#4-pdfmarkdown-로더-효과)
5. [정성적 사례 — 고난도 질문 테스트](#5-정성적-사례--고난도-질문-테스트)
6. [전체 실험 총괄표](#6-전체-실험-총괄표)
7. [최종 결론](#7-최종-결론)
8. [부록 A: 벡터스토어 git 추적 손상 재발견](#부록-a-벡터스토어-git-추적-손상-재발견)
9. [부록 B: HuggingFace API 결제 오류로 벤치마크 무효화](#부록-b-huggingface-api-결제-오류로-벤치마크-무효화)
10. [부록 C: GPU/CPU 설정 이슈](#부록-c-gpucpu-설정-이슈)

---

## 1. 실험 환경 및 방법

| 항목 | 값 |
|---|---|
| 비교 모델 | Qwen2.5-0.5B / 1.5B / 3B-Instruct (local, fp16), Qwen2.5-7B-Instruct (API, 결제오류로 무효) |
| 평가 지표 | BERTScore F1 (`lang="ko"`) |
| 평가셋 | 팀 공식 `BENCHMARK_DATASET` 5문항 — General QA 1 / Specific QA 2 / Trick·Unanswerable QA 2 (`src/evaluation.py`) |
| 벡터스토어 3종 | `chroma_db_multimodal`(기존), `chroma_db_markdown`(신규), `chroma_db_markdown_multilingual`(신규) |
| 임베딩 2종 | `all-MiniLM-L6-v2`(기존, 영어 위주) / `paraphrase-multilingual-MiniLM-L12-v2`(다국어) |
| GPU | RTX 4070 Laptop 8GB |
| 생성 설정 | `temperature=0.01`, `do_sample=False` |

### 3가지 응답 모드
| 모드 | 동작 |
|---|---|
| Direct LLM (RAG Off) | 검색 없이 모델 사전지식만으로 답변 |
| **Strict RAG (RAG On)** | 매뉴얼 검색 결과만 근거로 답변, 없으면 "매뉴얼에 없음" |
| Adaptive Fallback RAG | 매뉴얼에 없으면 Direct LLM으로 폴백 |

> 본 리포트의 주 비교는 **Strict RAG F1** 입니다 (팀 README의 3-Way 벤치마크와 동일 지표).

---

## 2. 임베딩 모델 비교 (전 모델 공통)

같은 마크다운 로더 청크(5,245개)를 **임베딩 모델만 바꿔서** 재인덱싱 후 비교.

| 모델 | 마크다운+영어전용(all-MiniLM) | 마크다운+다국어 | 격차 |
|---|:---:|:---:|:---:|
| Qwen 0.5B | 0.6105 | **0.6518** | **+0.0413** |
| Qwen 1.5B | (미측정) | 0.7150 | — |
| Qwen 3B | 0.8105 | **0.8228** | **+0.0123** |

### 해석
- **약한 모델일수록 임베딩 개선 효과가 큼** (0.5B: +0.041 vs 3B: +0.012) — 모델이 똑똑하면 다소 부정확한 검색 결과도 잘 활용하지만, 모델이 약할수록 정확한 근거 문서가 더 절실함
- 검색 로그를 직접 까본 결과, 기존 `all-MiniLM-L6-v2`는 한국어 질문 임베딩이 벡터 공간에서 영어 문서와 잘 안 붙는 문제가 실측 확인됨(부록 A 참고 전 단계 진단)
- 다국어 임베딩으로 바꾸는 것은 **재학습·재청킹 없이 임베딩 함수 인자 하나만 바꾸면 되는 가장 저비용 개선**

📁 `report/bench_qwen05b_markdown.csv` vs `report/bench_qwen05b_markdown_multilingual.csv`
📁 `report/bench_qwen3b_markdown_multilingual.csv` (마크다운+영어전용 3B는 이전 세션 값 0.8105 인용, 재현 파일 없음 — 재측정 권장)

---

## 3. 모델 크기 비교 (0.5B → 1.5B → 3B)

동일 벡터스토어(마크다운+다국어 임베딩), 동일 5문항.

| 모델 | Strict RAG F1 | 이전 대비 |
|---|:---:|:---:|
| Qwen2.5-0.5B | 0.6518 | — |
| Qwen2.5-1.5B | 0.7150 | +0.0632 |
| **Qwen2.5-3B** | **0.8228** | **+0.1078** |
| Qwen2.5-7B (API) | ~~0.5998~~ | **측정 무효** (부록 B) |

### 문항 유형별 (0.5B → 3B)

| 질문 유형 | 0.5B | 1.5B | 3B | 총 개선폭 |
|---|:---:|:---:|:---:|:---:|
| General QA | 0.6489 | 0.6690 | 0.7249 | +0.0760 |
| Specific(Stiffener) | 0.6746 | 0.7229 | 0.6533 | -0.0213 |
| Specific(Draft Angle) | 0.6825 | 0.5899 | 0.7359 | +0.0534 |
| Trick QA 1 | 0.6369 | 0.5930 | **1.0000** | +0.3631 |
| Trick QA 2 | 0.6164 | **1.0000** | **1.0000** | +0.3836 |

### 해석
- **모델 크기 효과가 임베딩·로더 개선보다 한 자릿수 큼** — 3B 도달 시 트릭 질문 F1이 만점(1.0000)에 도달, 즉 **환각을 완전히 차단**
- 1.5B에서 이미 트릭 QA 2번이 만점을 달성한 걸 보면, "일정 크기 이상이면 검색 근거 없음을 정직하게 인정하는 능력"이 생기는 임계점이 있는 것으로 추정
- Specific(Stiffener) 문항은 1.5B가 3B보다 오히려 높음(0.7229 vs 0.6533) — 모든 문항에서 크기가 항상 이기는 건 아니며, 개별 문항 단위로는 노이즈가 있음(5문항이라 표본이 작다는 한계, 6절 참고)

📁 `report/bench_qwen05b_markdown_multilingual.csv`, `bench_qwen15b_markdown_multilingual.csv`, `bench_qwen3b_markdown_multilingual.csv`

---

## 4. PDF→Markdown 로더 효과

기존 `multimodal` 로더(순수 텍스트 + "🖼️ N개 이미지 포함" 태그) vs 신규 `markdown`(pymupdf4llm, 헤더/표 구조 보존 + `MarkdownHeaderTextSplitter`).

| 모델 | 기존(multimodal) | 마크다운 | 격차 |
|---|:---:|:---:|:---:|
| Qwen 0.5B | 0.6018 | 0.6105 | +0.0087 |
| Qwen 3B | 0.8071¹ | 0.8105 | +0.0034 |

¹ 이 수치는 이후 부록 A에서 밝혀진 `chroma_db_multimodal` 손상 이전 세션에서 측정됨 — 정확한 재검증 필요(6절 총괄표의 ⚠️ 참고)

### 발견한 부수 효과 (수치화 안 됐지만 중요)
- **이미지 태그 전용 청크의 검색 노이즈 제거**: "🖼️ [매뉴얼 도면/스크린샷 포함: N개...]" 만 있고 본문이 거의 없는 페이지가, 여러 무관한 질문에서 최상위로 검색되던 현상을 마크다운 로더가 근본적으로 제거함 (본문 구조를 보존하니 애초에 이런 정보량 낮은 페이지가 안 생김)
- **헤더 메타데이터 보존**: 청크에 `h1~h4`(예: "Saving Assembly Information") 메타데이터가 남아서, 향후 출처 인용이나 섹션별 필터링에 활용 가능

📁 `report/bench_old_multimodal_summary.csv`, `bench_markdown.csv`

---

## 5. 정성적 사례 — 고난도 질문 테스트

팀원이 카카오톡으로 공유한 **정답 없는 17문항** ("복사 요소를 클립보드에 저장하는 툴 아이콘은?" 등 초세부 UI 조작 질문)을 마크다운+다국어 임베딩 조합(Qwen 7B API)으로 4문항 샘플 테스트.

| 질문 | 답변 |
|---|---|
| 클립보드 저장 아이콘? | "매뉴얼에 없음" (정직한 거절) |
| Specification Tree 토글 키? | **"F3"** — 출처 4건과 함께 구체적 정답 제시 |
| Snap to Point 아이콘 명칭? | "매뉴얼에 없음" (정직한 거절) |
| Hole V-Bottom Type 옵션 탭? | "매뉴얼에 없음" (정직한 거절) |

**4문항 중 3문항 정직 거절 + 1문항 정확한 출처 기반 답변** — 정답 세트가 없어 정량 채점은 못 했지만, 초세부 질문에서도 **근거 없이 지어내지 않는** 행동을 확인. 나머지 13문항 채점 및 정식 평가셋 편입은 향후 과제로 남김.

📁 원본 질문: `C:\Users\...\카카오톡 받은 파일\CATIA_RAG_High_Gap_Questions_Only.txt` (팀 공유 파일)

---

## 6. 전체 실험 총괄표

| # | 실험 | 모델 | 조합 | F1 | 판정 |
|:---:|---|---|---|:---:|:---:|
| 1 | 임베딩 비교 | 0.5B | 마크다운+영어전용 → 다국어 | 0.6105 → **0.6518** | 🟢 |
| 2 | 임베딩 비교 | 3B | 마크다운+영어전용 → 다국어 | 0.8105 → **0.8228** | 🟢 |
| 3 | 모델 크기 | 0.5B→1.5B→3B | 마크다운+다국어 고정 | 0.6518→0.7150→**0.8228** | 🟢 |
| 4 | 로더 비교 | 0.5B | 멀티모달→마크다운 | 0.6018→0.6105 | 🟡 |
| 5 | 로더 비교 | 3B | 멀티모달⚠️→마크다운 | 0.8071⚠️→0.8105 | ⚠️ 재검증 필요 |
| 6 | API 모델 | 7B (HF Inference API) | 마크다운+다국어 | ~~0.5998~~ | 🔴 무효(결제오류) |
| 7 | API 모델 | Llama-3.1-8B (API) | — | 미측정 | ⏸️ 비용 우려로 중단 |
| 8 | 고난도 정성 테스트 | 7B API | 마크다운+다국어, 4/17문항 | 정답 없음(정성 평가만) | ✅ |

⚠️ = 부록 A의 vect/ git 추적 손상 발견 **이전**에 측정된 값으로, 당시 `chroma_db_multimodal`이 정상이었는지 확인되지 않음(재빌드 후 값과 다를 수 있음)

---

## 7. 최종 결론

### 7-1. 성능을 실제로 올린 것 (효과 큰 순서)

| 레버 | 효과 | 비고 |
|---|:---:|---|
| **모델 크기 (0.5B→3B)** | **+0.171** | 로컬 GPU 메모리가 허용하는 한 가장 확실한 레버 |
| **다국어 임베딩 교체** | **+0.012~+0.050** | 재학습 불필요, 임베딩 함수 인자 하나로 적용. 약한 모델일수록 효과 큼 |
| PDF→Markdown 로더 | +0.003~+0.009 | 작지만 퇴보 없음 + 이미지태그 노이즈 제거 부수효과 |

### 7-2. 확인 안 된 것 / 실패한 것

| 항목 | 결과 | 이유 |
|---|:---:|---|
| Qwen 7B (API) | 측정 무효 | HuggingFace Inference API 402 결제 오류, 5문항 중 3~4문항 실패 |
| Llama-3.1-8B (API) | 접근은 되나 벤치마크 미실행 | 7B와 같은 결제 벽에 부딪힐 위험 커서 사용자 판단으로 중단 |

### 7-3. 최종 권장 설정

```yaml
모델:        Qwen/Qwen2.5-3B-Instruct (local GPU) — VRAM 허용 시 가장 큰 로컬 모델
RAG:         Strict RAG ON
로더:        markdown (pymupdf4llm)
임베딩:      paraphrase-multilingual-MiniLM-L12-v2
```

### 7-4. 발표용 스토리

1. **로컬 소형 LLM은 크기가 커질수록, 그리고 검색 품질이 좋을수록 RAG 답변 품질이 좋아진다** — 0.5B(0.65)→3B(0.82)까지 일관된 상승
2. **다국어 임베딩 하나 바꾸는 것만으로 무료로 성능이 오른다** — 약한 모델(0.5B)에서 효과가 가장 컸다는 게 실용적 시사점 (자원이 부족한 상황일수록 검색 품질 투자가 중요)
3. **API 기반 대형 모델은 비용 문제로 이번엔 검증하지 못함** — 무료 크레딧이 벤치마크 도중 소진되는 걸 직접 겪음, 향후 유료 크레딧 확보 후 재시도 필요
4. **버그를 스스로 찾아 고친 과정 자체가 성과** — vect/ git 추적 손상 재발견(부록 A), API 결제오류로 인한 무효 데이터 식별(부록 B), GPU 강제-CPU 설정 발견(부록 C)

### 7-5. 한계 및 향후 과제

- **평가 문항이 5개뿐**: 참고한 팀 리포트(61문항)에 비해 표본이 매우 작아 개별 문항 노이즈(Specific-Stiffener에서 1.5B>3B 역전 등)에 취약. 61문항 평가셋으로 재검증 필요
- **3B+마크다운+영어전용(all-MiniLM) 조합 재검증 필요**: 0.8071/0.8105 값이 vect/ 손상 발견 이전에 측정되어 신뢰도 낮음
- **7B/Llama8B 미검증**: 유료 크레딧 확보 또는 로컬 양자화(4-bit) 실행으로 재시도 가능
- **고난도 17문항 중 13문항 미채점**: 정답 세트 확보 후 정식 평가셋 편입 필요

---

## 부록 A: 벡터스토어 git 추적 손상 재발견

팀원(Racoon7828)이 먼저 발견/공유한 버그를 이 컴퓨터에서 독립적으로 재현·검증함.

### 증상
`chroma_db_multimodal` 검색기가 `top_k=2/4/6` **무관하게 문서 1개만 반환** (컬렉션엔 5,278개 있음에도).

### 원인
`.gitignore`에 `vect/`가 있었지만 **`chroma.sqlite3`가 이미 추적 중이던 파일이라 안 풀림** → 매 `git merge`마다 로컬에서 새로 빌드한 인덱스를 커밋된 옛날 sqlite가 덮어써서 메타데이터-실제 인덱스 불일치 발생.

### 검증
- `chroma_db_multimodal`(git 추적됨): k=2/4/6 전부 1개 반환 — **손상 재현**
- `chroma_db_markdown`(한 번도 git 추적 안 됨): k값대로 정상 반환 — **오염은 git 추적 폴더에 국한**됨을 확인

### 조치
- main에는 이미 `fe5027b`(Racoon7828)로 수정 완료 확인
- 이 컴퓨터의 `chroma_db_multimodal`을 삭제 후 `python -u -m src.build_multimodal_vectorstore`로 클린 재빌드, 정상 반환 재확인
- 재발 방지: 벡터스토어 로드 직후 `len(vs.similarity_search(q, k=N)) == N` 스모크 테스트 권장

---

## 부록 B: HuggingFace API 결제 오류로 벤치마크 무효화

### 무슨 일이 있었나
`qwen_7b_api` 프리셋으로 5문항 3-Way 벤치마크 실행 중, **총 15개 호출(Direct 5 + Strict RAG 5 + Adaptive 5) 중 다수가 `402 Payment Required`로 실패**:

| 응답 종류 | 실패 건수 |
|---|:---:|
| Direct LLM Answer | 3/5 |
| Strict RAG Answer | 3/5 |
| Adaptive Fallback Answer | 4/5 |

`RAGPipeline`이 예외를 잡아서 `"⚠️ [RAG 응답 생성 오류]: Client error '402 Payment Required'..."` 문자열을 그대로 답변 필드에 넣고, 이 에러 문자열이 그대로 BERTScore로 채점되면서 **F1 0.5998이라는 그럴듯하지만 의미 없는 숫자**가 나왔음.

### 어떻게 발견했나
7B(더 큰 모델)가 0.5B보다 F1이 낮게 나온 게 이상해서, 결과 CSV의 답변 컬럼을 직접 열어봄 → 에러 메시지가 답변으로 저장되어 있는 것을 확인.

### 교훈
- **API 모델 벤치마크 결과는 반드시 답변 원문을 직접 확인할 것** — F1 숫자만 보면 절대 못 잡아냄
- 무료 크레딧은 첫 호출(단발 테스트) 성공만으로 안심할 수 없음 — 전체 벤치마크(약 15회 호출) 도중 소진될 수 있음
- Llama-3.1-8B API는 접근 권한(토큰 fine-grained 권한 설정)까지는 확인했으나, 동일한 결제 벽 위험 때문에 **전체 벤치마크는 사용자 판단으로 중단**

---

## 부록 C: GPU/CPU 설정 이슈

### 발견 1 — 임베딩이 GPU를 안 쓰고 있었음
`src/vector_store.py`의 `get_embedding_function()`이 `model_kwargs={'device': 'cpu'}`로 **하드코딩**되어 있어서, RTX 4070이 있는데도 임베딩 계산에 GPU를 전혀 안 쓰고 있었음. `torch.cuda.is_available()` 체크를 추가해 GPU 있으면 자동 사용하도록 수정.

### 발견 2 — 두 모델을 한 프로세스에서 연달아 로드하면 VRAM 부족
Qwen 모델을 한 스크립트 안에서 두 번(비교 대상별로) 로드하면 8GB GPU가 부족해져 일부 레이어가 CPU로 offload되며 극도로 느려짐(1시간 넘게 안 끝남). **벤치마크는 항상 완전히 분리된 프로세스로 실행**해야 함.

### 발견 3 — Python 출력 버퍼링으로 "멈춘 것처럼" 보임
`pymupdf4llm`의 대용량 PDF(784페이지 등) 변환은 실제로 페이지당 수 초씩 걸려 32개 PDF에 정상적으로 ~16분이 걸리는데, 출력이 버퍼링되어 그 사이 아무 로그도 안 보여서 "죽은 줄" 착각하고 프로세스를 성급히 kill한 적이 있음. `python -u`(unbuffered)로 실행하면 실시간 로그 확인 가능.

---

## 실행 방법

```bash
# 마크다운 벡터스토어 빌드 (영어전용 임베딩, 기존 기본값)
python -u -m src.build_markdown_vectorstore

# 마크다운 + 다국어 임베딩 벡터스토어 빌드
python -u -m src.build_markdown_multilingual_vectorstore

# 벡터스토어 상태 확인 (필수, 손상 여부 스모크 테스트)
python -c "
from src.vector_store import get_embedding_function
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DB_DIR
vs = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=get_embedding_function())
print('검색:', len(vs.similarity_search('test', k=4)), '개 (4개가 나와야 정상)')
"

# 특정 모델 + 벡터스토어 조합으로 벤치마크 (반드시 별도 프로세스로)
python -u -c "
from langchain_community.vectorstores import Chroma
from src.vector_store import get_embedding_function
from src.config import CHROMA_DB_DIR_MARKDOWN_ML, MULTILINGUAL_EMBEDDING_MODEL
from src.rag_chain import RAGPipeline
from src.evaluation import run_evaluation_benchmark

vs = Chroma(persist_directory=CHROMA_DB_DIR_MARKDOWN_ML, embedding_function=get_embedding_function(MULTILINGUAL_EMBEDDING_MODEL))
pipe = RAGPipeline(model_name='Qwen/Qwen2.5-3B-Instruct', vectorstore=vs)
df = run_evaluation_benchmark(pipe)
print(df[['Type','Strict RAG F1']])
"
```
