# team-03-project

**KDT14기 3팀**

## 팀 구성

| 역할 | 이름 | 담당 |
|---|---|---|
| 팀장 | 조현우 | 프로젝트 총괄(기획·관리·결과 도출), 발표자료 제작 |
| 팀원 | 김병욱 | 여러 LLM 검토, Evaluation(BERTScore·환각) 작성 |
| 팀원 | 정재영 | 매뉴얼 로드·분할·저장 전략으로 성능 향상 |
| 팀원 | 정우렬 | 벡터DB·유사도 검색으로 성능 향상, 절차 검증 로직 |

---

## 주제

### CATIA·AutoForm 매뉴얼 기반 RAG 질의응답 및 작업 절차 정합성 검증 시스템

CATIA·AutoForm 매뉴얼을 기반으로 **(1)** 툴 사용법에 답하고, **(2)** 사용자가 입력한 작업 절차를 매뉴얼의 표준과 비교해 정합성을 피드백하는 RAG 시스템을 구현한다. 다양한 LLM·임베딩·벡터스토어·프롬프트 조합의 성능을 정량 비교하고, 문서에 근거 없는 답변의 환각 발생을 측정·저감한다.

> **부제** — 사용자가 입력한 CATIA·AutoForm 작업 절차가 매뉴얼의 표준(정석)과 얼마나 일치하는지 검색·비교해 피드백하는 RAG 시스템

---

## 주제 선정 이유

**실무 연관성 — 자동차 부품 제조 현장과 직결**
CATIA는 차체 부품 3D 설계, AutoForm은 판재 성형(프레스) 해석에 쓰이는 실제 현장 툴이다. "제조 현장 신입 엔지니어의 온보딩·작업표준 검증"이라는 명확한 실사용 시나리오가 있어, 프로젝트가 단순 실습을 넘어 포트폴리오로서 가치를 갖는다.

**환각(Hallucination) 저감 스토리가 자연스럽다**
툴 사용법이나 작업 절차를 틀리게 답하면 엔지니어의 시간 손실·불량으로 이어진다. 따라서 "매뉴얼에 근거하지 않은 답은 지어내지 않고 '모른다'로 응답"하는 것이 핵심 품질 지표가 되며, 이는 과제 평가항목(Hallucination 발생 사례)과 정확히 맞물린다.

**RAG 구조에 적합하다**
매뉴얼은 방대한 텍스트 문서라 청킹·임베딩·검색 실험거리가 풍부하다. 특히 "내 절차 입력 → 매뉴얼에서 표준 절차 검색 → 비교·피드백"은 검색 품질이 결과를 좌우하는, RAG의 강점을 보여주기 좋은 과제다.

---

## 기능 구조

| 구분 | 기능 | 설명 |
|---|---|---|
| **필수 기능** | 매뉴얼 사용법 QA | 매뉴얼 기반 툴 사용법 질의응답 (과제 채점 대상) |
| **차별화 기능** | 작업 절차 정합성 검증 | 사용자 작업 절차 ↔ 매뉴얼 표준 비교·피드백 |

---

## 목표

- LangChain 기반 RAG 질의응답 시스템 구현 (벡터스토어 포함)
- 다양한 설정 조합의 답변 품질을 BERTScore로 정량 비교
- 문서에 근거 없는 질문에 대한 환각 발생을 측정하고 프롬프트·검색 개선으로 저감
- 사용자 작업 절차와 매뉴얼 표준의 정합성 피드백 기능 구현

---

## 실험 항목

| 변수 | 후보 |
|---|---|
| LLM | HuggingFace 공개 모델 / OpenAPI(ChatGPT·Claude·Upstage) 중 선정 |
| 임베딩 모델 | ko-sroberta, bge-m3 등 |
| 벡터스토어 | FAISS / Chroma |
| 유사도 검색 | 코사인 / MMR 등 |
| 프롬프트 | 기본 / 근거강제("매뉴얼에 없으면 모른다고 답변") |
| 청킹·검색 파라미터 | `chunk_size`, `overlap_size`, `top_k` |

---

## 평가 지표

- **BERTScore** — 답변과 정답의 의미 유사도
- **응답 시간**
- **Hallucination 발생 사례** — 매뉴얼에 없는 질문에 지어내는 비율
- **정성 평가** — 절차 검증 피드백의 유용성 (팀원 직접 평가)

---

# 구현 스타터 — CATIA 교육자료 기반 RAG 파이프라인 (Module 12)

CATIA 교육자료(영어·한글 혼합) PDF를 근거로 질문에 답하는 RAG 파이프라인입니다. LLM·임베딩·벡터스토어·청킹 파라미터·프롬프트를
`config.py`에서 바꿔가며 실험하도록 구성했습니다 (요구사항의 비교 실험 항목과 1:1로 대응).

## 폴더 구조
```
team-03-project/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── (CATIA 교육자료 PDF 수십 개, 영어·한글 혼합)
├── src/
│   ├── config.py          실험 프리셋
│   ├── load_pdf.py         PDF 로드(PyPDFLoader) + 청킹
│   ├── build_vectorstore.py 임베딩 + 벡터스토어(FAISS/Chroma/Pinecone)
│   ├── rag_chain.py        LLM + 프롬프트 + 검색 체인
│   ├── evaluate.py         BERTScore + RAGAS + hallucination 컬럼
│   └── main.py             CLI 실행 진입점
├── eval/
│   ├── questions_template.csv
│   └── results.csv        (실행 후 생성)
├── report/                 최종 보고서 넣는 곳
└── slides/                 발표자료 넣는 곳
```

## 설치
1. `conda activate DL_PY311` (또는 새 가상환경 — ragas 충돌 때문에 새 가상환경 권장, 아래 참고 참조)
2. `cd src` 후 `pip install -r ../requirements.txt --break-system-packages` (필요시)
3. `.env.example`을 `.env`로 복사하고 사용할 API 키만 입력 (`HF_TOKEN`도 lab_01처럼 필요할 수 있음)

## 실행
`src` 폴더 기준으로 실행합니다.
- 기본 조합: `python main.py --preset baseline`
- 다른 조합: `python main.py --preset large_chunk` / `kr_sbert_embed` / `chroma_store` / `openai_llm` 등
- RAGAS 없이 빠르게: `python main.py --preset baseline --no_ragas`
- 답변 직접 테스트: `python main.py --mode chat`
- **RAG 적용 vs 미적용(Direct LLM) 비교**: `python main.py --mode compare --preset baseline`
  - CATIA 매뉴얼이 LLM 사전학습에 이미 포함됐을 가능성이 높다는 지적에 대응하기 위한 비교 실험입니다.
  - 같은 질문셋을 RAG 체인과 검색 없는 Direct LLM 체인에 동시에 흘려서 `eval/results_compare.csv`에 나란히 저장합니다.
  - `eval/questions_template.csv`의 `question_type`이 `trick`인 질문(매뉴얼에 없는 가상 기능)에서 Direct LLM은 그럴듯하게 지어내지만 RAG는 "문서에서 답을 찾을 수 없습니다"라고 답하는지가 환각 저감의 핵심 증거입니다.
  - 실험 결과와 문제 해결 과정은 [`EXPERIMENTS.md`](EXPERIMENTS.md)에 기록합니다.

새 조합을 실험하려면 `config.py`의 `EXPERIMENT_PRESETS`에 딕셔너리 하나만 추가하면 됩니다.

문서가 많을 때를 위해 벡터스토어는 `vecstore_cache/`에 chunk_size·overlap·embed 조합별로 캐싱됩니다. LLM/top_k/prompt만 바꾼 재실행은 재임베딩 없이 빠르게 돌아가고, chunk_size나 embed_model을 바꾸면 그 조합에 맞는 새 캐시가 자동으로 만들어집니다. (`vecstore_cache/`는 실행할 때 생기는 산출물이라 제출 시 제외해도 됩니다.)

## 평가 데이터셋
`eval/questions_template.csv`에 질문·정답·근거 문장을 **10개 이상** 채워 넣으세요 (필수 요구사항).
실행하면 `eval/results.csv`에 아래 컬럼이 저장됩니다.
- `predicted_answer`, `retrieved_context`, `response_time_sec`
- `bertscore_precision` / `recall` / `f1` (필수 지표)
- `ragas_faithfulness` / `answer_relevancy` / `context_precision` / `context_recall` (RAGAS 정상 설치 시 — lab_06과 동일한 지표)
- `hallucination_flag`, `reviewer_note` — 팀원이 직접 채우는 칸 (RAGAS faithfulness와 교차검증용)

## 평가 기준과의 매핑
| 평가 항목 | 관련 파일 |
|---|---|
| RAG / Evaluation | `config.py`의 프리셋들 + `eval/results.csv` 비교 |
| Document Retrieval | `load_pdf.py`, `build_vectorstore.py` |
| Semantic Search | `rag_chain.py`의 `search_type` (similarity/mmr) |
| 팀원 역할 분담 | README에 역할별 담당 파일을 적어 제출 |
| 프로젝트 완성도 | `main.py` 실행 로그/스크린샷 |

## 참고
- lab_01~07 노트북 내용에 맞춰 PDF 로더는 `PyPDFLoader`, 벡터스토어는 `langchain_chroma`(standalone) + FAISS + Pinecone(선택), 임베딩 기본값은 `paraphrase-multilingual-MiniLM-L12-v2`로 맞췄습니다. 기본 LLM은 **Upstage `solar-pro2`** (제일 먼저 발급받는 키라 baseline 기본값으로 설정), `openai_llm` 프리셋으로 lab_06/07의 `gpt-5.4-nano`와도 비교 가능합니다. `kr_sbert_embed` 프리셋으로 lab_04의 한국어 특화 임베딩(`KR-SBERT`)과도 비교할 수 있습니다.
- **RAGAS 알려진 이슈**: 2026.07 기준 `ragas`가 최신 `langchain-community`와 임포트 충돌을 일으킬 수 있음을 확인했습니다 (langchain 생태계가 0.x→1.x 전환 중이라 발생). `evaluate.py`는 RAGAS가 실패해도 조용히 건너뛰고 BERTScore로는 계속 진행하도록 만들어뒀습니다. 안 되면: (1) `python main.py --no_ragas`로 일단 진행, (2) 새 가상환경에 `ragas`만 먼저 설치해보기, (3) 그래도 안 되면 BERTScore + 수동 hallucination_flag만으로도 평가 요구사항(필수 지표는 BERTScore)은 충족됩니다.
- 각 페이지에는 `lang`(ko/en) 메타데이터가 자동으로 붙어서, "한국어 문서 vs 영어 문서 검색 정확도" 같은 추가 비교 실험도 가능합니다. 각 청크에는 `chunk_id`도 붙습니다(lab_07과 동일).
- 로컬 GPU(RTX 4070)로 Hugging Face 모델을 직접 돌리려면 `llm_provider: huggingface`로 설정하세요. 7~8B급 모델은 4-bit 양자화(`bitsandbytes`)를 쓰면 VRAM에 더 여유가 있습니다.
- `pinecone_store` 프리셋을 쓰려면 `requirements.txt` 맨 아래 주석 처리된 `langchain-pinecone`/`pinecone`을 설치하고 `.env`에 `PINECONE_API_KEY`를 넣어야 합니다.
- LangChain은 패키지 업데이트가 잦아 설치된 버전에 따라 일부 import 경로가 조금 다를 수 있습니다. 에러 나면 해당 함수명으로 최신 문서를 검색해서 경로만 맞춰주면 됩니다.
