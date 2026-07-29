# team-03-project

**KDT14기 3팀**

## 팀 구성

| 역할 | 이름 | 담당 |
|---|---|---|
| 팀장 | 조현우 | 프로젝트 총괄(기획·관리·결과 도출), 임베딩 개선(BGE-M3) |
| 팀원 | 김병욱 | 이미지 검색 품질 개선(OCR), 응답 안정성 개선, 버그 확인 및 수정 |
| 팀원 | 정재영 | 매뉴얼 로드·분할·저장 전략, High-Gap 평가셋 구축, 대시보드 구축 |
| 팀원 | 정우렬 | 벡터DB·검색(BGE-M3/MMR/하이브리드) 최적화, 환각 방지(Grounding Check) 로직 |

---

## 주제

### CATIA V5 매뉴얼 기반 RAG 질의응답 시스템

CATIA V5 매뉴얼(공식 문서 + 실습 교재 + 강의 자료 32종, 2,400여 페이지)을 기반으로 툴 사용법·단축키·작업 절차를 질의응답하는 RAG 챗봇을 구현한다. 소형 오픈소스 LLM(Qwen2.5, 0.5B/1.5B/3B)을 대상으로 RAG 적용 전/후 성능 격차를 정량 비교하여, **자원이 제약된 소형 모델일수록 RAG의 가치가 크다**는 것을 실증한다.

> 프로젝트 초기에는 대형 LLM(Llama 3.1 8B 등)으로 RAG 효과를 검증하려 했으나, 대형 모델은 이미 CATIA 관련 사전지식이 많아 RAG 적용 전/후 차이가 잘 드러나지 않는 문제를 발견 → 소형 모델(Qwen2.5-0.5B)로 방향을 재설정.

---

## 디렉토리 구조

```
team-03-project/
├── .env / .env.example       # API KEY 설정 (HF_TOKEN, OPENAI_API_KEY 등)
├── data/                     # CATIA 매뉴얼 PDF 원본 32종
├── vect/                     # (git 미추적) Chroma 벡터DB, 추출 이미지, OCR 캐시 - 최초 실행 시 자동 생성
├── eval/                     # 평가 데이터셋 및 결과 CSV, 상세 실험 리포트(README.md)
├── report/                   # 실험 경과 보고서, 이슈 대응 문서
├── experiments/              # 파라미터 튜닝·임베딩 비교 실험 스크립트
│   └── rag_optimization/     # BGE-M3 등 임베딩/검색 조합 실험
├── src/
│   ├── config.py             # 기본 설정: BGE-M3 임베딩, chunk 500/overlap 100, MMR, top_k 6, Qwen2.5-0.5B
│   ├── load_pdf.py           # PyMuPDF 기반 멀티모달 텍스트+이미지 추출, OCR 텍스트 통합, Markdown 로더
│   ├── build_ocr_cache.py    # 추출 이미지 전체에 OCR을 돌려 텍스트 캐시 생성
│   ├── vector_store.py       # Chroma 벡터DB 구축/로드, 벡터 검색 + BM25 하이브리드 검색
│   ├── rag_chain.py          # RAGPipeline: Direct LLM / Strict RAG / Adaptive Fallback / Grounding Check
│   ├── models/llm_factory.py # Qwen2.5(0.5B/1.5B/3B) 등 로컬(HF)·API 모델 로더
│   ├── evaluation.py         # BERTScore + RAGAS(로컬 judge, API 키 불필요) 평가
│   ├── evaluate_multimodal.py# 터미널 기반 배치 평가 스크립트
│   └── app.py                # Streamlit 데모/평가 웹 애플리케이션
├── CHANGELOG.md
└── requirements.txt
```

---

## 실행 가이드

### 1. 환경 준비 및 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Streamlit 웹 애플리케이션 실행
```bash
streamlit run src/app.py
```
최초 실행 시 `data/`의 PDF를 파싱하여 `vect/`에 벡터DB를 자동 생성합니다(약 10~15분 소요). 이후 실행부터는 저장된 벡터DB를 그대로 불러옵니다.

### 3. (선택) 이미지 OCR 캐시 생성
스크린샷 이미지 속 텍스트(대화상자 옵션명, 단축키 등)까지 검색 대상에 포함하려면, 벡터DB 생성 전에 한 번 실행합니다.
```bash
python src/build_ocr_cache.py
```

### 4. 배치 평가 실행
```bash
python src/evaluate_multimodal.py --model Qwen/Qwen2.5-0.5B-Instruct --top_k 6
```

---

## 핵심 기술

- **멀티모달 문서 파싱**: PyMuPDF로 텍스트·이미지를 문단 단위로 추출, 이미지-문단 매칭에 OCR 키워드 겹침 활용
- **검색**: BGE-M3(다국어 임베딩) + Chroma + MMR 검색 — 기존 영어 전용 임베딩 대비 한국어 질의 검색 성능 개선
- **환각 방지**: 답변 생성 후 Grounding Check로 매뉴얼 근거 여부를 재검증, 근거 없는 답변은 거부
- **평가**: BERTScore(F1/Precision/Recall) + RAGAS(로컬 모델 기반, API 키 불필요) + LLM-as-judge 정성 평가

---

## 📊 정량적 평가 결과

### 모델 크기별 RAG On/Off 비교 (Solar LLM-as-judge, 1~5점)

| 비교 모델 | 파라미터 크기 | CATIA 사전학습 지식 | Direct LLM (RAG Off) | RAG LLM (RAG On) | Solar Score Gap | 최종 평가 |
|---|:---:|---|---|---|:---:|:---:|
| Llama 3.1 8B / 3B | 8B / 3B | 높음(과다 지식) | 정답 도출 | 정답 도출 | +1.20점(격차 적음) | 미채택 |
| Qwen 2.5 1.5B | 1.5B | 보통 | 일부 오답/환각 | 정답 도출 | +2.20점 | 미채택 |
| **Qwen 2.5 0.5B** | 0.5B(초경량) | 적음(제어 가능) | 다수 오답/환각 | 정답 도출 | **+3.42점(폭발적 상승)** | **최종 선정** |

→ 모델이 작을수록 사전학습 지식이 부족해 RAG 적용 전/후 격차가 뚜렷하게 드러남. **자원이 제약된 환경일수록 RAG의 가치가 크다**는 것을 확인하여 Qwen2.5-0.5B를 최종 모델로 선정.

### 임베딩 모델 비교 (40문항 벤치마크, BERTScore F1 / Faithfulness)

| 설정 | F1 | Faithfulness | 평균 응답시간 |
|---|:---:|:---:|:---:|
| MiniLM(영어 전용) + MMR | 0.7460 | 0.6893 | 0.78초 |
| BGE-M3(다국어) + MMR | 0.7597 | 0.7737 | 0.88초 |
| **BGE-M3 + MMR + 답변 정규화** | **0.7637** | **0.8371** | 0.76초 |

→ 기존 임베딩(all-MiniLM-L6-v2)은 한국어 질의에 대한 검색 성능이 낮아(한↔영 교차언어 유사도 실측 0.069) BGE-M3로 교체.

### High-Gap(고난도) 38문항 평가

| 지표 | Direct LLM | RAG 적용 | 격차 |
|---|:---:|:---:|:---:|
| BERTScore F1 | 0.6163 | 0.6517 | +0.035 |
| Solar 정성 평가(5점) | 1.58 | 3.05 | **+1.47** |

→ 정답이 짧고 구체적인(단축키, 대화상자 옵션명 등) 고난도 질문에서는 표면적 토큰 유사도(BERTScore)보다 **LLM 정성 평가 격차가 훨씬 크게 나타나**, RAG의 실질적 효과가 더 뚜렷이 드러남.

## 상세 실험 결과

모델 크기별 비교, 임베딩/검색 파라미터 튜닝, Query Expansion 실험, 발견된 버그와 한계 등 전체 실험 기록은 **[`eval/README.md`](eval/README.md)** 를 참고하세요.
