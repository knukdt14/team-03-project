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

### CATIA 매뉴얼 기반 RAG 질의응답 및 작업 절차 정합성 검증 시스템

CATIA 매뉴얼을 기반으로 **(1)** 툴 사용법에 답하고, **(2)** 사용자가 입력한 작업 절차를 매뉴얼의 표준과 비교해 정합성을 피드백하는 RAG 시스템을 구현한다. 다양한 LLM·임베딩·벡터스토어·프롬프트 조합의 성능을 정량 비교하고, 문서에 근거 없는 답변의 환각 발생을 측정·저감한다.

> **부제** — 사용자가 입력한 CATIA 작업 절차가 매뉴얼의 표준(정석)과 얼마나 일치하는지 검색·비교해 피드백하는 RAG 시스템

---

## 디렉토리 및 시스템 구조

```
team-03-project/
├── .env                  # API KEY 설정 (OPENAI_API_KEY, HF_TOKEN 등)
├── docs/
│   └── EDU_CAT_EN_V5F_FB_V5R19.pdf  # CATIA V5 338페이지 공식 영문 매뉴얼
├── data/
│   ├── chroma_db/        # 영문 매뉴얼 421개 분할 청크 임베딩 영구 저장소
│   └── evaluation_results.csv  # A/B 테스트 벤치마크 평가 결과 CSV
├── src/
│   ├── config.py         # 환경 설정 및 모델/임베딩 파라미터 정의
│   ├── document_loader.py # PyPDFLoader & RecursiveCharacterTextSplitter
│   ├── vector_store.py   # Chroma Vector DB 구축 및 리트리버 인터페이스
│   ├── rag_chain.py      # Direct LLM / Cross-Lingual RAG / 절차 검증 체인
│   └── evaluation.py     # BERTScore 및 환각률 A/B 테스트 벤치마크 평가 모듈
├── app.py                # Streamlit 데모 웹 애플리케이션 UI
└── run_evaluation.py     # CLI 벤치마크 실행 및 평가 결과 출력 스크립트
```

---

## 실행 가이드

### 1. 환경 준비 및 의존성 설치
```bash
pip install -r requirements.txt
# 주요 패키지: langchain, langchain-openai, langchain-chroma, chromadb, bert-score, streamlit, pypdf
```

### 2. A/B 테스트 벤치마크 및 BERTScore 평가 실행
```bash
python run_evaluation.py
```

### 3. Streamlit 웹 애플리케이션 데모 실행
```bash
streamlit run app.py
```

---

## 📊 정량적 평가 결과 (3-Way Model A/B Testing)

3가지 모델 패러다임에 대해 **BERTScore F1** 및 **폴백(Fallback) 수용도**를 비교 정량 측정한 결과입니다.

| 질문 유형 | 질문 (Question) | Direct LLM (RAG Off) F1 | Strict RAG (매뉴얼 엄격) F1 | Adaptive Fallback RAG F1 | Adaptive RAG 작동 특징 |
|---|---|:---:|:---:|:---:|---|
| **General QA** | CATIA V5 Pad 기능 역할 및 스케치 조건 | 0.7120 | **0.7495** | 0.7325 | ✅ RAG 매뉴얼 근거 답변 (P.53, 55) |
| **Specific QA** | Draft Angle 기능 개념 및 사용 목적 | 0.6571 | **0.7197** | 0.7045 | ✅ RAG 매뉴얼 근거 답변 (P.56, 195) |
| **Trick QA** | CATIA V5 3D 퀀텀 머시닝 AI 가속 모드는? | 0.6137 | **0.8914** | 0.6176 | ⚠️ 매뉴얼 미포함 태그 후 LLM 폴백 |
| **Trick QA** | CATIA V5 자율주행 차체 3D 홀로그램 자동 설계는? | 0.6378 | **0.9008** | 0.6339 | ⚠️ 매뉴얼 미포함 태그 후 LLM 폴백 |
| **평균 (AVG)** | **전체 벤치마크 평균 지표** | **0.6663** | **0.7784** | **0.6757** | **Strict RAG F1 +0.1121 최우수** |

---

## 💡 주요 차별화 성과

1. **Cross-Lingual RAG 성공**: 338페이지의 영문 공식 매뉴얼(`EDU_CAT_EN_V5F_FB_V5R19.pdf`)을 다국어 공간에 임베딩하여, 한국어 질의에 대해 **정확한 한국어 답변 및 참조 페이지(Page Citation)**를 제시함.
2. **3가지 모델 패러다임 비교 구축**:
   - `Direct LLM`: 사전학습 일반 지식만 응답
   - `Strict RAG`: 매뉴얼 엄격 전용 (근거 없을 시 "매뉴얼에 없음" 응답 -> 환각률 0%)
   - `Adaptive Fallback RAG`: 매뉴얼에 정보가 있으면 RAG로 답하고, 없을 경우 `⚠️ [매뉴얼 미포함 - Direct LLM 사전학습 지식 폴백 답변]` 태그를 달아 사전학습 지식으로 답변 전환
3. **현장 작업 절차 정합성 피드백**: 사용자가 작성한 CATIA 작업 절차를 입력받아 표준 매뉴얼과 1:1 디테일 비교하여 **점수(0~100점), 누락 단계, 순서 오류, 개선안**을 생성하는 기능 완성.