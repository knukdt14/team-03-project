# fix: HF Inference API 결제오류로 벤치마크 결과 오염 + 임베딩 모델 개선

## 문제

`qwen_7b_api` 프리셋으로 3-Way 벤치마크(Direct/Strict RAG/Adaptive)를 돌렸는데, **7B가 0.5B보다 F1이 낮게(0.5998 vs 0.6518) 나와서 이상함을 느끼고 답변 원문을 직접 확인**함.

## 원인 분석

- `RAGPipeline`의 `answer_direct()`/`answer_rag()`가 API 호출 실패 시 예외를 잡아서 `"⚠️ [응답 오류]: {e}"` 문자열을 **그대로 답변 필드에 저장**함
- 이 에러 문자열이 그대로 BERTScore로 채점되면서, **말이 되는 것처럼 보이는 가짜 F1 점수**가 나옴
- 실제 확인해보니 5문항 벤치마크(호출 15회) 중 **Direct 3/5, Strict RAG 3/5, Adaptive 4/5가 `402 Payment Required`로 실패**
- HuggingFace 토큰의 "Fine-grained" 권한에 **"Make calls to Inference Providers"가 기본적으로 꺼져있어서** 처음엔 403(권한 부족)이 났고, 권한 켠 뒤엔 402(결제 필요)로 바뀜 — 즉 **무료 크레딧이 벤치마크 도중 소진**된 것으로 보임
- 단발 테스트 호출(`llm.invoke("안녕...")`)은 성공해서 "되는구나" 착각하기 쉬운데, 실제 5문항×3모드 벤치마크처럼 **호출 수가 많아지면 무료 크레딧이 중간에 바닥남**

## 해결

1. `report/bench_qwen7b_markdown_multilingual.csv`를 **무효 데이터로 표시**, 최종 리포트에서 제외
2. Llama-3.1-8B API는 접근 권한(토큰 설정)까지는 확인했으나, **동일한 결제 벽 위험 때문에 전체 벤치마크는 실행하지 않기로 결정** (`report/EMBEDDING_TUNING_REPORT.md` 부록 B 참고)
3. 로컬 모델(0.5B/1.5B/3B) 기준으로 다국어 임베딩(`paraphrase-multilingual-MiniLM-L12-v2`) 옵션 추가 — 영어 전용 임베딩(`all-MiniLM-L6-v2`) 대비 전 모델에서 개선 확인:
   - 0.5B: 0.6105 → 0.6518 (+0.041)
   - 3B: 0.8105 → 0.8228 (+0.012)
4. `src/vector_store.py`의 `get_embedding_function()`이 `device='cpu'`로 하드코딩되어 RTX 4070을 안 쓰고 있던 것도 같이 수정 (GPU 자동 감지)

## 진행 중 발견한 부가 이슈

- **API 벤치마크는 반드시 답변 원문을 직접 열어서 확인할 것** — F1 숫자만 보면 결제 오류를 놓치기 쉬움 (숫자 자체는 "그럴듯하게" 나옴)
- `vect/chroma_db_multimodal` git 추적 손상 버그(Racoon7828님 리포트)를 이 컴퓨터에서도 재현 확인 → main의 `fe5027b` 수정이 이미 반영된 것 확인, 로컬 벡터스토어 클린 재빌드함
- 모델 크기별 성능: 0.5B(0.6518) → 1.5B(0.7150) → 3B(0.8228)로 일관 상승, 트릭 질문은 3B에서 F1 1.0000(완전 환각 차단) 달성

## 변경 파일

- `src/vector_store.py`: `get_embedding_function()` GPU 자동 감지, `build_or_load_vectorstore()`에 `embed_model` 파라미터 추가
- `src/config.py`: `MULTILINGUAL_EMBEDDING_MODEL`, `CHROMA_DB_DIR_MARKDOWN_ML`, `qwen_0.5b_local` 프리셋 추가
- `src/build_markdown_multilingual_vectorstore.py` (신규)
- `report/EMBEDDING_TUNING_REPORT.md` (신규): 전체 실험 상세 기록
- `report/bench_qwen{05b,15b,3b}_markdown_multilingual.csv`: 로컬 모델 3종 정식 벤치마크 결과
- `report/bench_qwen7b_markdown_multilingual.csv`: ⚠️ 결제오류로 무효, 참고용으로만 보존

## 남은 작업 / 확인 필요

- [ ] HuggingFace 유료 크레딧 확보 후 7B/Llama-8B API 재측정
- [ ] 5문항은 표본이 작음 — 61문항 등 확장된 평가셋으로 재검증 권장
- [ ] 3B + 마크다운 + 영어전용(all-MiniLM) 조합(0.8105)이 vect/ 손상 발견 **이전**에 측정된 값이라 재검증 필요
- [ ] 팀원이 공유한 고난도 17문항(정답 없음)에 정답 세트 만들어서 정식 평가셋에 편입
