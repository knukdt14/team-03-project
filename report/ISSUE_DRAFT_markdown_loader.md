# fix: PDF→Markdown 로더 추가 및 로컬 파이프라인 실행 중 발견한 버그 수정

## 문제

교수님 피드백 대응(RAG 적용 전/후 정성적 차이 입증) 작업을 하다가, main 최신 코드를 실제로 로컬에서 처음부터 끝까지(패키지 설치 → 벡터DB 빌드 → 로컬 Qwen 모델 로딩 → 답변 생성)로 돌려본 적이 없다는 걸 확인함. 실제로 돌려보니 여러 군데서 조용히 fallback되거나 죽는 지점이 있었음.

## 원인 분석

기존 `src/load_pdf.py`의 멀티모달 로더(`extract_multimodal_text_from_pdf`)는 PyPDFLoader/PyMuPDF로 텍스트만 뽑고, 이미지가 있으면 `🖼️ [N개 포함]` 태그만 페이지 앞에 붙이는 구조였음.

- 헤더/표/리스트 같은 문서 구조 정보가 전부 사라진 채로 `RecursiveCharacterTextSplitter`에 들어가서, 한 청크 안에 서로 다른 주제가 섞이는 문제가 있었음
- 이미지 개수 태그만 있고 본문이 짧은 페이지(예: "🖼️ 4개 CAD 도면 포함...\n\n6\n실습5")가 정보량이 적어서 임베딩 공간에서 여러 무관한 질문에 대해 최상위로 검색되는 "노이즈 자석" 현상 발견 (한국어 질문, 영어 질문 둘 다에서 동일 청크가 1위로 잡힘)
- `requirements.txt`에 `accelerate`가 빠져 있어서 `LLMFactory`가 로컬 GPU 모델 로딩을 시도하다 조용히 API 모드로 폴백 → API 모드도 `HF_TOKEN` 미설정으로 실패 → 결국 `⚠️ [RAG 응답 생성 오류]: Cannot select auto-router...`로 답변 자체가 안 나가는 상태였음 (에러 메시지만 봐서는 원인 파악이 어려움)
- `rag_chain.py`의 `expand_query()`(한국어 질문을 CATIA 영문 용어로 확장)가 가끔 엉뚱한 용어를 붙이는 것도 확인함 (예: "솔리드 파트는 어떤 확장자로 저장되나요?" → 뒤에 SolidWorks 확장자인 `.sldprt`를 붙임 — CATIA는 `.CATPart`인데 오히려 헷갈리는 방향으로 확장됨)

## 해결

1. `pymupdf4llm`(이미 쓰고 있던 PyMuPDF 계열)으로 PDF를 헤더(`#`/`##`/`###`)·표·리스트 구조가 살아있는 Markdown으로 추출하는 `extract_markdown_text_from_pdf`/`load_and_split_markdown_pdf`를 `src/load_pdf.py`에 추가
2. `MarkdownHeaderTextSplitter`로 헤더 경계부터 먼저 나눈 뒤 `chunk_size`로 재분할 → 청크에 헤더 정보(`h1`~`h4`)가 메타데이터로 남아서 주제가 덜 섞임
3. `src/vector_store.py`의 `build_or_load_vectorstore()`에 `loader="markdown"` 옵션 추가 (기존 `loader="multimodal"`은 그대로 default 유지, 기존 파이프라인 안 건드림)
4. 기존 벡터DB(`vect/chroma_db_multimodal`)와 별도로 `vect/chroma_db_markdown`에 저장해서 나란히 비교 가능하도록 `src/build_markdown_vectorstore.py` 신규 스크립트 추가
5. `requirements.txt`에 `pymupdf4llm`, `accelerate` 추가

## 진행 중 발견한 부가 버그 (같이 수정)

- **`accelerate` 누락**: 로컬 GPU 모델 로딩(`device_map="auto"`)에 필수인데 `requirements.txt`에 없어서, 새 환경에서 `pip install -r requirements.txt`만 하면 로컬 모드가 조용히 API 모드로 깨지는 문제 → `requirements.txt`에 추가
- **이미지 태그 전용 청크의 검색 노이즈**: 정보량 적은 태그-only 청크가 여러 무관한 질문에서 최상위로 잡히는 현상 확인 (마크다운 로더는 실제 본문 구조를 보존해서 이 문제가 없음)
- **`expand_query()` 오확장 사례**: CATIA 질문에 SolidWorks 확장자(`.sldprt`)를 붙이는 등, 쿼리 확장이 항상 도움이 되는 건 아니라는 반례 발견 — 검색 전/후 비교 로깅을 추가하면 도움이 될 것 같음 (별도 이슈로 분리 제안)
- **임베딩 모델 언어 한계**: 현재 `sentence-transformers/all-MiniLM-L6-v2`는 사실상 영어 전용 모델이라, 다국어 임베딩(`paraphrase-multilingual-MiniLM-L12-v2`)으로 바꿔서 같은 마크다운 청크로 재검색해봐도 일부 한국어 질문은 여전히 실패함 → 마크다운/청킹보다 **임베딩 모델 선택이 한국어 질의 검색 품질의 더 근본적인 병목**일 가능성이 큼 (별도 이슈로 분리 제안, `EXPERIMENTS.md` 참고)

## 변경 파일

- `src/load_pdf.py` (+90여 줄): `extract_markdown_text_from_pdf`, `load_all_markdown_pdfs`, `split_markdown_documents`, `load_and_split_markdown_pdf` 추가
- `src/vector_store.py` (+6/-4): `build_or_load_vectorstore()`에 `loader` 파라미터 추가
- `src/build_markdown_vectorstore.py` (신규): 마크다운 벡터DB 빌드 진입점
- `src/config.py` (+1): `CHROMA_DB_DIR_MARKDOWN` 경로 추가
- `requirements.txt` (+2): `pymupdf4llm`, `accelerate` 추가
- `EXPERIMENTS.md`: 이번 실험 과정/원인 분석 기록

## 정량 비교 (공식 3-Way 벤치마크, `run_evaluation_benchmark`)

| 벡터스토어 | Strict RAG 평균 F1 |
|---|---|
| 기존(multimodal, 이미지 태그) | 0.8071 |
| **마크다운(pymupdf4llm)** | **0.8105** (+0.0034, 퇴보 없음) |

| 질문 | 기존 F1 | 마크다운 F1 |
|---|---:|---:|
| General QA (Pad 기능) | 0.6613 | 0.6691 ✅ |
| Specific QA (Stiffener) | 0.6533 | 0.6533 (동일) |
| Specific QA (Draft Angle) | 0.7211 | 0.7300 ✅ |
| Trick QA ×2 (가상 기능) | 1.0000 | 1.0000 (동일, 둘 다 완벽 거절) |

전 카테고리에서 퇴보 없이 general/specific 일부 항목이 개선됨. 원본 CSV: `report/bench_old_multimodal_summary.csv`, `report/bench_markdown.csv`

⚠️ 참고: 첫 벤치마크 시도 때 한 스크립트 안에서 Qwen 모델을 두 번(기존용 → 마크다운용) 연달아 로드했다가 VRAM이 부족해져 일부 레이어가 CPU로 offload되면서 극도로 느려지는 현상 발생(8GB GPU 기준). 두 벤치마크는 반드시 별도 프로세스로 분리 실행해야 함.

## 남은 작업 / 확인 필요

- [x] 마크다운 벡터스토어 정식 벤치마크 결과로 표 채우기
- [ ] `expand_query()` 오확장 사례 재현 케이스 모아서 별도 이슈로 분리
- [ ] 임베딩 모델을 다국어 모델로 교체하는 실험 (별도 이슈로 분리 제안)
- [ ] 팀원 검토 후 `loader="markdown"`을 default로 승격할지 결정
- [ ] 전체 32개 PDF 재빌드 후 실제 UI(`app.py`)에서 이미지 관련성 육안 확인
