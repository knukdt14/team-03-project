# RAG 검색 최적화 실험

기존 `src/`의 Streamlit·멀티모달 기능을 변경하지 않고, 이 폴더에는
문서 기반 CATIA 질의응답의 검색 성능을 개선하기 위한 재현 실험 코드를 둡니다.

## 현재 선두 설정

- 임베딩: `BAAI/bge-m3`
- 벡터 DB: Chroma
- 청크: 500, overlap: 100
- 검색: MMR, `top_k=6`
- 생성: 문서 근거 기반 strict prompt + 출처 표기

40문항에서 BERTScore F1 `0.7597`, Faithfulness `0.7737`, 평균 응답시간
`0.88초`, 수동 환각률 `7.5% (3/40)`를 기록했습니다.

## 실행

저장소 루트에서 실행합니다.

```powershell
pip install -r requirements.txt -r experiments/rag_optimization/requirements.txt
```

```powershell
python experiments/rag_optimization/compare_presets.py --embedding_only --no_ragas
python experiments/rag_optimization/main.py --preset bge_m3_chroma_mmr_500_strict --output_csv eval/results_bge_m3_chroma_mmr_500_strict.csv
```

`eval/questions_template.csv`는 확장된 40문항 평가셋이고,
`eval/questions_template_baseline_10.csv`에는 기존 10문항을 보존했습니다.

모델 캐시, 벡터스토어 캐시, API 키는 Git에 포함하지 않습니다.
