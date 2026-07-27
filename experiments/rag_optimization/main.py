### [ 1. 라이브러리 임포트 ] ###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from dotenv import load_dotenv

from config import DEFAULT_CONFIG, EXPERIMENT_PRESETS, PROJECT_ROOT
from load_pdf import load_pdfs_from_folder, split_documents
from build_vectorstore import get_embeddings, build_vectorstore, load_vectorstore
from rag_chain import get_llm, build_rag_chain, build_bm25_retriever, build_reranker
from evaluate import run_evaluation

load_dotenv()


### [ 2. 벡터스토어 캐시 경로 ] ###
def get_cache_path(cfg: dict):
    ## => chunk_size/overlap/embed 조합이 같으면 벡터스토어를 재사용
    ## => 문서가 수십 개일 때 프리셋마다 매번 재임베딩하지 않도록 캐싱
    key = f"{cfg.get('corpus_tag', 'base')}_{cfg['chunk_size']}_{cfg['overlap_size']}_{cfg['embed_provider']}_{cfg['embed_model']}_{cfg['vectorstore']}"
    return os.path.join(PROJECT_ROOT, "vecstore_cache", key.replace("/", "-"))


### [ 3. 파이프라인 실행 ] ###
def run_pipeline(cfg: dict):
    embeddings = get_embeddings(cfg["embed_provider"], cfg["embed_model"])
    cache_path = get_cache_path(cfg)

    chunks = None
    if os.path.exists(cache_path):
        print(f"?? ????? ???: {cache_path}")
        vecStore = load_vectorstore(embeddings, cfg["vectorstore"], cache_path)
    else:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        pages = load_pdfs_from_folder(cfg["pdf_path"])
        chunks = split_documents(pages, cfg["chunk_size"], cfg["overlap_size"])
        print(f"?? {len(chunks)}? (chunk_size={cfg['chunk_size']}, overlap={cfg['overlap_size']}) -> {cache_path}? ??")
        vecStore = build_vectorstore(chunks, embeddings, cfg["vectorstore"], cache_path)

    bm25_retriever = None
    if cfg.get("retrieval_mode") in {"bm25", "hybrid_rrf"}:
        if chunks is None:
            pages = load_pdfs_from_folder(cfg["pdf_path"])
            chunks = split_documents(pages, cfg["chunk_size"], cfg["overlap_size"])
        bm25_retriever = build_bm25_retriever(chunks, cfg.get("bm25_k", cfg["top_k"]))

    reranker = build_reranker(cfg["reranker_model"]) if cfg.get("reranker_model") else None
    llmModel = get_llm(cfg["llm_provider"], cfg["llm_model"], cfg.get("max_new_tokens"))
    chain, retriever = build_rag_chain(
        vecStore, llmModel, cfg["top_k"], cfg["search_type"], cfg["prompt_style"], cfg.get("score_threshold"), cfg.get("fetch_k"), cfg.get("retrieval_mode", "vector"), bm25_retriever, reranker, cfg.get("candidate_k")
    )
    return chain, retriever, llmModel, embeddings


### [ 4. CLI 실행 ] ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="baseline", help="config.py EXPERIMENT_PRESETS 키")
    parser.add_argument("--questions_csv", default=os.path.join(PROJECT_ROOT, "eval", "questions_template.csv"))
    parser.add_argument("--output_csv", default=os.path.join(PROJECT_ROOT, "eval", "results.csv"))
    parser.add_argument("--mode", choices=["eval", "chat"], default="eval")
    parser.add_argument("--no_ragas", action="store_true", help="RAGAS 평가 건너뛰기 (LLM 호출 여러 번 추가되니 빠른 테스트땐 생략 가능)")
    args = parser.parse_args()

    cfg = EXPERIMENT_PRESETS.get(args.preset, DEFAULT_CONFIG)
    chain, retriever, llmModel, embeddings = run_pipeline(cfg)

    if args.mode == "eval":
        run_evaluation(chain, retriever, llmModel, embeddings, args.questions_csv,
                        args.output_csv, use_ragas=not args.no_ragas)
    else:
        print("질문 입력 (종료: exit)")
        while True:
            q = input("Q> ")
            if q.strip().lower() == "exit":
                break
            print("A>", chain.invoke(q))
