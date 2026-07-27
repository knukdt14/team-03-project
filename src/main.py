### [ 1. 라이브러리 임포트 ] ###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
## => Windows 콘솔 기본 인코딩(cp949)에서는 PDF/LLM 출력에 섞인 유니코드 특수문자(예: 프랑스어 매뉴얼의 ç)를
##    print()할 때 UnicodeEncodeError로 죽는다. stdout/stderr를 UTF-8로 강제해 방지.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
from dotenv import load_dotenv

from config import DEFAULT_CONFIG, EXPERIMENT_PRESETS, PROJECT_ROOT
from load_pdf import load_pdfs_from_folder, split_documents
from build_vectorstore import get_embeddings, build_vectorstore, load_vectorstore
from rag_chain import get_llm, build_rag_chain, build_direct_llm_chain
from evaluate import run_evaluation, run_comparison

load_dotenv()


### [ 2. 벡터스토어 캐시 경로 ] ###
def get_cache_path(cfg: dict):
    ## => chunk_size/overlap/embed 조합이 같으면 벡터스토어를 재사용
    ## => 문서가 수십 개일 때 프리셋마다 매번 재임베딩하지 않도록 캐싱
    key = f"{cfg['chunk_size']}_{cfg['overlap_size']}_{cfg['embed_provider']}_{cfg['embed_model']}_{cfg['vectorstore']}"
    return os.path.join("vecstore_cache", key.replace("/", "-"))


### [ 3. 파이프라인 실행 ] ###
def run_pipeline(cfg: dict):
    embeddings = get_embeddings(cfg["embed_provider"], cfg["embed_model"])
    cache_path = get_cache_path(cfg)

    if os.path.exists(cache_path):
        print(f"기존 벡터스토어 재사용: {cache_path}")
        vecStore = load_vectorstore(embeddings, cfg["vectorstore"], cache_path)
    else:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        pages = load_pdfs_from_folder(cfg["pdf_path"])
        chunks = split_documents(pages, cfg["chunk_size"], cfg["overlap_size"])
        print(f"청크 {len(chunks)}개 (chunk_size={cfg['chunk_size']}, overlap={cfg['overlap_size']}) -> {cache_path}에 저장")
        vecStore = build_vectorstore(chunks, embeddings, cfg["vectorstore"], cache_path)

    llmModel = get_llm(cfg["llm_provider"], cfg["llm_model"])
    chain, retriever = build_rag_chain(
        vecStore, llmModel, cfg["top_k"], cfg["search_type"], cfg["prompt_style"]
    )
    return chain, retriever, llmModel, embeddings


### [ 4. CLI 실행 ] ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="baseline", help="config.py EXPERIMENT_PRESETS 키")
    parser.add_argument("--questions_csv", default=os.path.join(PROJECT_ROOT, "eval", "questions_template.csv"))
    parser.add_argument("--output_csv", default=os.path.join(PROJECT_ROOT, "eval", "results.csv"))
    parser.add_argument("--mode", choices=["eval", "chat", "compare"], default="eval")
    parser.add_argument("--no_ragas", action="store_true", help="RAGAS 평가 건너뛰기 (LLM 호출 여러 번 추가되니 빠른 테스트땐 생략 가능)")
    parser.add_argument("--compare_output_csv", default=os.path.join(PROJECT_ROOT, "eval", "results_compare.csv"))
    args = parser.parse_args()

    cfg = EXPERIMENT_PRESETS.get(args.preset, DEFAULT_CONFIG)
    chain, retriever, llmModel, embeddings = run_pipeline(cfg)

    if args.mode == "eval":
        run_evaluation(chain, retriever, llmModel, embeddings, args.questions_csv,
                        args.output_csv, use_ragas=not args.no_ragas)
    elif args.mode == "compare":
        ## => RAG 적용 vs 미적용(Direct LLM) 대조 — 교수님이 요청한 "RAG 유무 정성적 차이" 근거 자료
        direct_chain = build_direct_llm_chain(llmModel)
        run_comparison(chain, direct_chain, retriever, args.questions_csv, args.compare_output_csv)
    else:
        print("질문 입력 (종료: exit)")
        while True:
            q = input("Q> ")
            if q.strip().lower() == "exit":
                break
            print("A>", chain.invoke(q))
