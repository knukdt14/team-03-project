### [ 1. 라이브러리 임포트 ] ###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import glob
from langchain_community.document_loaders import PyPDFLoader   ## => lab_06/lab_07과 동일한 로더
from langchain_text_splitters import RecursiveCharacterTextSplitter


### [ 2. 언어 태깅 (한/영 혼합 자료라 실험 분석용으로 남겨둠) ] ###
def detect_lang(text: str) -> str:
    ## => 한글 유니코드 비율로 간단히 판별 (완벽하진 않지만 분석용으론 충분)
    hangul_cnt = len(re.findall(r"[\uac00-\ud7a3]", text))
    return "ko" if hangul_cnt > 20 else "en"


### [ 3. PDF 로드 (파일 1개) ] ###
def load_pdf(pdf_path: str):
    ## => 페이지 단위 Document 리스트로 로드
    ## => 한글 텍스트가 깨져 나오면 PyMuPDFLoader(pip install pymupdf)로 교체 고려
    loader = PyPDFLoader(pdf_path)
    return loader.load()


### [ 3-1. PDF 폴더 전체 로드 (자료가 수십 개일 때) ] ###
def load_pdfs_from_folder(folder_path: str):
    ## => data/ 폴더 안의 *.pdf를 전부 하나로 합쳐서 로드
    ## => 어느 문서·어느 언어에서 나온 답인지 추적하려고 metadata에 남김
    all_pages = []
    for pdf_file in sorted(glob.glob(os.path.join(folder_path, "*.pdf"))):
        pages = load_pdf(pdf_file)
        for p in pages:
            p.metadata["source_file"] = os.path.basename(pdf_file)
            p.metadata["lang"] = detect_lang(p.page_content)
        all_pages.extend(pages)
    if not all_pages:
        raise FileNotFoundError(f"{folder_path} 안에 PDF가 없습니다")
    return all_pages


### [ 4. 청킹 (실험 변수: chunk_size, overlap_size) ] ###
def split_documents(pages, chunk_size: int = 500, overlap_size: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i   ## => lab_07과 동일하게 청크 추적용 ID 부여
    return chunks


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    pages = load_pdfs_from_folder(DEFAULT_CONFIG["pdf_path"])
    chunks = split_documents(pages)
    print(f"페이지 수: {len(pages)} / 청크 수: {len(chunks)}")
    ko_cnt = sum(1 for p in pages if p.metadata["lang"] == "ko")
    print(f"한글 페이지: {ko_cnt} / 영어 페이지: {len(pages) - ko_cnt}")
