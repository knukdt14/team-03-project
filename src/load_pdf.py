import os
import re
import glob
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Try loading PyMuPDF (fitz) with safe fallback
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def detect_lang(text: str) -> str:
    """Detects Korean vs English text ratio."""
    hangul_cnt = len(re.findall(r"[\uac00-\ud7a3]", text))
    return "ko" if hangul_cnt > 20 else "en"


def load_pdf(pdf_path: str) -> List[Document]:
    """Loads single PDF document."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def load_pdfs_from_folder(folder_path: str = DATA_DIR) -> List[Document]:
    """Loads all PDF documents from data/ folder with metadata tagging."""
    all_pages = []
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    for pdf_file in pdf_files:
        pages = load_pdf(pdf_file)
        for p in pages:
            p.metadata["source_file"] = os.path.basename(pdf_file)
            p.metadata["lang"] = detect_lang(p.page_content)
        all_pages.extend(pages)
    if not all_pages:
        raise FileNotFoundError(f"{folder_path} 안에 PDF가 없습니다")
    return all_pages


def split_documents(pages, chunk_size: int = CHUNK_SIZE, overlap_size: int = CHUNK_OVERLAP):
    """Splits documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def extract_multimodal_text_from_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts text, table layouts, and image/drawing tags from a PDF file using PyMuPDF (fitz).
    Falls back to PyPDFLoader if PyMuPDF is not installed.
    """
    filename = os.path.basename(pdf_path)
    
    if not HAS_PYMUPDF:
        return load_pdf(pdf_path)
        
    docs = []
    try:
        pdf_doc = fitz.open(pdf_path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            text = page.get_text("text")
            image_list = page.get_images(full=True)
            img_count = len(image_list)
            
            page_content_parts = []
            if img_count > 0:
                page_content_parts.append(f"🖼️ [매뉴얼 도면/스크린샷 포함: {img_count}개의 CAD 도면 및 설정 창 캡처 그림이 포함된 페이지입니다.]")
                
            if text.strip():
                page_content_parts.append(text.strip())
                
            full_page_text = "\n\n".join(page_content_parts)
            
            if full_page_text.strip():
                doc = Document(
                    page_content=full_page_text,
                    metadata={
                        "source_file": filename,
                        "page": page_num,
                        "has_images": img_count > 0,
                        "image_count": img_count,
                        "lang": detect_lang(full_page_text)
                    }
                )
                docs.append(doc)
        pdf_doc.close()
    except Exception as e:
        print(f"[MultimodalLoader Warning] Could not process {filename} via fitz, falling back: {e}")
        return load_pdf(pdf_path)
        
    return docs


def load_all_multimodal_pdfs(data_dir: str = DATA_DIR) -> List[Document]:
    """
    Scans data/ folder and loads all PDF manuals with multimodal image/layout tags.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at: {data_dir}")
        
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    print(f"[MultimodalLoader] Found {len(pdf_files)} PDF manual files in '{data_dir}'. Extracting multimodal text & diagrams...")
    
    all_raw_docs = []
    for pdf_path in pdf_files:
        raw_docs = extract_multimodal_text_from_pdf(pdf_path)
        all_raw_docs.extend(raw_docs)
        
    print(f"[MultimodalLoader] Total {len(all_raw_docs)} pages extracted across {len(pdf_files)} PDF files.")
    return all_raw_docs


def load_and_split_multimodal_pdf(
    target_path_or_dir: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """Loads PDF manuals from data/ with multimodal layout tags and splits into chunks."""
    target = target_path_or_dir if target_path_or_dir else DATA_DIR
    
    if os.path.isdir(target):
        raw_docs = load_all_multimodal_pdfs(target)
    elif os.path.isfile(target):
        raw_docs = extract_multimodal_text_from_pdf(target)
    else:
        raise FileNotFoundError(f"Target path not found: {target}")
        
    chunks = split_documents(raw_docs, chunk_size=chunk_size, overlap_size=chunk_overlap)
    return chunks


if __name__ == "__main__":
    pages = load_pdfs_from_folder(DATA_DIR)
    chunks = split_documents(pages)
    print(f"페이지 수: {len(pages)} / 청크 수: {len(chunks)}")
