import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_all_pdfs(data_dir: str = DATA_DIR) -> List[Document]:
    """
    Scans data_dir and loads all PDF manual documents.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at: {data_dir}")
    
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    print(f"[DocumentLoader] Found {len(pdf_files)} PDF manual files in '{data_dir}'. Loading...")
    
    all_raw_docs = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
            all_raw_docs.extend(docs)
        except Exception as e:
            print(f"[DocumentLoader Warning] Could not load {filename}: {e}")
            
    print(f"[DocumentLoader] Total {len(all_raw_docs)} pages extracted across {len(pdf_files)} PDF files.")
    return all_raw_docs


def load_and_split_pdf(
    target_path_or_dir: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Loads PDF manuals (all PDFs in data_dir by default) and splits them into text chunks with page & file metadata.
    """
    target = target_path_or_dir if target_path_or_dir else DATA_DIR
    
    if os.path.isdir(target):
        raw_docs = load_all_pdfs(target)
    elif os.path.isfile(target):
        filename = os.path.basename(target)
        loader = PyPDFLoader(target)
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["source_file"] = filename
    else:
        raise FileNotFoundError(f"Target path not found: {target}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(raw_docs)
    print(f"[DocumentLoader] Successfully split into {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    chunks = load_and_split_pdf()
    if chunks:
        print(f"Sample Chunk 0 Metadata: {chunks[0].metadata}")
        print(f"Sample Chunk 0 Content:\n{chunks[0].page_content[:200]}")
