import os
import sys

# Import functions from src.load_pdf for full compatibility
from src.load_pdf import (
    extract_multimodal_text_from_pdf,
    load_all_multimodal_pdfs,
    load_and_split_multimodal_pdf,
    load_pdfs_from_folder,
    split_documents,
    detect_lang
)

if __name__ == "__main__":
    chunks = load_and_split_multimodal_pdf()
    if chunks:
        print(f"Sample Multimodal Chunk 0 Metadata: {chunks[0].metadata}")
        print(f"Sample Chunk 0 Content:\n{chunks[0].page_content[:300]}")
