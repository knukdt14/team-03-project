import os
import re
import glob
import json
import numpy as np
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.config import DATA_DIR, VECT_DIR, CHUNK_SIZE, CHUNK_OVERLAP

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Try loading PyMuPDF (fitz) with safe fallback
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Try loading pymupdf4llm (PDF -> 구조 보존 Markdown 변환) with safe fallback
try:
    import pymupdf4llm
    HAS_PYMUPDF4LLM = True
except ImportError:
    HAS_PYMUPDF4LLM = False

EXTRACTED_IMAGE_DIR = os.path.join(VECT_DIR, "extracted_images")
OCR_CACHE_PATH = os.path.join(VECT_DIR, "ocr_texts.json")

## => OCR 캐시: build_ocr_cache.py가 만든 {이미지파일명: 이미지 속 텍스트} 매핑.
##    영문 실습 매뉴얼은 대화상자 탭명/옵션명 같은 핵심 정보가 스크린샷 안에만 있어서,
##    이 텍스트를 청크에 합쳐줘야 검색이 가능해진다. 캐시가 없으면 기존 동작 그대로.
_ocr_cache = None


def _load_ocr_cache() -> dict:
    global _ocr_cache
    if _ocr_cache is None:
        if os.path.exists(OCR_CACHE_PATH):
            with open(OCR_CACHE_PATH, encoding="utf-8") as f:
                _ocr_cache = json.load(f)
        else:
            _ocr_cache = {}
    return _ocr_cache


def _ocr_text_for(image_path: str) -> str:
    return _load_ocr_cache().get(os.path.basename(image_path), "")


_WORD_RE = re.compile(r"[A-Za-z가-힣0-9]{3,}")


def _keyword_overlap(text_a: str, text_b: str) -> int:
    """두 텍스트가 공유하는 단어(3자 이상, 대소문자 무시) 수."""
    words_a = {w.lower() for w in _WORD_RE.findall(text_a)}
    words_b = {w.lower() for w in _WORD_RE.findall(text_b)}
    return len(words_a & words_b)


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


def extract_images_from_page(pdf_doc, page, page_num: int, filename: str, output_dir: str = EXTRACTED_IMAGE_DIR) -> List[dict]:
    """페이지에 삽입된 이미지들을 파일로 추출하고, 페이지 내 위치(rect)를 함께 반환한다.
    소프트 마스크(SMask, 알파)가 따로 있으면 합성해서 저장한다 -- 이 매뉴얼들의 아이콘/스크린샷
    상당수가 '본체 이미지 + 별도 SMask' 형태로 저장돼 있어서, SMask를 빼먹으면 검은 사각형으로 보인다."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(filename)[0]
    images = []
    for img_idx, img in enumerate(page.get_images(full=True)):
        xref, smask_xref = img[0], img[1]
        try:
            pix = fitz.Pixmap(pdf_doc, xref)
            if pix.colorspace and pix.colorspace.n >= 4:  ## => CMYK -> RGB 변환 (PNG는 CMYK 저장 불가)
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if smask_xref:
                if pix.alpha:  ## => Pixmap(base, mask) 합성은 base에 알파 채널이 없어야 함
                    pix = fitz.Pixmap(pix, 0)
                mask = fitz.Pixmap(pdf_doc, smask_xref)
                pix = fitz.Pixmap(pix, mask)
            ## => 얇은 구분선/장식용 이미지(가로선 등)는 리사이즈 시 "height/width must be > 0" 에러를 유발 -> 제외
            if pix.width <= 8 or pix.height <= 8:
                continue
            if pix.alpha:
                ## => 페이지 하단 워터마크/배경 장식은 알파(불투명도)가 거의 0으로 설정되어 사실상 안 보이게
                ##    디자인된 것 -> 평균 알파가 매우 낮으면(10% 미만) 눈에 안 보이는 장식으로 보고 제외
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                mean_alpha = arr[:, :, -1].mean() / 255.0
                if mean_alpha < 0.1:
                    continue
            out_path = os.path.join(output_dir, f"{base_name}_p{page_num + 1}_{img_idx}.png")
            pix.save(out_path)
            rects = page.get_image_rects(xref)
            images.append({"path": out_path, "rect": rects[0] if rects else None})
        except Exception:
            continue
    return images


def _is_text_duplicate_image(img_rect, blocks: list, overlap_threshold: float = 0.1) -> bool:
    """일부 PDF는 제목/키워드 같은 텍스트를 실제 텍스트가 아니라 이미지로 별도 렌더링해서 겹쳐놓는 경우가 있음
    (예: "Coincidence" 캡션 옆에 똑같은 "Coincidence" 글자를 이미지로도 박아놓음).
    이런 이미지는 진짜 텍스트 블록과 대부분 겹치는 반면, 실제 CAD 스크린샷/도면은 빈 공간에 배치되어
    텍스트 블록과 거의 겹치지 않음 -> 겹침 비율로 구분."""
    img_area = max(img_rect.width * img_rect.height, 1)
    for block in blocks:
        bx0, by0, bx1, by1 = block[:4]
        ix0, iy0 = max(img_rect.x0, bx0), max(img_rect.y0, by0)
        ix1, iy1 = min(img_rect.x1, bx1), min(img_rect.y1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter_area = (ix1 - ix0) * (iy1 - iy0)
        if inter_area / img_area >= overlap_threshold:
            return True
    return False


def _assign_images_to_nearest_block(images: List[dict], blocks: list) -> dict:
    """각 이미지를 설명하는 텍스트 블록 하나에 배정한다.

    1차 기준: OCR 키워드 겹침 -- 이미지 속 텍스트(예: 대화상자의 "Hole Definition")와
    같은 단어를 가장 많이 공유하는 블록. 거리만으로는 헷갈리는 배치도 내용으로 정확히 붙는다.
    2차 기준(글자 없는 이미지 또는 겹침 0): 기존 방식대로 세로 중심 거리가 가장 가까운 블록.
    매뉴얼이 대부분 단일 컬럼이라, 세로 위치가 가까운 문단일수록 그 이미지를 설명할 가능성이 높음."""
    assignment = {i: [] for i in range(len(blocks))}
    for img in images:
        rect = img.get("rect")
        if rect is None or not blocks:
            continue
        img_center_y = (rect.y0 + rect.y1) / 2
        dists = [abs(img_center_y - (b[1] + b[3]) / 2) for b in blocks]
        best_idx = min(range(len(blocks)), key=lambda i: dists[i])

        ocr_text = _ocr_text_for(img["path"])
        if ocr_text:
            overlaps = [_keyword_overlap(ocr_text, b[4]) for b in blocks]
            max_overlap = max(overlaps)
            if max_overlap >= 1:
                ## => 겹침이 같으면 그중 거리가 가까운 블록을 선택
                candidates = [i for i, o in enumerate(overlaps) if o == max_overlap]
                best_idx = min(candidates, key=lambda i: dists[i])

        assignment[best_idx].append(img["path"])
    return assignment


def _group_blocks_into_chunks(blocks_with_images: list, chunk_size: int) -> list:
    """읽는 순서대로 연속된 텍스트 블록들을 chunk_size 정도 크기로 묶는다.
    이때 이미지는 그 청크에 실제로 포함된 블록에 배정된 것만 함께 딸려간다."""
    ## => 슬라이드형 매뉴얼은 페이지 전체 글자 수가 chunk_size보다 훨씬 작아서, 글자 수 기준만으로는
    ##    한 페이지 안의 서로 다른 주제(예: Coincidence 문단 vs Concentricity 문단)가 한 청크로
    ##    합쳐져 버림 -> "새 블록이 이미 그룹에 없는 자기만의 이미지를 데려오면" 그걸 주제 전환 신호로 보고 끊음
    groups = []
    cur_texts, cur_images, cur_len = [], [], 0
    cur_has_own_image = False
    for text, image_paths in blocks_with_images:
        exceeds_size = cur_texts and cur_len + len(text) > chunk_size
        starts_new_topic = cur_has_own_image and any(p not in cur_images for p in image_paths)
        if cur_texts and (exceeds_size or starts_new_topic):
            groups.append((cur_texts, cur_images))
            cur_texts, cur_images, cur_len = [], [], 0
            cur_has_own_image = False
        cur_texts.append(text)
        if image_paths:
            cur_has_own_image = True
            for p in image_paths:
                if p not in cur_images:
                    cur_images.append(p)
        cur_len += len(text)
    if cur_texts:
        groups.append((cur_texts, cur_images))
    return groups


MIN_CHUNK_TEXT_LEN = 40  ## => 이보다 텍스트가 짧으면 "제목/캡션만 있고 실제 정의는 이미지 안에만 있는" 얇은 청크로 간주


def _merge_thin_groups(groups: list, min_text_len: int = MIN_CHUNK_TEXT_LEN) -> list:
    """제목/캡션 한 줄만 있고 본문이 거의 없는 얇은 청크(예: "3) Concentricity"만 있고 실제 정의는
    스크린샷 이미지 안에만 있는 경우)가 정보량 없이 검색 상위에 노이즈로 뜨는 문제(GitHub 이슈 #16) 완화.
    이미지 자체의 텍스트(OCR)까지 뽑는 건 아니지만, 최소한 텍스트 없는 청크가 혼자 매칭되는 건 막는다.
    다음 그룹과 합쳐서 문맥을 붙이고, 문서 마지막 그룹이 얇으면 바로 이전 그룹에 흡수시킨다."""
    if not groups:
        return groups

    merged = []
    pending_texts, pending_images = [], []
    for texts, images in groups:
        combined_texts = pending_texts + texts
        combined_images = pending_images + [p for p in images if p not in pending_images]
        text_len = sum(len(t) for t in combined_texts)
        if text_len < min_text_len:
            pending_texts, pending_images = combined_texts, combined_images
            continue
        merged.append((combined_texts, combined_images))
        pending_texts, pending_images = [], []

    if pending_texts:
        if merged:
            last_texts, last_images = merged[-1]
            merged[-1] = (
                last_texts + pending_texts,
                last_images + [p for p in pending_images if p not in last_images],
            )
        else:
            merged.append((pending_texts, pending_images))
    return merged


def extract_multimodal_text_from_pdf(pdf_path: str, chunk_size: int = CHUNK_SIZE) -> List[Document]:
    """
    Extracts text and embedded images from a PDF using PyMuPDF (fitz), binding each image to
    its nearest text block (paragraph) by on-page position -- so retrieval only surfaces images
    that are actually near the retrieved text, not every image that happens to share the page.
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

            raw_blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
            raw_blocks.sort(key=lambda b: (round(b[1]), b[0]))  ## => 읽는 순서: 위->아래, 같은 줄이면 왼쪽->오른쪽

            page_images = extract_images_from_page(pdf_doc, page, page_num, filename)
            page_images = [
                img for img in page_images
                if img["rect"] is None or not _is_text_duplicate_image(img["rect"], raw_blocks)
            ]
            image_assignment = _assign_images_to_nearest_block(page_images, raw_blocks)

            blocks_with_images = [
                (block[4].strip(), image_assignment.get(i, []))
                for i, block in enumerate(raw_blocks)
            ]
            if not blocks_with_images:
                continue

            page_groups = _merge_thin_groups(_group_blocks_into_chunks(blocks_with_images, chunk_size))
            for texts, image_paths in page_groups:
                chunk_text = "\n\n".join(texts)
                if image_paths:
                    chunk_text = f"🖼️ [이 문단 근처 CAD 도면/스크린샷 {len(image_paths)}개 포함]\n\n{chunk_text}"
                    ## => 스크린샷 속 대화상자 탭명/옵션명/수치는 텍스트 레이어에 없음 ->
                    ##    OCR로 뽑은 텍스트를 청크 본문에 합쳐 검색 대상으로 만든다 (과도한 청크 비대 방지 위해 500자 제한)
                    ocr_snippets = [t for t in (_ocr_text_for(p) for p in image_paths) if t]
                    if ocr_snippets:
                        chunk_text += f"\n\n[이미지 속 텍스트(OCR): {' / '.join(ocr_snippets)[:500]}]"

                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": filename,
                        "page": page_num,
                        "has_images": len(image_paths) > 0,
                        "image_count": len(image_paths),
                        "image_paths": ";".join(image_paths),  ## => Chroma 메타데이터는 스칼라만 허용 -> 세미콜론 구분 문자열
                        "lang": detect_lang(chunk_text)
                    }
                )
                docs.append(doc)
        pdf_doc.close()
    except Exception as e:
        print(f"[MultimodalLoader Warning] Could not process {filename} via fitz, falling back: {e}")
        return load_pdf(pdf_path)

    return docs


def load_all_multimodal_pdfs(data_dir: str = DATA_DIR, chunk_size: int = CHUNK_SIZE) -> List[Document]:
    """
    Scans data/ folder and loads all PDF manuals with multimodal image/layout tags.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at: {data_dir}")

    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    print(f"[MultimodalLoader] Found {len(pdf_files)} PDF manual files in '{data_dir}'. Extracting multimodal text & diagrams...")

    all_raw_docs = []
    for pdf_path in pdf_files:
        raw_docs = extract_multimodal_text_from_pdf(pdf_path, chunk_size=chunk_size)
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
        raw_docs = load_all_multimodal_pdfs(target, chunk_size=chunk_size)
    elif os.path.isfile(target):
        raw_docs = extract_multimodal_text_from_pdf(target, chunk_size=chunk_size)
    else:
        raise FileNotFoundError(f"Target path not found: {target}")

    ## => raw_docs는 이미 문단(블록) 단위로 chunk_size에 맞춰 그룹핑되어 있음 -> 대부분 그대로 통과,
    ##    드물게 한 문단 자체가 chunk_size보다 크면 여기서 추가로 쪼개짐 (그 경우 이미지도 그대로 상속)
    chunks = split_documents(raw_docs, chunk_size=chunk_size, overlap_size=chunk_overlap)
    return chunks


### [ 5. Markdown 변환 로더 (실험: PDF 구조(제목/표/리스트) 보존 후 청킹) ] ###
## => PyPDFLoader/PyMuPDF의 순수 텍스트 추출은 헤더·표 구조가 사라져서,
##    RecursiveCharacterTextSplitter가 서로 다른 주제를 한 청크에 섞어버리는 문제가 있었음
##    (report/EMBEDDING_TUNING_REPORT.md 참고: chunk_size를 줄여도 특정 사실이 top-k 밖으로 밀리는 현상 확인).
##    pymupdf4llm으로 헤더(#, ##, ###)를 보존한 Markdown으로 뽑고,
##    MarkdownHeaderTextSplitter로 헤더 경계부터 먼저 나눈 뒤 chunk_size로 재분할하면
##    한 청크 안에 여러 주제가 섞이는 걸 줄일 수 있음.
def extract_markdown_text_from_pdf(pdf_path: str) -> List[Document]:
    """PDF를 페이지 단위 Markdown(제목/표/리스트 구조 보존)으로 추출. pymupdf4llm 미설치 시 일반 텍스트로 폴백."""
    filename = os.path.basename(pdf_path)

    if not HAS_PYMUPDF4LLM:
        return load_pdf(pdf_path)

    try:
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    except Exception as e:
        print(f"[MarkdownLoader Warning] Could not process {filename} via pymupdf4llm, falling back: {e}")
        return load_pdf(pdf_path)

    docs = []
    for p in md_pages:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        page_number_1indexed = p.get("metadata", {}).get("page_number", 1)
        docs.append(Document(
            page_content=text,
            metadata={
                "source_file": filename,
                "page": page_number_1indexed - 1,  ## => PyPDFLoader와 동일하게 0-indexed로 통일
                "lang": detect_lang(text),
            }
        ))
    return docs


def load_all_markdown_pdfs(data_dir: str = DATA_DIR) -> List[Document]:
    """data/ 폴더 내 모든 PDF를 Markdown 구조 보존 방식으로 로드."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} 안에 PDF가 없습니다")

    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    print(f"[MarkdownLoader] {len(pdf_files)}개 PDF를 Markdown으로 변환 중...")
    all_docs = []
    for pdf_file in pdf_files:
        all_docs.extend(extract_markdown_text_from_pdf(pdf_file))
    print(f"[MarkdownLoader] 총 {len(all_docs)}페이지 추출 완료")
    return all_docs


def split_markdown_documents(pages: List[Document], chunk_size: int = CHUNK_SIZE, overlap_size: int = CHUNK_OVERLAP) -> List[Document]:
    """Markdown 헤더 경계를 우선 존중해서 분할한 뒤, chunk_size 기준으로 재분할."""
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for page in pages:
        try:
            sections = header_splitter.split_text(page.page_content)
        except Exception:
            sections = [Document(page_content=page.page_content, metadata={})]

        for section in sections:
            section.metadata = {**page.metadata, **section.metadata}  ## => source_file/page/lang + 헤더 정보(h1~h4) 함께 보존
            all_chunks.extend(char_splitter.split_documents([section]))

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i
    return all_chunks


def load_and_split_markdown_pdf(
    target_path_or_dir: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """data/ PDF를 Markdown 구조 보존 방식으로 로드하고 청킹."""
    target = target_path_or_dir if target_path_or_dir else DATA_DIR

    if os.path.isdir(target):
        raw_docs = load_all_markdown_pdfs(target)
    elif os.path.isfile(target):
        raw_docs = extract_markdown_text_from_pdf(target)
    else:
        raise FileNotFoundError(f"Target path not found: {target}")

    return split_markdown_documents(raw_docs, chunk_size=chunk_size, overlap_size=chunk_overlap)


if __name__ == "__main__":
    pages = load_pdfs_from_folder(DATA_DIR)
    chunks = split_documents(pages)
    print(f"페이지 수: {len(pages)} / 청크 수: {len(chunks)}")
