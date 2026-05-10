import re
from typing import Callable, Dict, List, Optional, Set, Tuple

from config import SCANNED_PAGE_OVERLAP
from utils import chunk_pages, item_bbox, item_page, is_placeholder
from models import DocumentChunk, ImageChunk, VlmConfig
from vlm import encode_image_jpeg

# Maximum characters per page OCR chunk
PAGE_OCR_CHUNK_MAX_CHARS = 3500


def markdown_headings(text: str) -> List[str]:
    """Extract markdown headings from text."""
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        title = stripped.lstrip("#").strip()
        if title:
            headings.append(title)
        if len(headings) >= 3:
            break
    return headings


def split_page_ocr_text(text: str, max_chars: int = PAGE_OCR_CHUNK_MAX_CHARS) -> List[str]:
    """Split page OCR text into chunks respecting paragraph boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_current():
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for part in parts:
        if len(part) > max_chars:
            flush_current()
            lines = part.splitlines() or [part]
            line_group: List[str] = []
            line_len = 0
            for line in lines:
                addition = len(line) + (1 if line_group else 0)
                if line_group and line_len + addition > max_chars:
                    chunks.append("\n".join(line_group).strip())
                    line_group = []
                    line_len = 0
                line_group.append(line)
                line_len += addition
            if line_group:
                chunks.append("\n".join(line_group).strip())
            continue

        addition = len(part) + (2 if current else 0)
        if current and current_len + addition > max_chars:
            flush_current()
        current.append(part)
        current_len += addition

    flush_current()
    return [chunk for chunk in chunks if chunk]


# ---------------------------------------------------------------------------
# Chunk building
# ---------------------------------------------------------------------------

def build_chunks(
    doc,
    replacements: Dict[str, str],
    hybrid_chunker,
    sem_chunker: Callable[[str], list[str]],
    page_vlm_text: Optional[Dict[int, str]] = None,
    suppress_native_pages: Optional[Set[int]] = None,
) -> Tuple[List[DocumentChunk], Dict[str, int]]:
    """Build DocumentChunks using HybridChunker for digital PDFs, semchunk for scanned pages.

    HybridChunker is used for native PDF text to preserve document structure.
    semchunk is used for VLM-extracted scanned page text (no structure available).
    
    Pages with successful OCR (suppress_native_pages) will have their native chunks suppressed.
    """
    page_vlm_text    = page_vlm_text or {}
    suppress_native_pages = suppress_native_pages or set()
    seen_refs: Set[str] = set()
    chunks: List[DocumentChunk] = []
    pages_covered: Set[int]  = set()

    stats = dict(
        native_chunks_kept=0,
        native_chunks_suppressed=0,
        page_ocr_chunks_emitted=0,
        from_vlm=0,
        skipped_placeholder=0,
        skipped_duplicate=0,
    )

    for chunk in hybrid_chunker.chunk(doc):
        items = list(getattr(chunk.meta, "doc_items", []) or [])
        pages = chunk_pages(chunk) or [1]
        
        # Skip native chunks for pages that had successful OCR
        if suppress_native_pages.intersection(pages):
            stats["native_chunks_suppressed"] += 1
            continue
        
        text, source = _resolve_chunk_text(chunk, items, replacements, seen_refs, stats)

        text = (text or "").strip()
        if not text or is_placeholder(text):
            stats["skipped_placeholder"] += 1
            continue

        headings = list(getattr(chunk.meta, "headings", None) or [])
        pages_covered.update(pages)
        bbox     = next((item_bbox(item) for item in items if item_bbox(item)), None)

        chunks.append(DocumentChunk(text=text, headings=headings, page_numbers=pages, bbox=bbox))
        stats["native_chunks_kept"] += 1

    # Mark picture refs as seen so the scanned-page loop below doesn't
    # double-inject them.  Picture VLM text lives only in image_chunks.
    pic_refs_noted = 0
    for pic in getattr(doc, "pictures", []):
        ref = getattr(pic, "self_ref", None)
        if ref in seen_refs or ref not in replacements:
            continue
        pages_covered.add(item_page(pic))
        seen_refs.add(ref)
        pic_refs_noted += 1

    # Chunk scanned page VLM text using semchunk
    pages_injected = 0
    chunks_injected = 0
    for pg_no, text in sorted(page_vlm_text.items()):
        text = text.strip()
        if not text or is_placeholder(text):
            continue
        pages_covered.add(pg_no)

        # Extract headings from the page text for use in all chunks
        headings = markdown_headings(text)
        
        # Use semchunk to split VLM text into coherent chunks
        page_chunks = sem_chunker(text, overlap=SCANNED_PAGE_OVERLAP)
        for chunk_text in page_chunks:
            chunks.append(DocumentChunk(text=chunk_text, headings=headings, page_numbers=[pg_no], bbox=None))
            stats["page_ocr_chunks_emitted"] += 1
            chunks_injected += 1
        pages_injected += 1

    total_pages = len(getattr(doc, "pages", {}))
    uncovered   = sorted(set(range(1, total_pages + 1)) - pages_covered)

    print(
        f"Chunks: {len(chunks)} total  "
        f"({stats['native_chunks_kept']} from trusted text layer, {stats['page_ocr_chunks_emitted']} from VLM page OCR, "
        f"{pic_refs_noted} picture(s) in image_chunks only)\n"
        f"  page OCR: {pages_injected} pages → {chunks_injected} chunks, "
        f"{stats['native_chunks_suppressed']} native chunk(s) suppressed by page OCR, "
        f"{stats['skipped_placeholder']} empty/placeholder, "
        f"{stats['skipped_duplicate']} duplicate table sub-chunks\n"
        f"  page coverage: {len(pages_covered)}/{total_pages}"
        + (f"  |  no-chunk pages: {uncovered}" if uncovered else "")
    )
    return chunks, stats


def _resolve_chunk_text(chunk, items, replacements, seen_refs, stats):
    """Return (text, source) for one HybridChunker chunk."""
    if len(items) == 1:
        ref = getattr(items[0], "self_ref", None)
        if ref and ref in replacements:
            if ref in seen_refs:
                stats["skipped_duplicate"] += 1
                return None, "duplicate"
            seen_refs.add(ref)
            return replacements[ref], "vlm"

    # Multi-item chunk: mark any replacement refs as seen
    for item in items:
        ref = getattr(item, "self_ref", None)
        if ref and ref in replacements:
            seen_refs.add(ref)

    return chunk.text, "text"


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def extract_images(
    doc,
    replacements: Dict[str, str],
    scanned_pages: Optional[Set[int]] = None,
    skip_pages: Optional[Set[int]] = None,
) -> Tuple[List[ImageChunk], int]:
    """Extract and encode all pictures that are not on scanned or skip pages."""
    scanned_pages = scanned_pages or set()
    skip_pages = skip_pages or set()
    excluded_pages = scanned_pages.union(skip_pages)
    pictures      = getattr(doc, "pictures", [])
    eligible      = [p for p in pictures if item_page(p) not in excluded_pages]
    skipped       = len(pictures) - len(eligible)

    if skipped:
        print(f"Images: extracting {len(eligible)} ({skipped} skipped on scanned/OCR pages)")
    else:
        print(f"Images: extracting {len(eligible)}")

    chunks: List[ImageChunk] = []
    for idx, pic in enumerate(eligible):
        try:
            pil = pic.get_image(doc=doc)
            if pil is None:
                continue

            width, height = pil.size
            b64           = encode_image_jpeg(pil)
            text          = _picture_text(pic, replacements, idx)

            chunks.append(ImageChunk(
                text=text,
                image_base64=b64,
                page_number=item_page(pic),
                bbox=item_bbox(pic),
                width=width,
                height=height,
            ))
        except Exception as exc:
            print(f"  image {idx} skipped: {exc}")

    print(f"Images: extracted {len(chunks)}")
    return chunks, skipped


def _picture_text(pic, replacements: Dict[str, str], idx: int) -> str:
    """Resolve the best available text for a picture (VLM > meta > caption > placeholder)."""
    if pic.self_ref in replacements:
        return replacements[pic.self_ref]

    meta = getattr(pic, "meta", None)
    if meta and getattr(meta, "description", None):
        return str(meta.description.text).strip()

    if getattr(pic, "caption", None):
        return str(pic.caption).strip()

    return f"[Image {idx} on page {item_page(pic)}]"
