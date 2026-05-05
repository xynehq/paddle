from typing import Dict, List, Optional, Set

from utils import chunk_pages, item_bbox, item_page, is_placeholder
from models import DocumentChunk, ImageChunk, VlmConfig
from vlm import encode_image_jpeg


# ---------------------------------------------------------------------------
# Chunk building
# ---------------------------------------------------------------------------

def build_chunks(
    doc,
    replacements: Dict[str, str],
    chunker,
    page_vlm_text: Optional[Dict[int, str]] = None,
    scanned_pages: Optional[Set[int]] = None,
) -> List[DocumentChunk]:
    """Build DocumentChunks from the HybridChunker output.

    HybridChunker never yields PictureItems, so picture VLM text lives only in
    image_chunks (returned by extract_images).  Scanned pages always have their
    VLM text injected here as plain text chunks.
    """
    scanned_pages    = scanned_pages or set()
    page_vlm_text    = page_vlm_text or {}
    seen_refs: Set[str] = set()
    chunks: List[DocumentChunk] = []
    pages_covered: Set[int]  = set()

    stats = dict(from_text=0, from_vlm=0, skipped_placeholder=0, skipped_duplicate=0)

    for chunk in chunker.chunk(doc):
        items = list(getattr(chunk.meta, "doc_items", []) or [])
        text, source = _resolve_chunk_text(chunk, items, replacements, seen_refs, stats)

        text = (text or "").strip()
        if not text or is_placeholder(text):
            stats["skipped_placeholder"] += 1
            continue

        headings = list(getattr(chunk.meta, "headings", None) or [])
        pages    = chunk_pages(chunk) or [1]
        pages_covered.update(pages)
        bbox     = next((item_bbox(item) for item in items if item_bbox(item)), None)

        chunks.append(DocumentChunk(text=text, headings=headings, page_numbers=pages, bbox=bbox))
        stats["from_vlm" if source == "vlm" else "from_text"] += 1

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

    # Inject full-page VLM text for scanned pages
    pages_injected = 0
    for pg_no, text in sorted(page_vlm_text.items()):
        text = text.strip()
        if not text or is_placeholder(text):
            continue
        # Non-scanned pages: only inject when no chunk already covers them
        if pg_no not in scanned_pages and pg_no in pages_covered:
            continue
        pages_covered.add(pg_no)
        chunks.append(DocumentChunk(text=text, headings=[], page_numbers=[pg_no], bbox=None))
        stats["from_vlm"] += 1
        pages_injected   += 1

    total_pages = len(getattr(doc, "pages", {}))
    uncovered   = sorted(set(range(1, total_pages + 1)) - pages_covered)

    print(
        f"Chunks: {len(chunks)} total  "
        f"({stats['from_text']} from text layer, {stats['from_vlm']} from VLM, "
        f"{pic_refs_noted} picture(s) in image_chunks only)\n"
        f"  skipped: {stats['skipped_placeholder']} empty/placeholder, "
        f"{stats['skipped_duplicate']} duplicate table sub-chunks\n"
        f"  page coverage: {len(pages_covered)}/{total_pages}"
        + (f"  |  no-chunk pages: {uncovered}" if uncovered else "")
    )
    return chunks


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
) -> List[ImageChunk]:
    """Extract and encode all pictures that are not on scanned pages."""
    scanned_pages = scanned_pages or set()
    pictures      = getattr(doc, "pictures", [])
    eligible      = [p for p in pictures if item_page(p) not in scanned_pages]
    skipped       = len(pictures) - len(eligible)

    if skipped:
        print(f"Images: extracting {len(eligible)} ({skipped} skipped on scanned pages)")
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
    return chunks


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
