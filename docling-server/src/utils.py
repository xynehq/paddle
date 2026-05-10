import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

from config import SCANNED_PAGE_CHAR_THRESHOLD, VLM_CONCURRENCY
from docling_core.types.doc import DocItemLabel
from models import TocEntry, VlmConfig
from vlm import call_vlm

# Matches docling's missing-text placeholder tokens
_MISSING_TEXT_RE = re.compile(r"^(<!--\s*missing-text\s*-->\s*)+$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

def is_placeholder(text: str) -> bool:
    """Return True when *text* contains only whitespace or missing-text tokens."""
    return not text.strip() or bool(_MISSING_TEXT_RE.fullmatch(text.strip()))


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def extract_tables(doc) -> Dict[str, str]:
    """Export every table to GFM Markdown using docling's native cell structure.

    Returns a mapping of ``table.self_ref → markdown text``.
    """
    replacements: Dict[str, str] = {}
    tables = getattr(doc, "tables", [])
    print(f"Tables: extracting {len(tables)} table(s)")

    for i, tbl in enumerate(tables):
        label = f"table {i + 1}/{len(tables)} [{tbl.self_ref}]"
        try:
            md = tbl.export_to_markdown(doc=doc)
            if md and md.strip():
                replacements[tbl.self_ref] = md.strip()
                print(f"  {label}: {len(md)} chars")
            else:
                print(f"  {label}: empty")
        except Exception as exc:
            print(f"  {label}: FAILED — {exc}")

    return replacements


# ---------------------------------------------------------------------------
# Scanned-page detection
# ---------------------------------------------------------------------------

def detect_scanned_pages(doc) -> Set[int]:
    """Return page numbers whose embedded text falls below the char threshold.

    These pages are treated as scanned / image-only and sent to the VLM as
    full-page images instead of having their pictures processed individually.
    """
    page_chars: Dict[int, int] = {}
    for item in getattr(doc, "texts", []):
        txt = getattr(item, "text", "") or ""
        for prov in getattr(item, "prov", []):
            page_chars[prov.page_no] = page_chars.get(prov.page_no, 0) + len(txt)

    pages = getattr(doc, "pages", {})
    return {pg for pg in pages if page_chars.get(pg, 0) < SCANNED_PAGE_CHAR_THRESHOLD}


# ---------------------------------------------------------------------------
# VLM: individual pictures
# ---------------------------------------------------------------------------

def process_images_with_vlm(
    doc,
    cfg: VlmConfig,
    table_refs: Set[str],
    scanned_pages: Set[int],
) -> Dict[str, str]:
    """OCR every picture (excluding those on scanned pages) via the VLM.

    Returns a mapping of ``picture.self_ref → text``.

    VLM calls run concurrently up to VLM_CONCURRENCY in-flight requests.
    """
    pictures = getattr(doc, "pictures", [])

    eligible = [p for p in pictures if item_page(p) not in scanned_pages]
    skipped  = len(pictures) - len(eligible)

    if skipped:
        print(f"VLM pictures: skipping {skipped} on scanned pages")

    total = len(eligible)
    if total == 0:
        print("VLM pictures: processing 0 picture(s)")
        return {}

    workers = min(VLM_CONCURRENCY, total)
    print(f"VLM pictures: processing {total} with concurrency={workers}")

    # Pre-render all images on the main thread (docling doc access not
    # guaranteed thread-safe), then send the VLM requests concurrently.
    rendered: List[Tuple[int, str, Optional[Image.Image]]] = []
    for i, pic in enumerate(eligible):
        try:
            img = pic.get_image(doc=doc)
        except Exception as exc:
            print(f"  picture {i+1}/{total} [{pic.self_ref}]: render FAILED — {exc}")
            img = None
        rendered.append((i, pic.self_ref, img))

    def ocr_one(item: Tuple[int, str, Optional[Image.Image]]) -> Tuple[str, Optional[str]]:
        i, ref, img = item
        if img is None:
            print(f"  picture {i+1}/{total} [{ref}]: no image, skipped")
            return ref, None
        t = time.time()
        try:
            text = call_vlm(cfg, img, cfg.image_prompt)
            elapsed = time.time() - t
            if text:
                print(f"  picture {i+1}/{total} [{ref}]: {len(text)} chars in {elapsed:.1f}s")
                return ref, text
            print(f"  picture {i+1}/{total} [{ref}]: empty response in {elapsed:.1f}s")
            return ref, None
        except Exception as exc:
            print(f"  picture {i+1}/{total} [{ref}]: FAILED in {time.time()-t:.1f}s — {exc}")
            return ref, None

    replacements: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ref, text in ex.map(ocr_one, rendered):
            if text:
                replacements[ref] = text

    return replacements


# ---------------------------------------------------------------------------
# VLM: full scanned pages
# ---------------------------------------------------------------------------

def process_scanned_pages_with_vlm(
    doc,
    cfg: VlmConfig,
    scanned_pages: Set[int],
) -> Dict[int, str]:
    """Send each scanned page as a full-page image to the VLM.

    Returns a mapping of ``page_no → text``.
    """
    if not scanned_pages:
        print("Page VLM: all pages have embedded text, no scanned pages detected")
        return {}

    pages  = getattr(doc, "pages", {})
    scanned = sorted(scanned_pages)
    workers = min(VLM_CONCURRENCY, len(scanned))
    print(
        f"Page VLM: {len(scanned)}/{len(pages)} scanned page(s) "
        f"(< {SCANNED_PAGE_CHAR_THRESHOLD} chars) concurrency={workers}: {scanned}"
    )

    # Snapshot rendered page images on the main thread before fanning out.
    rendered: List[Tuple[int, Optional[Image.Image]]] = []
    for pg_no in scanned:
        try:
            page = pages[pg_no]
            img_ref = getattr(page, "image", None)
            pil = getattr(img_ref, "pil_image", None) if img_ref else None
        except Exception as exc:
            print(f"  page {pg_no}: render FAILED — {exc}")
            pil = None
        rendered.append((pg_no, pil))

    def ocr_page(item: Tuple[int, Optional[Image.Image]]) -> Tuple[int, Optional[str]]:
        pg_no, pil = item
        if pil is None:
            print(f"  page {pg_no}: no rendered image available, skipped")
            return pg_no, None
        t = time.time()
        try:
            text = call_vlm(cfg, pil, cfg.image_prompt)
            elapsed = time.time() - t
            if text:
                print(f"  page {pg_no}: {len(text)} chars in {elapsed:.1f}s")
                return pg_no, text
            print(f"  page {pg_no}: empty VLM response in {elapsed:.1f}s")
            return pg_no, None
        except Exception as exc:
            print(f"  page {pg_no}: FAILED in {time.time()-t:.1f}s — {exc}")
            return pg_no, None

    results: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pg_no, text in ex.map(ocr_page, rendered):
            if text:
                results[pg_no] = text

    return results

def item_bbox(item) -> Optional[Dict[str, float]]:
    """Return the bounding box of the first provenance entry, or None."""
    provs = getattr(item, "prov", [])
    if provs and hasattr(provs[0], "bbox"):
        b = provs[0].bbox
        return {"l": round(b.l, 4), "t": round(b.t, 4),
                "r": round(b.r, 4), "b": round(b.b, 4)}
    return None


def item_page(item) -> int:
    """Return the page number of the first provenance entry (default 1)."""
    provs = getattr(item, "prov", [])
    return provs[0].page_no if provs and hasattr(provs[0], "page_no") else 1


def chunk_pages(chunk) -> List[int]:
    """Return sorted page numbers covered by a HybridChunker chunk."""
    pages = set()
    for doc_item in getattr(chunk.meta, "doc_items", []) or []:
        for prov in getattr(doc_item, "prov", []):
            if hasattr(prov, "page_no"):
                pages.add(prov.page_no)
    return sorted(pages)

def build_toc(doc) -> List[TocEntry]:
    """Extract a flat TOC with parent-index linkage from section headers / titles."""
    entries: List[TocEntry] = []
    counter = 0

    for item in getattr(doc, "texts", []):
        if getattr(item, "label", None) not in (
            DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE
        ):
            continue

        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue

        level   = getattr(item, "level", None) or 1
        counter += 1

        entry = TocEntry(
            section_number=str(counter),
            section_title=text,
            page_number=item_page(item),
            level=level,
            bbox=item_bbox(item),
        )

        # Attach to the nearest ancestor with a lower level
        for j in range(len(entries) - 1, -1, -1):
            if entries[j].level < entry.level:
                entry.parent_index = j
                break

        entries.append(entry)

    return entries
