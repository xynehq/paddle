import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

from config import (
    LIGHTON_MIN_CROP_AREA,
    LIGHTON_MIN_CROP_HEIGHT,
    LIGHTON_MIN_CROP_WIDTH,
    SCANNED_PAGE_CHAR_THRESHOLD,
    VLM_CONCURRENCY,
)
from docling_core.types.doc import DocItemLabel
from models import ImageChunk, TocEntry, VlmConfig, VlmDetectedImage, VlmPageResult
from vlm import call_vlm, encode_image_jpeg

# Matches docling's missing-text placeholder tokens
_MISSING_TEXT_RE = re.compile(r"^(<!--\s*missing-text\s*-->\s*)+$", re.MULTILINE)
_FLOAT_VALUE_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FLOAT = r"-?\d+(?:\.\d+)?"
_COORDS = rf"(?:\[\s*{_FLOAT}\s*,\s*{_FLOAT}\s*,\s*{_FLOAT}\s*,\s*{_FLOAT}\s*\]|{_FLOAT}\s*,\s*{_FLOAT}\s*,\s*{_FLOAT}\s*,\s*{_FLOAT})"
_LIGHTON_IMAGE_MARKER_RE = re.compile(
    rf"!\[image\]\([^)]*\)\s*(?P<coords>{_COORDS})",
    re.IGNORECASE,
)
_LIGHTON_IMAGE_TAG_RE = re.compile(r"!\[image\]\([^)]*\)\s*", re.IGNORECASE)

# Permissive marker pattern: catches `![image](...)` followed by ANY
# comma-separated number list (1-6 numbers, optional brackets). The model
# occasionally emits 3-number coords (cx,cy,size) instead of the canonical
# 4-number corners. The strict matcher above misses those, leaving the raw
# marker text to leak into chunks. This permissive form is used only as a
# final stripping safety net — never for crop OCR (we don't know what the
# 3-number format means semantically).
_LIGHTON_IMAGE_MARKER_LOOSE_RE = re.compile(
    r"!\[image\]\([^)]*\)\s*"
    r"(?:\[?\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){0,5}\s*\]?)?"
    r"\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

def is_placeholder(text: str) -> bool:
    """Return True when *text* contains only whitespace or missing-text tokens."""
    return not text.strip() or bool(_MISSING_TEXT_RE.fullmatch(text.strip()))


# ---------------------------------------------------------------------------
# LightOnOCR markdown image marker parsing
# ---------------------------------------------------------------------------

def parse_lighton_image_regions(text: str, page_size: Tuple[int, int]) -> List[VlmDetectedImage]:
    """Parse LightOnOCR markdown image markers with adjacent bbox coordinates."""
    regions: List[VlmDetectedImage] = []
    for match in _LIGHTON_IMAGE_MARKER_RE.finditer(text or ""):
        coords = _parse_marker_coords(match.group("coords"))
        if coords is None:
            continue

        region = _build_detected_region(
            marker=match.group(0),
            coords=coords,
            page_size=page_size,
            start=match.start(),
            end=match.end(),
        )
        if region:
            regions.append(region)

    return regions


def strip_lighton_image_markers(text: str) -> str:
    """Remove LightOnOCR image markers from crop OCR output."""
    text = _LIGHTON_IMAGE_MARKER_RE.sub("", text or "")
    text = _LIGHTON_IMAGE_TAG_RE.sub("", text)
    return _normalize_removed_marker_spacing(text)


def strip_lighton_image_markers_loose(text: str) -> str:
    """Aggressively strip ``![image](...)`` + any trailing comma-separated
    number list from page OCR output. Used as a safety net for markers that
    escape strict bbox parsing (malformed coord counts, alternate formats).
    Always safe to call — never alters real prose.
    """
    return _normalize_removed_marker_spacing(
        _LIGHTON_IMAGE_MARKER_LOOSE_RE.sub("", text or "")
    )


def remove_processed_lighton_markers(text: str, regions: List[VlmDetectedImage]) -> str:
    """Remove only page markers whose crops produced OCR text."""
    if not regions:
        return text

    cleaned = text or ""
    for region in sorted(regions, key=lambda item: item.start, reverse=True):
        before = cleaned[:region.start]
        after = cleaned[region.end:]
        if before.endswith((" ", "\t")) and after.startswith((" ", "\t")):
            after = after[1:]
        cleaned = before + after
    return _normalize_removed_marker_spacing(cleaned)


def is_small_lighton_crop(region: VlmDetectedImage) -> bool:
    """Return True when a detected region is too small to OCR usefully."""
    return (
        region.width < LIGHTON_MIN_CROP_WIDTH
        or region.height < LIGHTON_MIN_CROP_HEIGHT
        or region.width * region.height < LIGHTON_MIN_CROP_AREA
    )


def _parse_marker_coords(value: str) -> Optional[Tuple[float, float, float, float]]:
    numbers = [float(part) for part in _FLOAT_VALUE_RE.findall(value or "")]
    if len(numbers) != 4:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _build_detected_region(
    marker: str,
    coords: Tuple[float, float, float, float],
    page_size: Tuple[int, int],
    start: int,
    end: int,
) -> Optional[VlmDetectedImage]:
    page_w, page_h = page_size
    if page_w <= 0 or page_h <= 0:
        return None

    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None

    if all(0 <= value <= 1 for value in coords):
        scaled = (x1 * page_w, y1 * page_h, x2 * page_w, y2 * page_h)
    elif all(0 <= value <= 100 for value in coords):
        scaled = (
            x1 * page_w / 100,
            y1 * page_h / 100,
            x2 * page_w / 100,
            y2 * page_h / 100,
        )
    else:
        scaled = coords

    left = _clamp_int(round(scaled[0]), 0, page_w)
    top = _clamp_int(round(scaled[1]), 0, page_h)
    right = _clamp_int(round(scaled[2]), 0, page_w)
    bottom = _clamp_int(round(scaled[3]), 0, page_h)

    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None

    return VlmDetectedImage(
        marker=marker,
        bbox={"l": round(x1, 4), "t": round(y1, 4), "r": round(x2, 4), "b": round(y2, 4)},
        pixel_bbox=(left, top, right, bottom),
        width=width,
        height=height,
        start=start,
        end=end,
    )


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _normalize_removed_marker_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
# Page text quality detection
# ---------------------------------------------------------------------------

# Legacy font / mojibake detection patterns
LEGACY_FONT_RE = re.compile(
    r"(?:H\$|H\$s|Ho\$|\{[A-Za-z]|A§|qg|pñ|J«|Jw|ñ|¶|à|µ|»|‹|›|ß|Ë|ã|œ)"
)
STRONG_LEGACY_MARKERS = ("H$", "{", "}", "A§", "¶", "ñ", "à", "µ", "»", "‹", "›")
NOISE_CHARS = set("{}$~^`\\|")

NON_LATIN_SCRIPT_RANGES = (
    (0x0370, 0x03FF),   # Greek
    (0x0400, 0x052F),   # Cyrillic
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x0900, 0x097F),   # Devanagari
    (0x0980, 0x09FF),   # Bengali
    (0x0A00, 0x0A7F),   # Gurmukhi
    (0x0A80, 0x0AFF),   # Gujarati
    (0x0B00, 0x0B7F),   # Odia
    (0x0B80, 0x0BFF),   # Tamil
    (0x0C00, 0x0C7F),   # Telugu
    (0x0C80, 0x0CFF),   # Kannada
    (0x0D00, 0x0D7F),   # Malayalam
    (0x0E00, 0x0E7F),   # Thai
    (0x0E80, 0x0EFF),   # Lao
    (0x1000, 0x109F),   # Myanmar
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x1780, 0x17FF),   # Khmer
    (0x3040, 0x30FF),   # Hiragana/Katakana
    (0x3400, 0x9FFF),   # CJK
    (0xAC00, 0xD7AF),   # Hangul
)


def is_non_latin_script_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in NON_LATIN_SCRIPT_RANGES)


def has_real_non_latin_script(text: str) -> bool:
    return sum(1 for ch in text if ch.isalpha() and is_non_latin_script_char(ch)) >= 3


def is_bad_native_text_layer(text: str) -> bool:
    """Detect legacy-font / mojibake extraction while preserving real Unicode scripts."""
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) < 80:
        return False

    if has_real_non_latin_script(compact):
        return False

    nonspace = [ch for ch in compact if not ch.isspace()]
    if not nonspace:
        return False

    marker_hits = len(LEGACY_FONT_RE.findall(compact))
    strong_marker_hits = sum(compact.count(marker) for marker in STRONG_LEGACY_MARKERS)
    noise_hits = sum(1 for ch in compact if ch in NOISE_CHARS)
    latin1_suspect_hits = sum(1 for ch in compact if ch in "ñàáâãäåæçèéêëìíîïòóôõöùúûüýÿ")

    marker_density = marker_hits / max(len(compact), 1)
    noise_density = noise_hits / len(nonspace)

    if marker_hits >= 8 and marker_density >= 0.015:
        return True
    if marker_hits >= 4 and strong_marker_hits >= 8 and noise_density >= 0.025:
        return True
    if latin1_suspect_hits >= 8 and strong_marker_hits >= 8 and noise_density >= 0.035:
        return True

    return False


def collect_page_text(doc) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Collect text and char counts per page."""
    page_text: Dict[int, List[str]] = {pg_no: [] for pg_no in getattr(doc, "pages", {})}
    page_chars: Dict[int, int] = {pg_no: 0 for pg_no in getattr(doc, "pages", {})}
    for item in getattr(doc, "texts", []):
        txt = getattr(item, "text", "") or ""
        for prov in getattr(item, "prov", []):
            pg_no = getattr(prov, "page_no", None)
            if pg_no is None:
                continue
            page_text.setdefault(pg_no, []).append(txt)
            page_chars[pg_no] = page_chars.get(pg_no, 0) + len(txt)
    return {pg_no: "\n".join(parts) for pg_no, parts in page_text.items()}, page_chars


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


def is_mostly_english(text: str) -> bool:
    """Check if text is predominantly English (ASCII Latin letters)."""
    if not text:
        return True  # Empty text counts as English
    
    # Count Latin letters vs non-Latin scripts
    latin_alpha = sum(1 for ch in text if ch.isalpha() and ord(ch) < 128)
    total_alpha = sum(1 for ch in text if ch.isalpha())
    
    if total_alpha == 0:
        return True  # No letters at all, treat as neutral
    
    # If less than 70% Latin letters, consider it non-English
    return (latin_alpha / total_alpha) >= 0.70


def analyze_page_text_quality(doc) -> Dict[int, Dict[str, any]]:
    """Analyze text quality per page and return OCR candidates.
    
    ALL pages are OCR candidates - native text layer is not trusted.
    VLM OCR is used for every page.
    
    Returns dict with page_no -> {native_chars, decision, ocr_candidate}
    """
    page_text, page_chars = collect_page_text(doc)
    qualities: Dict[int, Dict[str, any]] = {}
    pages = getattr(doc, "pages", {})

    print(f"Page quality: ALL pages sent to VLM OCR (native layer bypassed)")
    for pg_no in sorted(pages):
        text = page_text.get(pg_no, "")
        native_chars = page_chars.get(pg_no, 0)
        
        # All pages go to OCR
        decision = "force_ocr"
        ocr_candidate = True
        
        qualities[pg_no] = {
            "page_no": pg_no,
            "native_chars": native_chars,
            "decision": decision,
            "ocr_candidate": ocr_candidate,
        }
        print(f"  page {pg_no}: chars={native_chars} decision={decision}")

    candidates = list(pages.keys())
    print(f"Page quality: {len(candidates)}/{len(pages)} page(s) selected for page OCR: {candidates}")
    return qualities


# ---------------------------------------------------------------------------
# VLM: individual pictures
# ---------------------------------------------------------------------------

def process_images_with_vlm(
    doc,
    cfg: VlmConfig,
    table_refs: Set[str],
    scanned_pages: Set[int],
    skip_pages: Optional[Set[int]] = None,
) -> Tuple[Dict[str, str], int]:
    """OCR every picture (excluding those on scanned or skip pages) via the VLM.

    Returns a mapping of ``picture.self_ref → text`` and count of skipped pictures.

    VLM calls run concurrently up to VLM_CONCURRENCY in-flight requests.
    """
    skip_pages = skip_pages or set()
    pictures = getattr(doc, "pictures", [])

    # Combine scanned_pages and skip_pages
    excluded_pages = scanned_pages.union(skip_pages)
    
    eligible = [p for p in pictures if item_page(p) not in excluded_pages]
    skipped = len(pictures) - len(eligible)

    total = len(pictures)
    selected_total = len(eligible)
    
    if total == 0:
        print("VLM pictures: processing 0 picture(s)")
        return {}, 0
    if selected_total == 0:
        print(f"VLM pictures: processing 0/{total} picture(s); skipped {skipped} on page-OCR pages")
        return {}, skipped

    workers = min(VLM_CONCURRENCY, selected_total)
    print(f"VLM pictures: processing {selected_total}/{total} with concurrency={workers}"
          + (f"; skipped {skipped} on page-OCR pages" if skipped else ""))

    # Pre-render all images on the main thread (docling doc access not
    # guaranteed thread-safe), then send the VLM requests concurrently.
    rendered: List[Tuple[int, str, Optional[Image.Image]]] = []
    for i, pic in enumerate(eligible):
        try:
            img = pic.get_image(doc=doc)
        except Exception as exc:
            print(f"  picture {i+1}/{selected_total} [{pic.self_ref}]: render FAILED — {exc}")
            img = None
        rendered.append((i, pic.self_ref, img))

    def ocr_one(item: Tuple[int, str, Optional[Image.Image]]) -> Tuple[str, Optional[str]]:
        i, ref, img = item
        if img is None:
            print(f"  picture {i+1}/{selected_total} [{ref}]: no image, skipped")
            return ref, None
        t = time.time()
        try:
            text = call_vlm(cfg, img, cfg.image_prompt)
            elapsed = time.time() - t
            if text:
                print(f"  picture {i+1}/{selected_total} [{ref}]: {len(text)} chars in {elapsed:.1f}s")
                return ref, text
            print(f"  picture {i+1}/{selected_total} [{ref}]: empty response in {elapsed:.1f}s")
            return ref, None
        except Exception as exc:
            print(f"  picture {i+1}/{selected_total} [{ref}]: FAILED in {time.time()-t:.1f}s — {exc}")
            return ref, None

    replacements: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ref, text in ex.map(ocr_one, rendered):
            if text:
                replacements[ref] = text

    return replacements, skipped


# ---------------------------------------------------------------------------
# VLM: full scanned pages
# ---------------------------------------------------------------------------

def process_scanned_pages_with_vlm(
    doc,
    cfg: VlmConfig,
    pages_to_ocr: List[int],
) -> Dict[int, VlmPageResult]:
    """OCR selected pages via the external VLM.

    Returns per-page OCR text plus first-level LightOnOCR bbox crop OCR results.
    """
    pages = getattr(doc, "pages", {})
    selected = [pg_no for pg_no in sorted(set(pages_to_ocr)) if pg_no in pages]

    if not selected:
        print("Page VLM: no pages selected for page OCR")
        return {}

    workers = min(VLM_CONCURRENCY, len(selected))
    print(f"Page VLM: processing {len(selected)}/{len(pages)} selected page(s) "
          f"with concurrency={workers}: {selected}")

    # Snapshot rendered page images on the main thread before fanning out.
    rendered: List[Tuple[int, Optional[Image.Image]]] = []
    for pg_no in selected:
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

    raw_results: Dict[int, Tuple[str, Image.Image]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pg_no, text in ex.map(ocr_page, rendered):
            if text:
                pil = next((img for page_no, img in rendered if page_no == pg_no), None)
                if pil is not None:
                    raw_results[pg_no] = (text, pil)

    results: Dict[int, VlmPageResult] = {}
    crop_jobs: List[Tuple[int, VlmDetectedImage, Image.Image]] = []

    for pg_no, (text, pil) in raw_results.items():
        page_result = VlmPageResult(text=text)
        regions = parse_lighton_image_regions(text, pil.size)
        page_result.image_regions_detected = len(regions)

        for region in regions:
            if is_small_lighton_crop(region):
                page_result.crop_ocr_skipped += 1
                page_result.crop_ocr_skipped_small += 1
                print(
                    f"  page {pg_no}: crop skipped small "
                    f"{region.width}x{region.height} bbox={region.bbox}"
                )
                continue

            try:
                crop = pil.crop(region.pixel_bbox)
            except Exception as exc:
                page_result.crop_ocr_skipped += 1
                print(f"  page {pg_no}: crop failed bbox={region.bbox} - {exc}")
                continue

            page_result.crop_ocr_attempted += 1
            crop_jobs.append((pg_no, region, crop))

        results[pg_no] = page_result

    if crop_jobs:
        crop_workers = min(VLM_CONCURRENCY, len(crop_jobs))
        print(f"Page VLM crops: processing {len(crop_jobs)} crop(s) with concurrency={crop_workers}")

        def ocr_crop(
            item: Tuple[int, VlmDetectedImage, Image.Image]
        ) -> Tuple[int, VlmDetectedImage, Image.Image, Optional[str]]:
            pg_no, region, crop = item
            t = time.time()
            try:
                text = call_vlm(cfg, crop, cfg.image_prompt)
                elapsed = time.time() - t
                clean_text = strip_lighton_image_markers(text)
                if clean_text and not is_placeholder(clean_text):
                    print(
                        f"  page {pg_no}: crop {region.width}x{region.height} "
                        f"{len(clean_text)} chars in {elapsed:.1f}s"
                    )
                    return pg_no, region, crop, clean_text

                print(f"  page {pg_no}: crop empty response in {elapsed:.1f}s")
                return pg_no, region, crop, None
            except Exception as exc:
                print(f"  page {pg_no}: crop FAILED in {time.time()-t:.1f}s - {exc}")
                return pg_no, region, crop, None

        successful_regions: Dict[int, List[VlmDetectedImage]] = {}
        with ThreadPoolExecutor(max_workers=crop_workers) as ex:
            for pg_no, region, crop, text in ex.map(ocr_crop, crop_jobs):
                page_result = results[pg_no]
                if not text:
                    page_result.crop_ocr_failed += 1
                    continue

                page_result.crop_ocr_success += 1
                successful_regions.setdefault(pg_no, []).append(region)
                page_result.image_chunks.append(ImageChunk(
                    text=text,
                    image_base64=encode_image_jpeg(crop),
                    page_number=pg_no,
                    bbox=region.bbox,
                    width=region.width,
                    height=region.height,
                ))

        for pg_no, regions in successful_regions.items():
            results[pg_no].text = remove_processed_lighton_markers(results[pg_no].text, regions)

    # Safety-net sweep: strip any image markers that escaped strict parsing
    # (e.g. 3-number coords like `407,835,547`). Without this, malformed
    # markers leak into the page text and pollute downstream chunks.
    for pg_no in results:
        results[pg_no].text = strip_lighton_image_markers_loose(results[pg_no].text)

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

def build_toc(doc, suppress_pages: Optional[Set[int]] = None) -> List[TocEntry]:
    """Extract a flat TOC with parent-index linkage from section headers / titles."""
    suppress_pages = suppress_pages or set()
    entries: List[TocEntry] = []
    counter = 0

    for item in getattr(doc, "texts", []):
        page_number = item_page(item)
        if page_number in suppress_pages:
            continue
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
            page_number=page_number,
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


def build_page_ocr_toc(page_vlm_text: Dict[int, str]) -> List[TocEntry]:
    """Extract TOC entries from OCR text by looking for markdown headings."""
    entries: List[TocEntry] = []
    for pg_no, text in sorted(page_vlm_text.items()):
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            hashes = len(stripped) - len(stripped.lstrip("#"))
            if hashes < 1 or hashes > 6:
                continue
            title = stripped.lstrip("#").strip()
            if not title or title.startswith("!"):
                continue
            entries.append(TocEntry(
                section_number="",
                section_title=title[:200],
                page_number=pg_no,
                level=hashes,
                bbox=None,
                parent_index=None,
            ))
    return entries


def normalize_toc_entries(entries: List[TocEntry]) -> List[TocEntry]:
    """Normalize TOC entries with sequential numbering and parent links."""
    normalized: List[TocEntry] = []
    for _, original in sorted(enumerate(entries), key=lambda item: (item[1].page_number, item[0])):
        next_entry = TocEntry(
            section_number=str(len(normalized) + 1),
            section_title=original.section_title,
            page_number=original.page_number,
            level=original.level,
            bbox=original.bbox,
            parent_index=None,
        )
        for j in range(len(normalized) - 1, -1, -1):
            if normalized[j].level < next_entry.level:
                next_entry.parent_index = j
                break
        normalized.append(next_entry)
    return normalized
