"""Top-level document processing pipeline."""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from chunking import build_chunks, extract_images
from utils import (
    analyze_page_text_quality,
    build_page_ocr_toc,
    build_toc,
    extract_tables,
    normalize_toc_entries,
    process_images_with_vlm,
    process_scanned_pages_with_vlm,
)
from models import VlmConfig


def process_document(
    file_path: str,
    doc_id: str,
    doc_converter,
    hybrid_chunker,
    sem_chunker: Callable[[str], list[str]],
    vlm_config: Optional[VlmConfig],
    set_stage: Callable[[str], None] = lambda _stage: None,
) -> Dict[str, Any]:
    """Convert a PDF and return structured TOC, text chunks, and images.

    Uses HybridChunker for digital PDFs with structure, semchunk for scanned pages.
    Implements page text quality detection to decide between native text vs OCR.
    """
    t0 = time.time()

    # ── Conversion ──────────────────────────────────────────────────────────
    set_stage("docling")
    doc         = doc_converter.convert(file_path).document
    total_pages = len(getattr(doc, "pages", {}))
    print(
        f"Converted in {time.time() - t0:.2f}s  |  "
        f"{total_pages} pages, "
        f"{len(getattr(doc, 'texts', []))} text items, "
        f"{len(getattr(doc, 'tables', []))} tables, "
        f"{len(getattr(doc, 'pictures', []))} pictures"
    )

    # ── Page Quality Analysis ───────────────────────────────────────────────
    page_quality = analyze_page_text_quality(doc)
    page_ocr_candidates: List[int] = sorted(
        pg_no for pg_no, quality in page_quality.items()
        if quality["ocr_candidate"]
    )
    
    # ── TOC (will be built after determining which pages to suppress) ────────
    print(f"TOC: pending (will combine native + OCR sources)")

    t1 = time.time()

    # Tables are always extracted natively (no VLM needed)
    set_stage("tables")
    replacements   = extract_tables(doc)
    tables_count   = len(replacements)

    pictures_count = 0
    page_vlm_text: Dict[int, str] = {}
    page_ocr_failed: List[int] = []
    picture_ocr_skipped = 0
    crop_image_chunks = []
    crop_stats = dict(
        page_image_regions_detected=0,
        crop_ocr_attempted=0,
        crop_ocr_success=0,
        crop_ocr_failed=0,
        crop_ocr_skipped=0,
        crop_ocr_skipped_small=0,
    )

    if vlm_config:
        # Step 1: OCR candidate pages (full-page)
        set_stage("vlm_page_ocr")
        page_vlm_results = process_scanned_pages_with_vlm(doc, vlm_config, page_ocr_candidates)
        page_vlm_text = {pg_no: result.text for pg_no, result in page_vlm_results.items()}
        page_ocr_success: Set[int] = set(page_vlm_results.keys())
        page_ocr_failed = [pg_no for pg_no in page_ocr_candidates if pg_no not in page_ocr_success]
        for result in page_vlm_results.values():
            crop_image_chunks.extend(result.image_chunks)
            crop_stats["page_image_regions_detected"] += result.image_regions_detected
            crop_stats["crop_ocr_attempted"] += result.crop_ocr_attempted
            crop_stats["crop_ocr_success"] += result.crop_ocr_success
            crop_stats["crop_ocr_failed"] += result.crop_ocr_failed
            crop_stats["crop_ocr_skipped"] += result.crop_ocr_skipped
            crop_stats["crop_ocr_skipped_small"] += result.crop_ocr_skipped_small
        
        # Log failed pages
        for pg_no in page_ocr_failed:
            quality = page_quality.get(pg_no)
            native_chars = quality["native_chars"] if quality else 0
            reason = quality["decision"] if quality else "unknown"
            print(f"  page {pg_no}: chars={native_chars} decision=ocr_failed_keep_native original_reason={reason}")

        # Step 2: OCR individual pictures (skip pages that had successful page OCR)
        set_stage("vlm_image_ocr")
        img_replacements, picture_ocr_skipped = process_images_with_vlm(
            doc,
            vlm_config,
            set(replacements),
            scanned_pages=set(),  # We're using page_ocr_success now
            skip_pages=page_ocr_success,
        )
        replacements.update(img_replacements)
        pictures_count = len(img_replacements)
    else:
        page_ocr_success = set()
        print("VLM not configured — skipping image OCR and page OCR processing")

    # Determine which pages to suppress native chunks for
    suppress_native_pages: Set[int] = set(page_vlm_text.keys())

    # Build combined TOC
    native_toc = build_toc(doc, suppress_pages=suppress_native_pages)
    ocr_toc = build_page_ocr_toc(page_vlm_text)
    toc = normalize_toc_entries(native_toc + ocr_toc)
    print(f"TOC: {len(toc)} entries ({len(native_toc)} native, {len(ocr_toc)} page OCR)")

    print(
        f"Enrichment: {len(replacements)} replacements + "
        f"{len(page_vlm_text)} page OCR result(s) in {time.time() - t1:.2f}s"
    )

    # ── Chunking & images ────────────────────────────────────────────────────
    set_stage("chunking")
    chunks, chunk_stats = build_chunks(
        doc, replacements, hybrid_chunker,
        sem_chunker=sem_chunker,
        page_vlm_text=page_vlm_text,
        suppress_native_pages=suppress_native_pages,
    )
    images, image_extract_skipped = extract_images(
        doc, replacements, 
        scanned_pages=set(),
        skip_pages=suppress_native_pages
    )
    if crop_image_chunks:
        print(f"Images: adding {len(crop_image_chunks)} LightOn crop OCR image chunk(s)")
        images.extend(crop_image_chunks)

    # ── Response payload ─────────────────────────────────────────────────────
    return {
        "metadata": {
            "doc_id":          doc_id,
            "filename":        Path(file_path).name,
            "num_pages":       total_pages,
            "num_images":      len(images),
            "processing_time": round(time.time() - t0, 2),
            "has_toc":         bool(toc),
            "vlm": {
                "enabled":            vlm_config is not None,
                "preset":             vlm_config.preset if vlm_config else None,
                "model":              vlm_config.model  if vlm_config else None,
                "tables_replaced":    tables_count,
                "pictures_replaced":  pictures_count,
                "page_ocr_candidates": page_ocr_candidates,
                "page_ocr_success":   sorted(page_vlm_text.keys()),
                "page_ocr_failed":    page_ocr_failed,
                "native_chunks_suppressed": chunk_stats.get("native_chunks_suppressed", 0),
                "picture_ocr_skipped": max(picture_ocr_skipped, image_extract_skipped),
                **crop_stats,
            },
        },
        "toc": {
            "entries": [
                {
                    "section_number": e.section_number,
                    "section_title":  e.section_title,
                    "page_number":    e.page_number,
                    "level":          e.level,
                    "bbox":           e.bbox,
                    "parent_index":   e.parent_index,
                }
                for e in toc
            ]
        },
        "chunks": [
            {
                "text":         c.text,
                "headings":     c.headings,
                "page_numbers": c.page_numbers,
                "bbox":         c.bbox,
            }
            for c in chunks
        ],
        "image_chunks": [
            {
                "text":        ic.text,
                "page_number": ic.page_number,
                "bbox":        ic.bbox,
                "width":       ic.width,
                "height":      ic.height,
            }
            for ic in images
        ],
        "images": {f"img_{i}": ic.image_base64 for i, ic in enumerate(images)},
    }
