"""Top-level document processing pipeline."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

from chunking import build_chunks, extract_images
from utils import (
    build_toc,
    detect_scanned_pages,
    extract_tables,
    process_images_with_vlm,
    process_scanned_pages_with_vlm,
)
from models import VlmConfig


def process_document(
    file_path: str,
    doc_id: str,
    doc_converter,
    chunker,
    vlm_config: Optional[VlmConfig],
) -> Dict[str, Any]:
    """Convert a PDF and return structured TOC, text chunks, and images."""
    t0 = time.time()

    # ── Conversion ──────────────────────────────────────────────────────────
    doc         = doc_converter.convert(file_path).document
    total_pages = len(getattr(doc, "pages", {}))
    print(
        f"Converted in {time.time() - t0:.2f}s  |  "
        f"{total_pages} pages, "
        f"{len(getattr(doc, 'texts', []))} text items, "
        f"{len(getattr(doc, 'tables', []))} tables, "
        f"{len(getattr(doc, 'pictures', []))} pictures"
    )

    # ── TOC ─────────────────────────────────────────────────────────────────
    toc = build_toc(doc)
    print(f"TOC: {len(toc)} entries")

    t1             = time.time()
    scanned_pages  = detect_scanned_pages(doc)
    if scanned_pages:
        print(f"Detected {len(scanned_pages)} scanned page(s): {sorted(scanned_pages)}")

    # Tables are always extracted natively (no VLM needed)
    replacements   = extract_tables(doc)
    tables_count   = len(replacements)

    pictures_count = 0
    page_vlm_text  = {}

    if vlm_config:
        img_replacements = process_images_with_vlm(
            doc, vlm_config, set(replacements), scanned_pages
        )
        replacements.update(img_replacements)
        pictures_count = len(img_replacements)
        page_vlm_text  = process_scanned_pages_with_vlm(doc, vlm_config, scanned_pages)
    else:
        print("VLM not configured — skipping image OCR and scanned page processing")

    print(
        f"Enrichment: {len(replacements)} replacements + "
        f"{len(page_vlm_text)} scanned pages in {time.time() - t1:.2f}s"
    )

    # ── Chunking & images ────────────────────────────────────────────────────
    chunks = build_chunks(
        doc, replacements, chunker,
        page_vlm_text=page_vlm_text,
        scanned_pages=scanned_pages,
    )
    images = extract_images(doc, replacements, scanned_pages)

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
                "scanned_pages_ocrd": len(page_vlm_text),
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
