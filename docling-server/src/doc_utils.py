"""Lightweight document-item helpers and TOC extraction."""

from typing import Dict, List, Optional

from docling_core.types.doc import DocItemLabel

from models import TocEntry


# ---------------------------------------------------------------------------
# Per-item helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TOC builder
# ---------------------------------------------------------------------------

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
