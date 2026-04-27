import os
import re
import json
import tempfile
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice, PdfPipelineOptions
from docling.chunking import HybridChunker
from docling_core.types.doc import DocItemLabel

app = FastAPI(title="Docling Document Processing Service")

# Initialize models globally
doc_converter: Optional[DocumentConverter] = None
chunker: Optional[HybridChunker] = None


@dataclass
class TocEntry:
    section_number: str
    section_title: str
    page_number: int
    level: int
    bbox: Optional[Dict[str, float]] = None
    parent_index: Optional[int] = None


@dataclass
class DocumentChunk:
    text: str
    headings: List[str]
    page_numbers: List[int]
    section_path: List[str]
    bbox: Optional[Dict[str, float]] = None


def initialize_models():
    """Initialize Docling models on startup"""
    global doc_converter, chunker
    
    print("Initializing Docling models...")
    
    # Configure accelerator (use MPS on Mac, CUDA on Linux)
    accel_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.MPS if os.uname().sysname == "Darwin" else AcceleratorDevice.CPU
    )
    
    # Document converter with proper pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accel_options
    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Hybrid chunker with BGE-Small tokenizer
    chunker = HybridChunker(
        tokenizer="BAAI/bge-small-en-v1.5",
        max_tokens=512,
        merge_peers=True
    )
    
    print("Models initialized successfully")


def extract_section_number(text: str) -> Optional[str]:
    """Extract section number from header text"""
    text = text.strip()
    
    # Pattern 1: "1.1", "1.1.2", etc.
    match = re.match(r'^(\d+(?:\.\d+)*)[.\s]+', text)
    if match:
        return match.group(1)
    
    # Pattern 2: "CHAPTER 1", "Section 2", etc.
    match = re.match(r'^(CHAPTER|Chapter|SECTION|Section)\s+(\d+)[.:\s]*', text, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()} {match.group(2)}"
    
    # Pattern 3: Single digit at start (e.g., "1 Introduction")
    match = re.match(r'^(\d+)[.\s]+', text)
    if match:
        return match.group(1)
    
    return None


def calculate_level(section_number: str) -> int:
    """Calculate hierarchy level from section number"""
    if not section_number:
        return 1
    
    # CHAPTER X = level 1
    if section_number.upper().startswith("CHAPTER"):
        return 1
    
    # Count dots in numeric sections (1.1.2 = level 3)
    return section_number.count(".") + 1


def extract_bbox(item) -> Optional[Dict[str, float]]:
    """Extract bounding box from item provenance"""
    if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
        prov = item.prov[0]
        if hasattr(prov, 'bbox'):
            return {
                "l": round(prov.bbox.l, 4),
                "t": round(prov.bbox.t, 4),
                "r": round(prov.bbox.r, 4),
                "b": round(prov.bbox.b, 4)
            }
    return None


def extract_page_number(item) -> int:
    """Extract page number from item provenance"""
    if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
        prov = item.prov[0]
        if hasattr(prov, 'page_no'):
            return prov.page_no
    return 1


def clean_title(text: str, section_number: Optional[str]) -> str:
    """Clean section title by removing section number prefix"""
    text = text.strip()
    
    # Remove section number prefix
    if section_number:
        patterns = [
            rf'^{re.escape(section_number)}[.\s]+',
            r'^(CHAPTER|Chapter|SECTION|Section)\s+\d+[.:\s]*',
            r'^\d+[.\s]+'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Strip whitespace
    return text.strip()


def build_toc_entries(doc) -> List[TocEntry]:
    """Extract TOC from section headers in document"""
    entries = []
    
    for item in getattr(doc, 'texts', []):
        label = getattr(item, 'label', None)
        
        # Filter for headers only
        if label not in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            continue
        
        text = getattr(item, 'text', '')
        if not text:
            continue
        
        # Extract section number
        section_number = extract_section_number(text)
        if not section_number:
            continue  # Skip items without section numbers
        
        # Clean title
        title = clean_title(text, section_number)
        if not title:
            continue
        
        entry = TocEntry(
            section_number=section_number,
            section_title=title,
            page_number=extract_page_number(item),
            level=calculate_level(section_number),
            bbox=extract_bbox(item)
        )
        entries.append(entry)
    
    # Build parent-child relationships
    for i, entry in enumerate(entries):
        entry.parent_index = None
        for j in range(i - 1, -1, -1):
            if entries[j].level < entry.level:
                entry.parent_index = j
                break
    
    return entries


def extract_section_path(headings: List[str]) -> List[str]:
    """Build section path array from chunk headings"""
    path = []
    for heading in headings:
        section = extract_section_number(heading)
        if section and section not in path:
            path.append(section)
    return path


def extract_chunk_pages(chunk) -> List[int]:
    """Extract page numbers from chunk metadata"""
    pages = set()
    if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
        for item in chunk.meta.doc_items:
            if hasattr(item, 'prov'):
                for prov in item.prov:
                    if hasattr(prov, 'page_no'):
                        pages.add(prov.page_no)
    return sorted(list(pages))


def extract_chunk_bbox(chunk) -> Optional[Dict[str, float]]:
    """Extract bounding box from first item in chunk"""
    if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
        for item in chunk.meta.doc_items:
            bbox = extract_bbox(item)
            if bbox:
                return bbox
    return None


def build_chunks(doc) -> List[DocumentChunk]:
    """Generate context-aware chunks using HybridChunker"""
    chunks = []
    
    for chunk in chunker.chunk(doc):
        # Get contextualized text
        text = chunker.contextualize(chunk)
        if not text or not text.strip():
            continue
        
        # Strip whitespace
        text = text.strip()
        
        # Extract metadata
        headings = getattr(chunk.meta, 'headings', None) if hasattr(chunk, 'meta') else None
        headings = headings or []  # Handle None case
        page_numbers = extract_chunk_pages(chunk)
        section_path = extract_section_path(headings)
        bbox = extract_chunk_bbox(chunk)
        
        chunks.append(DocumentChunk(
            text=text,
            headings=headings,
            page_numbers=page_numbers if page_numbers else [1],
            section_path=section_path,
            bbox=bbox
        ))
    
    return chunks


def process_document_sync(file_path: str) -> Dict[str, Any]:
    """Synchronously process document"""
    start_time = time.time()
    
    # Convert document
    conv_result = doc_converter.convert(file_path)
    document = conv_result.document
    
    print(f"Document converted in {time.time() - start_time:.2f}s")
    
    # Build TOC
    toc_start = time.time()
    toc_entries = build_toc_entries(document)
    print(f"TOC extracted: {len(toc_entries)} entries in {time.time() - toc_start:.2f}s")
    
    # Build chunks
    chunk_start = time.time()
    chunks = build_chunks(document)
    print(f"Chunks generated: {len(chunks)} chunks in {time.time() - chunk_start:.2f}s")
    
    # Format response
    result = {
        "metadata": {
            "filename": Path(file_path).name,
            "num_pages": len(getattr(document, 'pages', [])),
            "processing_time": round(time.time() - start_time, 2),
            "has_toc": len(toc_entries) > 0
        },
        "toc": {
            "entries": [
                {
                    "section_number": e.section_number,
                    "section_title": e.section_title,
                    "page_number": e.page_number,
                    "level": e.level,
                    "bbox": e.bbox,
                    "parent_index": e.parent_index
                }
                for e in toc_entries
            ]
        },
        "chunks": [
            {
                "text": c.text,
                "headings": c.headings,
                "page_numbers": c.page_numbers,
                "section_path": c.section_path,
                "bbox": c.bbox
            }
            for c in chunks
        ]
    }
    
    return result


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if doc_converter is None or chunker is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    return {"status": "ok", "models_loaded": True}


@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    Process PDF document and return structured TOC and chunks
    
    Returns JSON with:
    - metadata: document info
    - toc: structured table of contents with hierarchy
    - chunks: contextualized text chunks with metadata
    """
    tmp_path = None
    
    try:
        # Validate file type
        suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        if suffix != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files accepted")
        
        # Save to temp file
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'wb') as f:
                content = await file.read()
                f.write(content)
        except:
            os.close(fd)
            raise
        
        # Process in thread pool (blocking operation)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, process_document_sync, tmp_path)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
