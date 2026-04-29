import os
import json
import tempfile
import asyncio
import time
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image

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


@dataclass
class ImageChunk:
    text: str
    image_base64: str
    page_number: int
    bbox: Optional[Dict[str, float]] = None
    width: Optional[int] = None
    height: Optional[int] = None


def initialize_models():
    """Initialize Docling models on startup"""
    global doc_converter, chunker
    
    print("Initializing Docling models...")
    if torch.cuda.is_available():
        device = AcceleratorDevice.CUDA
    elif os.uname().sysname == "Darwin":
        device = AcceleratorDevice.MPS
    else:
        device = AcceleratorDevice.CPU

    accel_options = AcceleratorOptions(num_threads=4, device=device)

    
    # Document converter with proper pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accel_options
    pipeline_options.generate_picture_images = True  # Enable image extraction
    
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


def build_toc_entries(doc) -> List[TocEntry]:
    """Extract TOC from section headers in document using Docling's native hierarchy"""
    entries = []
    section_counter = 0
    
    for item in getattr(doc, 'texts', []):
        label = getattr(item, 'label', None)
        
        # Filter for headers only
        if label not in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            continue
        
        text = getattr(item, 'text', '')
        if not text:
            continue
        
        # Get heading level from Docling's native structure
        level = 1
        try:
            # Use Docling's _get_heading_level method if available
            level = doc._get_heading_level(item) or 1
        except Exception:
            # Fallback: use item.level if available
            level = getattr(item, 'level', 1) or 1
        
        section_counter += 1
        section_number = str(section_counter)
        
        entry = TocEntry(
            section_number=section_number,
            section_title=text.strip(),
            page_number=extract_page_number(item),
            level=level,
            bbox=extract_bbox(item)
        )
        entries.append(entry)
    
    # Build parent-child relationships based on native levels
    for i, entry in enumerate(entries):
        entry.parent_index = None
        for j in range(i - 1, -1, -1):
            if entries[j].level < entry.level:
                entry.parent_index = j
                break
    
    return entries


def extract_section_path(headings: List[str]) -> List[str]:
    """Build section path array from chunk headings"""
    # Return headings as-is since we no longer parse section numbers from text
    # This preserves the document structure without regex parsing
    return [h.strip() for h in headings if h and h.strip()]


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


def extract_images_as_base64(doc) -> List[ImageChunk]:
    """
    Extract images from document and encode as base64.

    Args:
        doc: DoclingDocument

    Returns:
        List of ImageChunk with OCR text, base64 data, and metadata
    """
    image_chunks = []

    # Process each picture
    pictures = getattr(doc, 'pictures', [])
    print(f"Found {len(pictures)} images in document")

    for idx, picture in enumerate(pictures):
        try:
            # Get image data
            if not hasattr(picture, 'image') or picture.image is None:
                continue

            # Get image PIL object
            pil_image = picture.image.pil_image if hasattr(picture.image, 'pil_image') else picture.image

            if pil_image is None:
                continue

            # Get dimensions
            width, height = pil_image.size if hasattr(pil_image, 'size') else (None, None)

            # Get page number and bbox
            page_number = 1
            bbox = None
            if hasattr(picture, 'prov') and picture.prov:
                prov = picture.prov[0] if len(picture.prov) > 0 else None
                if prov:
                    if hasattr(prov, 'page_no'):
                        page_number = prov.page_no
                    if hasattr(prov, 'bbox'):
                        bbox = {
                            "l": round(prov.bbox.l, 4),
                            "t": round(prov.bbox.t, 4),
                            "r": round(prov.bbox.r, 4),
                            "b": round(prov.bbox.b, 4)
                        }

            # Encode image as base64 PNG
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # OCR the image using Docling's pipeline
            # For now, we'll extract any caption/annotation text associated with the image
            ocr_text = ""
            if hasattr(picture, 'caption') and picture.caption:
                ocr_text = str(picture.caption)
            elif hasattr(picture, 'annotations') and picture.annotations:
                ocr_text = " ".join(str(a) for a in picture.annotations if a)

            # If no caption/annotations, use placeholder (XYNE will have fallback)
            if not ocr_text.strip():
                ocr_text = f"[Image {idx} on page {page_number}]"

            image_chunks.append(ImageChunk(
                text=ocr_text.strip(),
                image_base64=f"data:image/png;base64,{base64_data}",
                page_number=page_number,
                bbox=bbox,
                width=width,
                height=height
            ))

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            # Skip failed images (XYNE has fallback)
            continue

    print(f"Successfully processed {len(image_chunks)} images")
    return image_chunks


def process_document_sync(file_path: str, doc_id: str) -> Dict[str, Any]:
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
    
    # Build text chunks
    chunk_start = time.time()
    chunks = build_chunks(document)
    print(f"Chunks generated: {len(chunks)} chunks in {time.time() - chunk_start:.2f}s")
    
    # Extract and OCR images
    image_start = time.time()
    image_chunks = extract_images_as_base64(document)
    print(f"Images processed: {len(image_chunks)} images in {time.time() - image_start:.2f}s")
    
    # Format response
    result = {
        "metadata": {
            "doc_id": doc_id,
            "filename": Path(file_path).name,
            "num_pages": len(getattr(document, 'pages', [])),
            "num_images": len(image_chunks),
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
        ],
        "image_chunks": [
            {
                "text": ic.text,
                "page_number": ic.page_number,
                "bbox": ic.bbox,
                "width": ic.width,
                "height": ic.height
            }
            for ic in image_chunks
        ],
        "images": {
            f"img_{idx}": ic.image_base64
            for idx, ic in enumerate(image_chunks)
        }
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
async def process_document(file: UploadFile = File(...), doc_id: str = Form(...)):
    """
    Process PDF document and return structured TOC and chunks

    Args:
        file: PDF file to process
        doc_id: Document ID provided by caller

    Returns JSON with:
    - metadata: document info
    - toc: structured table of contents with hierarchy
    - chunks: contextualized text chunks with metadata
    - image_chunks: image metadata + text descriptions (for Vespa search)
    - images: base64-encoded images keyed as img_0, img_1, etc.
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
        result = await loop.run_in_executor(None, process_document_sync, tmp_path, doc_id)
        
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
