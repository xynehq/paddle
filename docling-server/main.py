import asyncio
import base64
import io
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import urllib3
import requests
import torch
import uvicorn
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    VlmConvertOptions,
)
from docling.datamodel.vlm_engine_options import VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

GPU_INSTANCE_URL = os.getenv("GPU_INSTANCE_URL", "http://10.8.0.100")
VLM_PRESET = os.getenv("VLM_PRESET", "").strip().lower()
VLM_URL = os.getenv("VLM_URL", "").strip()
VLM_MODEL = os.getenv("VLM_MODEL", "").strip()
VLM_PORT = os.getenv("VLM_PORT", "8000").strip()
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "60.0"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "4096"))
VLM_ACCESS_TOKEN = os.getenv("VLM_ACCESS_TOKEN", "").strip()
VLM_SSL_VERIFY = os.getenv("VLM_SSL_VERIFY", "true").strip().lower() not in ("false", "0", "no")
VLM_CONCURRENCY = max(1, int(os.getenv("VLM_CONCURRENCY", "8")))

if not VLM_SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

IMAGE_VLM_PROMPT = os.getenv("IMAGE_VLM_PROMPT", "Read all text in this image.")

SUPPORTED_VLM_PRESETS = {"granite_docling", "lightonocr"}

@dataclass
class VlmConfig:
    preset: str
    endpoint_url: str
    model: str
    timeout: float
    max_tokens: int
    image_prompt: str
    token: str = ""  # Bearer token for Authorization header


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
    bbox: Optional[Dict[str, float]] = None


@dataclass
class ImageChunk:
    text: str
    image_base64: str
    page_number: int
    bbox: Optional[Dict[str, float]] = None
    width: Optional[int] = None
    height: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    converter, chunker, vlm_config = initialize_models()
    app.state.doc_converter = converter
    app.state.chunker = chunker
    app.state.vlm_config = vlm_config
    yield


app = FastAPI(title="Docling Document Processing Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------


def normalize_model_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def resolve_served_model(endpoint_url: str, preferred: str, timeout: float, token: str = "") -> str:
    """Return the model id actually being served, falling back to *preferred*."""
    parsed = urlparse(endpoint_url)
    models_url = urlunparse(parsed._replace(path="/v1/models", query="", fragment=""))
    preferred_norm = normalize_model_name(preferred)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(models_url, headers=headers, timeout=min(timeout, 10.0), verify=VLM_SSL_VERIFY)
        resp.raise_for_status()
        models = resp.json().get("data", [])
    except Exception as exc:
        print(f"VLM model discovery skipped ({models_url}): {exc}")
        return preferred

    for m in models:
        mid = str(m.get("id") or "").strip()
        mroot = str(m.get("root") or "").strip()
        if not mid:
            continue
        if preferred in (mid, mroot) or preferred_norm in (normalize_model_name(mid), normalize_model_name(mroot)):
            return mid

    if len(models) == 1:
        sole = str(models[0].get("id") or "").strip()
        if sole:
            print(f"VLM: '{preferred}' not found in /v1/models; using sole model '{sole}'")
            return sole

    return preferred


def build_vlm_config() -> Optional[VlmConfig]:
    """Build VlmConfig from environment variables.

    Returns None (with a log message) for any misconfiguration so the server
    starts cleanly without a VLM — image processing is simply skipped.
    """
    if not VLM_PRESET:
        print("VLM disabled: VLM_PRESET not set")
        return None

    if VLM_PRESET not in SUPPORTED_VLM_PRESETS:
        print(f"VLM disabled: unsupported preset '{VLM_PRESET}' (supported: {sorted(SUPPORTED_VLM_PRESETS)})")
        return None

    try:
        preset = VlmConvertOptions.get_preset(VLM_PRESET)
        api_params = preset.model_spec.get_api_params(VlmEngineType.API)
    except Exception as exc:
        print(f"VLM disabled: failed to load preset '{VLM_PRESET}': {exc}")
        return None

    model = VLM_MODEL or str(api_params.get("model") or "")
    if not model:
        print(f"VLM disabled: could not resolve model name for preset '{VLM_PRESET}'")
        return None

    endpoint = VLM_URL or f"{GPU_INSTANCE_URL}:{VLM_PORT}/v1/chat/completions"
    model = resolve_served_model(endpoint, model, VLM_TIMEOUT, token=VLM_ACCESS_TOKEN)
    max_tokens = int(api_params.get("max_tokens") or VLM_MAX_TOKENS)

    print(f"VLM ready: preset={VLM_PRESET}, model={model}, url={endpoint}, auth={'yes' if VLM_ACCESS_TOKEN else 'no'}")
    return VlmConfig(
        preset=VLM_PRESET,
        endpoint_url=endpoint,
        model=model,
        timeout=VLM_TIMEOUT,
        max_tokens=max_tokens,
        image_prompt=IMAGE_VLM_PROMPT,
        token=VLM_ACCESS_TOKEN,
    )


# Maximum pixel dimension sent to the VLM.  Larger images are scaled down
# proportionally before encoding.  2048 px is sufficient for OCR-quality work
# while keeping the payload under typical API limits.
VLM_IMAGE_MAX_DIM = int(os.getenv("VLM_IMAGE_MAX_DIM", "2048"))


def encode_image_jpeg(img: Image.Image, max_dim: int = VLM_IMAGE_MAX_DIM, quality: int = 90) -> str:
    """Return a JPEG data-URL for *img*, resizing proportionally if needed.

    JPEG is ~10x smaller than PNG for document content, keeping VLM payloads
    small.  Transparent images are composited onto white before encoding.
    """
    # Resize proportionally if needed
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    # JPEG does not support alpha; composite onto white
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        mask = img.split()[-1] if img.mode in ("RGBA", "LA") else None
        bg.paste(img.convert("RGB"), mask=mask)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def truncate_repetition(text: str, max_repeats: int = 3) -> str:
    """Detect and truncate two classes of model generation loops:

    1. *Line-level loops*: the same line appears more than *max_repeats* times
       (e.g. "LIFE INSURANCE CORPORATION OF INDIA" × 400).
    2. *Character-level loops*: a single character is repeated consecutively
       beyond a sane limit within one line (e.g. "!!!!!!!!..." × 5000).
       We allow up to 20 consecutive identical characters — enough for any
       legitimate use (e.g. a horizontal rule "---") but well short of a loop.
    """
    # Character-level: collapse any run of 20+ identical chars to just 3
    text = re.sub(r'(.)\1{19,}', lambda m: m.group(1) * 3, text)

    # Line-level: stop at the first line that has been seen > max_repeats times
    lines = text.splitlines()
    counts: Dict[str, int] = {}
    result: List[str] = []
    for line in lines:
        key = line.strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > max_repeats:
                break
        result.append(line)
    return "\n".join(result).strip()


def clean_vlm_response(content: Any, prompt: str = "") -> str:
    """Clean the raw string from an OpenAI-compatible chat completion.

    - Strips markdown code fences.
    - Strips verbatim prompt echo (some models echo the instruction).
    - Truncates runaway repetition loops.
    """
    text = str(content or "").strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(inner).strip()

    # Strip verbatim prompt echo (exact full-prompt match only)
    if prompt:
        prompt_stripped = prompt.strip()
        if text.startswith(prompt_stripped):
            text = text[len(prompt_stripped):].lstrip()

    return truncate_repetition(text)


def call_vlm(config: VlmConfig, img: Image.Image, prompt: str) -> str:
    """Send *img* and *prompt* to the VLM and return the cleaned response text."""
    headers = {"Authorization": f"Bearer {config.token}"} if config.token else {}
    payload = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_image_jpeg(img)}},
                ],
            }
        ],
    }
    resp = requests.post(config.endpoint_url, json=payload, headers=headers, timeout=config.timeout, verify=VLM_SSL_VERIFY)
    resp.raise_for_status()
    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    return clean_vlm_response(content, prompt=prompt)


# ---------------------------------------------------------------------------
# Document enrichment
# ---------------------------------------------------------------------------


MISSING_TEXT_RE = re.compile(r'^(<!--\s*missing-text\s*-->\s*)+$', re.MULTILINE)


def is_placeholder(text: str) -> bool:
    """Return True if text consists entirely of docling missing-text tokens."""
    return not text.strip() or bool(MISSING_TEXT_RE.fullmatch(text.strip()))


def extract_tables(doc) -> Dict[str, str]:
    """Convert all tables to GFM Markdown using docling's native cell structure.

    Returns a mapping of table self_ref → markdown text.
    """
    replacements: Dict[str, str] = {}
    tables = getattr(doc, "tables", [])
    print(f"Tables: extracting {len(tables)} table(s)")
    for i, tbl in enumerate(tables):
        try:
            md = tbl.export_to_markdown(doc=doc)
            if md and md.strip():
                replacements[tbl.self_ref] = md.strip()
                print(f"  table {i+1}/{len(tables)} [{tbl.self_ref}]: {len(md)} chars")
            else:
                print(f"  table {i+1}/{len(tables)} [{tbl.self_ref}]: empty")
        except Exception as exc:
            print(f"  table {i+1}/{len(tables)} [{tbl.self_ref}]: FAILED — {exc}")
    return replacements


def process_images_with_vlm(doc, cfg: VlmConfig, table_refs: set) -> Dict[str, str]:
    """OCR all pictures via the VLM and return a self_ref → text mapping.

    *table_refs* is the set of refs already handled by extract_tables so we
    can log clearly if there is ever an unexpected overlap.

    VLM calls run concurrently up to VLM_CONCURRENCY in-flight requests.
    """
    replacements: Dict[str, str] = {}
    pictures = list(getattr(doc, "pictures", []))
    total = len(pictures)
    if total == 0:
        print("VLM: processing 0 picture(s)")
        return replacements

    workers = min(VLM_CONCURRENCY, total)
    print(f"VLM: processing {total} picture(s) with concurrency={workers}")

    # Pre-render all images on the main thread (docling doc access not
    # guaranteed thread-safe), then send the VLM requests concurrently.
    rendered: List[Tuple[int, str, Optional[Image.Image]]] = []
    for i, pic in enumerate(pictures):
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

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ref, text in ex.map(ocr_one, rendered):
            if text:
                replacements[ref] = text
    return replacements


# Minimum characters of extracted text on a page before we consider it
# "digitally readable" and skip sending it to the VLM.
# Pages below this threshold are treated as scanned / image-only.
SCANNED_PAGE_CHAR_THRESHOLD = int(os.getenv("SCANNED_PAGE_CHAR_THRESHOLD", "50"))


def process_scanned_pages_with_vlm(doc, cfg: VlmConfig) -> Dict[int, str]:
    """Detect pages with no extracted text and OCR them via the external VLM.

    Because local OCR is disabled, any page that came from a scanned PDF will
    have zero text items.  We render the full page image (available because
    ``generate_page_images=True``) and send it to the VLM.

    Returns a mapping of page_no → VLM text for every scanned page.
    """
    # Count characters extracted by the PDF backend per page
    page_chars: Dict[int, int] = {}
    for item in getattr(doc, "texts", []):
        txt = getattr(item, "text", "") or ""
        for prov in getattr(item, "prov", []):
            page_chars[prov.page_no] = page_chars.get(prov.page_no, 0) + len(txt)

    pages = getattr(doc, "pages", {})
    scanned = [
        pg_no for pg_no in sorted(pages)
        if page_chars.get(pg_no, 0) < SCANNED_PAGE_CHAR_THRESHOLD
    ]

    if not scanned:
        print("Page VLM: all pages have embedded text, no scanned pages detected")
        return {}

    workers = min(VLM_CONCURRENCY, len(scanned))
    print(f"Page VLM: {len(scanned)}/{len(pages)} scanned page(s) detected "
          f"(< {SCANNED_PAGE_CHAR_THRESHOLD} chars) concurrency={workers}: {scanned}")

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


def chunk_pages(chunk) -> List[int]:
    pages = set()
    for item in getattr(chunk.meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []):
            if hasattr(prov, "page_no"):
                pages.add(prov.page_no)
    return sorted(pages)


def item_bbox(item) -> Optional[Dict[str, float]]:
    provs = getattr(item, "prov", [])
    if provs and hasattr(provs[0], "bbox"):
        b = provs[0].bbox
        return {"l": round(b.l, 4), "t": round(b.t, 4), "r": round(b.r, 4), "b": round(b.b, 4)}
    return None


def item_page(item) -> int:
    provs = getattr(item, "prov", [])
    return provs[0].page_no if provs and hasattr(provs[0], "page_no") else 1


def build_chunks(
    doc,
    replacements: Dict[str, str],
    chunker: HybridChunker,
    page_vlm_text: Optional[Dict[int, str]] = None,
) -> List[DocumentChunk]:
    """Build chunks using HybridChunker directly.

    HybridChunker never yields PictureItems — it only processes text, table,
    heading, and title items.  Picture VLM replacements are therefore injected
    as extra chunks after the main loop for any picture ref that was not
    absorbed into a text chunk.
    """
    seen_refs: set = set()
    chunks = []
    n_from_text = 0
    n_from_vlm = 0
    n_skipped_placeholder = 0
    n_skipped_duplicate = 0
    pages_covered: set = set()

    for chunk in chunker.chunk(doc):
        items = list(getattr(chunk.meta, "doc_items", []) or [])

        # Single-item chunk: use replacement if available, else chunk.text
        if len(items) == 1:
            ref = getattr(items[0], "self_ref", None)
            if ref and ref in replacements:
                if ref in seen_refs:
                    n_skipped_duplicate += 1
                    continue  # duplicate sub-chunk of the same table/picture
                seen_refs.add(ref)
                text = replacements[ref]
                source = "vlm"
            else:
                text = chunk.text
                source = "text"
        else:
            for item in items:
                ref = getattr(item, "self_ref", None)
                if ref and ref in replacements:
                    seen_refs.add(ref)
            text = chunk.text
            source = "text"

        text = text.strip() if text else ""
        if not text or is_placeholder(text):
            n_skipped_placeholder += 1
            continue

        headings = list(getattr(chunk.meta, "headings", None) or [])
        pages = chunk_pages(chunk) or [1]
        pages_covered.update(pages)
        bbox = next(
            (item_bbox(item) for item in items if item_bbox(item)),
            None,
        )
        chunks.append(DocumentChunk(
            text=text,
            headings=headings,
            page_numbers=pages,
            bbox=bbox,
        ))
        if source == "vlm":
            n_from_vlm += 1
        else:
            n_from_text += 1

    # Track picture refs as seen so the scanned-page loop below doesn't
    # double-inject them, but do NOT add them to chunks — picture VLM text
    # lives only in image_chunks (returned by extract_images).
    pic_refs_injected = 0
    for pic in getattr(doc, "pictures", []):
        ref = getattr(pic, "self_ref", None)
        if ref in seen_refs or ref not in replacements:
            continue
        page = item_page(pic)
        pages_covered.add(page)
        seen_refs.add(ref)
        pic_refs_injected += 1

    # Inject VLM text for fully-scanned pages that produced no HybridChunker output
    pages_injected = 0
    for pg_no, text in sorted((page_vlm_text or {}).items()):
        text = text.strip()
        if not text or is_placeholder(text):
            continue
        # Only inject if this page has no chunk coverage yet
        # if pg_no in pages_covered:
        #     continue
        pages_covered.add(pg_no)
        chunks.append(DocumentChunk(
            text=text,
            headings=[],
            page_numbers=[pg_no],
            bbox=None,
        ))
        n_from_vlm += 1
        pages_injected += 1

    total_pages = len(getattr(doc, "pages", {}))
    uncovered = sorted(set(range(1, total_pages + 1)) - pages_covered)

    print(f"Chunks: {len(chunks)} total  "
          f"({n_from_text} from text layer, {n_from_vlm} from VLM scanned pages"
          f"  |  {pic_refs_injected} picture(s) in image_chunks only)")
    print(f"  skipped: {n_skipped_placeholder} empty/placeholder, "
          f"{n_skipped_duplicate} duplicate table sub-chunks")
    print(f"  page coverage: {len(pages_covered)}/{total_pages} pages have chunks"
          + (f"  |  no-chunk pages: {uncovered}" if uncovered else ""))

    return chunks


# ---------------------------------------------------------------------------
# TOC extraction
# ---------------------------------------------------------------------------


def build_toc(doc) -> List[TocEntry]:
    entries: List[TocEntry] = []
    counter = 0

    for item in getattr(doc, "texts", []):
        if getattr(item, "label", None) not in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            continue
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue

        level = getattr(item, "level", None) or 1

        counter += 1
        entry = TocEntry(
            section_number=str(counter),
            section_title=text,
            page_number=item_page(item),
            level=level,
            bbox=item_bbox(item),
        )
        # Find nearest parent
        for j in range(len(entries) - 1, -1, -1):
            if entries[j].level < entry.level:
                entry.parent_index = j
                break
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def extract_images(doc, replacements: Dict[str, str]) -> List[ImageChunk]:
    pictures = getattr(doc, "pictures", [])
    print(f"Extracting {len(pictures)} images")
    chunks = []

    for idx, pic in enumerate(pictures):
        try:
            # Use docling's page-image pipeline to render the picture.
            pil = pic.get_image(doc=doc)
            if pil is None:
                continue

            width, height = pil.size

            # Encode as JPEG data-URL (resized if needed) – consistent with
            # what is sent to the VLM and far smaller than raw PNG.
            b64 = encode_image_jpeg(pil)

            # Prefer VLM OCR text, then existing metadata, then a placeholder
            text = (
                replacements.get(pic.self_ref)
                or (str(pic.meta.description.text) if getattr(pic, "meta", None) and getattr(pic.meta, "description", None) else "")
                or (str(pic.caption) if getattr(pic, "caption", None) else "")
                or f"[Image {idx} on page {item_page(pic)}]"
            ).strip()

            chunks.append(ImageChunk(
                text=text,
                image_base64=b64,
                page_number=item_page(pic),
                bbox=item_bbox(pic),
                width=width,
                height=height,
            ))
        except Exception as exc:
            print(f"Image {idx} skipped: {exc}")

    print(f"Extracted {len(chunks)} images")
    return chunks


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_models() -> Tuple[DocumentConverter, HybridChunker, Optional[VlmConfig]]:
    print("Initializing Docling models...")

    if torch.cuda.is_available():
        device = AcceleratorDevice.CUDA
    elif os.uname().sysname == "Darwin":
        device = AcceleratorDevice.MPS
    else:
        device = AcceleratorDevice.CPU

    docling_threads = max(1, int(os.getenv("DOCLING_NUM_THREADS", "16")))
    layout_batch = max(1, int(os.getenv("DOCLING_LAYOUT_BATCH", "8")))
    table_batch = max(1, int(os.getenv("DOCLING_TABLE_BATCH", "8")))

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=docling_threads, device=device
    )
    pipeline_options.layout_batch_size = layout_batch
    pipeline_options.table_batch_size = table_batch
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = False  # tables use export_to_markdown(), not VLM
    # Disable local OCR entirely — no Tesseract/EasyOCR model needed.
    # Scanned pages (no embedded text) are detected post-conversion and sent
    # to the external VLM instead.  Digital PDFs still have their text layer
    # extracted natively by the PDF backend without any OCR.
    pipeline_options.do_ocr = False
    # Render full page images so scanned pages can be sent to the VLM.
    # scale=2.0 doubles the default 72 DPI → ~144 DPI, good enough for OCR.
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = 2.0

    print(
        f"Docling: device={device.value} num_threads={docling_threads} "
        f"layout_batch={layout_batch} table_batch={table_batch} "
        f"cuda_available={torch.cuda.is_available()}"
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    chunker = HybridChunker(tokenizer="./jina-tokenizer", max_tokens=1024, merge_peers=True)
    vlm_config = build_vlm_config()

    print("Models initialized")
    return converter, chunker, vlm_config


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_document(
    file_path: str,
    doc_id: str,
    doc_converter: DocumentConverter,
    chunker: HybridChunker,
    vlm_config: Optional[VlmConfig],
) -> Dict[str, Any]:
    t0 = time.time()

    conv = doc_converter.convert(file_path)
    doc = conv.document
    total_pages = len(getattr(doc, "pages", {}))
    n_texts = len(getattr(doc, "texts", []))
    n_tables = len(getattr(doc, "tables", []))
    n_pictures = len(getattr(doc, "pictures", []))
    print(f"Converted in {time.time() - t0:.2f}s  |  "
          f"{total_pages} pages, {n_texts} text items, "
          f"{n_tables} tables, {n_pictures} pictures")

    toc = build_toc(doc)
    print(f"TOC: {len(toc)} entries")

    t1 = time.time()

    # Tables are always extracted via docling's native Markdown (no VLM needed)
    replacements = extract_tables(doc)
    tables_replaced = len(replacements)

    # Images are OCR'd via VLM when one is configured; otherwise skipped
    pictures_replaced = 0
    page_vlm_text: Dict[int, str] = {}
    if vlm_config:
        image_replacements = process_images_with_vlm(doc, vlm_config, set(replacements))
        replacements.update(image_replacements)
        pictures_replaced = len(image_replacements)
        # Scanned pages (no embedded text) are sent as full-page images to the VLM
        page_vlm_text = process_scanned_pages_with_vlm(doc, vlm_config)
    else:
        print("VLM not configured — skipping image OCR and scanned page processing")

    vlm_stats = {
        "tables_replaced": tables_replaced,
        "pictures_replaced": pictures_replaced,
        "scanned_pages_ocrd": len(page_vlm_text),
    }
    print(f"Enrichment: {len(replacements)} replacements + {len(page_vlm_text)} scanned pages in {time.time() - t1:.2f}s")

    chunks = build_chunks(doc, replacements, chunker, page_vlm_text=page_vlm_text)

    images = extract_images(doc, replacements)
    print(f"Images: {len(images)}")

    return {
        "metadata": {
            "doc_id": doc_id,
            "filename": Path(file_path).name,
            "num_pages": len(getattr(doc, "pages", [])),
            "num_images": len(images),
            "processing_time": round(time.time() - t0, 2),
            "has_toc": bool(toc),
            "vlm": {
                "enabled": vlm_config is not None,
                "preset": vlm_config.preset if vlm_config else None,
                "model": vlm_config.model if vlm_config else None,
                **vlm_stats,
            },
        },
        "toc": {
            "entries": [
                {
                    "section_number": e.section_number,
                    "section_title": e.section_title,
                    "page_number": e.page_number,
                    "level": e.level,
                    "bbox": e.bbox,
                    "parent_index": e.parent_index,
                }
                for e in toc
            ]
        },
        "chunks": [
            {
                "text": c.text,
                "headings": c.headings,
                "page_numbers": c.page_numbers,
                "bbox": c.bbox,
            }
            for c in chunks
        ],
        "image_chunks": [
            {
                "text": ic.text,
                "page_number": ic.page_number,
                "bbox": ic.bbox,
                "width": ic.width,
                "height": ic.height,
            }
            for ic in images
        ],
        "images": {f"img_{i}": ic.image_base64 for i, ic in enumerate(images)},
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    if not getattr(app.state, "doc_converter", None) or not getattr(app.state, "chunker", None):
        raise HTTPException(status_code=503, detail="Models not initialized")
    return {"status": "ok", "models_loaded": True}


@app.post("/process")
async def process_document_endpoint(file: UploadFile = File(...), doc_id: str = Form(...)):
    """
    Process a PDF and return structured TOC, text chunks, and images.

    Returns:
    - metadata: document info and processing stats
    - toc: table of contents with hierarchy
    - chunks: contextualized text chunks with page/section metadata
    - image_chunks: image metadata + descriptions (for search indexing)
    - images: base64-encoded images keyed as img_0, img_1, …
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    print(f"Processing file: {file.filename} (doc_id={doc_id})", flush=True)
    suffix = Path(file.filename).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            process_document,
            tmp_path,
            doc_id,
            app.state.doc_converter,
            app.state.chunker,
            app.state.vlm_config,
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)