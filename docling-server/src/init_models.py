import os
from typing import Callable, Optional, Tuple

import torch
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from semchunk import chunkerify

from config import MAX_CHUNK_TOKENS
from models import VlmConfig
from vlm import build_vlm_config


def _select_device() -> AcceleratorDevice:
    if torch.cuda.is_available():
        return AcceleratorDevice.CUDA
    if os.uname().sysname == "Darwin":
        return AcceleratorDevice.MPS
    return AcceleratorDevice.CPU


def _build_pipeline_options() -> PdfPipelineOptions:
    device = _select_device()
    docling_threads = max(1, int(os.getenv("DOCLING_NUM_THREADS", "16")))
    layout_batch = max(1, int(os.getenv("DOCLING_LAYOUT_BATCH", "8")))
    table_batch = max(1, int(os.getenv("DOCLING_TABLE_BATCH", "8")))

    opts = PdfPipelineOptions()
    opts.accelerator_options     = AcceleratorOptions(num_threads=docling_threads, device=device)
    opts.layout_batch_size       = layout_batch
    opts.table_batch_size        = table_batch
    opts.generate_picture_images = True
    opts.generate_table_images   = False   # tables use export_to_markdown(), not VLM
    # Disable local OCR (Tesseract / EasyOCR) entirely.
    # Scanned pages are detected post-conversion and sent to the external VLM.
    opts.do_ocr                  = False
    # Render full-page images so scanned pages can be OCR'd by the VLM.
    # scale=2.0 doubles the default 72 DPI → ~144 DPI, sufficient for OCR.
    opts.generate_page_images    = True
    opts.images_scale            = 2.0

    print(
        f"Docling: device={device.value} num_threads={docling_threads} "
        f"layout_batch={layout_batch} table_batch={table_batch} "
        f"cuda_available={torch.cuda.is_available()}"
    )

    return opts


def initialize_models() -> Tuple[DocumentConverter, HybridChunker, Optional[VlmConfig], Callable[[str], list[str]]]:
    print("Initializing Docling models...")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_build_pipeline_options())
        }
    )
    
    # HybridChunker for digital PDFs with document structure
    # Using intfloat/multilingual-e5-large tokenizer
    tokenizer_path = "/app/e5-tokenizer" if os.path.exists("/app/e5-tokenizer") else "./e5-tokenizer"
    hybrid_chunker = HybridChunker(tokenizer=tokenizer_path, max_tokens=MAX_CHUNK_TOKENS, merge_peers=True)
    
    # semchunk for scanned page VLM text (raw text without structure)
    sem_chunker = chunkerify(tokenizer_path, chunk_size=MAX_CHUNK_TOKENS)
    
    vlm_config = build_vlm_config()

    print(f"Chunking: max_tokens={MAX_CHUNK_TOKENS} (HybridChunker for digital, semchunk for scanned)")
    print("Models initialized")
    return converter, hybrid_chunker, vlm_config, sem_chunker
