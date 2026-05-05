import os
from typing import Optional, Tuple

import torch
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from models import VlmConfig
from vlm import build_vlm_config


def _select_device() -> AcceleratorDevice:
    if torch.cuda.is_available():
        return AcceleratorDevice.CUDA
    if os.uname().sysname == "Darwin":
        return AcceleratorDevice.MPS
    return AcceleratorDevice.CPU


def _build_pipeline_options() -> PdfPipelineOptions:
    opts = PdfPipelineOptions()
    opts.accelerator_options     = AcceleratorOptions(num_threads=4, device=_select_device())
    opts.generate_picture_images = True
    opts.generate_table_images   = False   # tables use export_to_markdown(), not VLM
    # Disable local OCR (Tesseract / EasyOCR) entirely.
    # Scanned pages are detected post-conversion and sent to the external VLM.
    opts.do_ocr                  = False
    # Render full-page images so scanned pages can be OCR'd by the VLM.
    # scale=2.0 doubles the default 72 DPI → ~144 DPI, sufficient for OCR.
    opts.generate_page_images    = True
    opts.images_scale            = 2.0
    return opts


def initialize_models() -> Tuple[DocumentConverter, HybridChunker, Optional[VlmConfig]]:
    print("Initializing Docling models...")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_build_pipeline_options())
        }
    )
    chunker    = HybridChunker(tokenizer="../jina-tokenizer", max_tokens=1024, merge_peers=True)
    vlm_config = build_vlm_config()

    print("Models initialized")
    return converter, chunker, vlm_config
