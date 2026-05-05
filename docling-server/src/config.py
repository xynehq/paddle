"""Environment-driven configuration for the Docling processing service."""

import os

# ---------------------------------------------------------------------------
# VLM connection
# ---------------------------------------------------------------------------
GPU_INSTANCE_URL  = os.getenv("GPU_INSTANCE_URL", "http://10.8.0.100")
VLM_PRESET        = os.getenv("VLM_PRESET", "").strip().lower()
VLM_URL           = os.getenv("VLM_URL", "").strip()
VLM_MODEL         = os.getenv("VLM_MODEL", "").strip()
VLM_PORT          = os.getenv("VLM_PORT", "8000").strip()
VLM_TIMEOUT       = float(os.getenv("VLM_TIMEOUT", "60.0"))
VLM_MAX_TOKENS    = int(os.getenv("VLM_MAX_TOKENS", "4096"))
VLM_ACCESS_TOKEN  = os.getenv("VLM_ACCESS_TOKEN", "").strip()
VLM_IMAGE_MAX_DIM = int(os.getenv("VLM_IMAGE_MAX_DIM", "2048"))

IMAGE_VLM_PROMPT = os.getenv("IMAGE_VLM_PROMPT", "Read all text in this image.")

# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------
# Pages with fewer extracted chars than this threshold are treated as
# scanned / image-only and sent to the VLM as full-page images.
SCANNED_PAGE_CHAR_THRESHOLD = int(os.getenv("SCANNED_PAGE_CHAR_THRESHOLD", "50"))

SUPPORTED_VLM_PRESETS = frozenset({"granite_docling", "lightonocr"})
