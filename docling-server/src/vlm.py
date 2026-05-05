
import base64
import io
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image

from config import (
    GPU_INSTANCE_URL,
    IMAGE_VLM_PROMPT,
    SUPPORTED_VLM_PRESETS,
    VLM_ACCESS_TOKEN,
    VLM_IMAGE_MAX_DIM,
    VLM_MAX_TOKENS,
    VLM_MODEL,
    VLM_PORT,
    VLM_PRESET,
    VLM_TIMEOUT,
    VLM_URL,
)
from models import VlmConfig


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_vlm_config() -> Optional[VlmConfig]:
    """Build a VlmConfig from environment variables.

    Returns None (with a descriptive log) for any misconfiguration so the
    server starts cleanly without a VLM — image processing is simply skipped.
    """
    if not VLM_PRESET:
        print("VLM disabled: VLM_PRESET not set")
        return None

    if VLM_PRESET not in SUPPORTED_VLM_PRESETS:
        print(f"VLM disabled: unsupported preset '{VLM_PRESET}' "
              f"(supported: {sorted(SUPPORTED_VLM_PRESETS)})")
        return None

    try:
        from docling.datamodel.pipeline_options import VlmConvertOptions
        from docling.datamodel.vlm_engine_options import VlmEngineType

        preset     = VlmConvertOptions.get_preset(VLM_PRESET)
        api_params = preset.model_spec.get_api_params(VlmEngineType.API)
    except Exception as exc:
        print(f"VLM disabled: failed to load preset '{VLM_PRESET}': {exc}")
        return None

    model = VLM_MODEL or str(api_params.get("model") or "")
    if not model:
        print(f"VLM disabled: could not resolve model name for preset '{VLM_PRESET}'")
        return None

    endpoint   = VLM_URL or f"{GPU_INSTANCE_URL}:{VLM_PORT}/v1/chat/completions"
    model      = _resolve_served_model(endpoint, model, VLM_TIMEOUT, token=VLM_ACCESS_TOKEN)
    max_tokens = int(api_params.get("max_tokens") or VLM_MAX_TOKENS)

    print(f"VLM ready: preset={VLM_PRESET}, model={model}, url={endpoint}, "
          f"auth={'yes' if VLM_ACCESS_TOKEN else 'no'}")

    return VlmConfig(
        preset=VLM_PRESET,
        endpoint_url=endpoint,
        model=model,
        timeout=VLM_TIMEOUT,
        max_tokens=max_tokens,
        image_prompt=IMAGE_VLM_PROMPT,
        token=VLM_ACCESS_TOKEN,
    )


def _resolve_served_model(
    endpoint_url: str,
    preferred: str,
    timeout: float,
    token: str = "",
) -> str:
    """Return the model ID actually being served, falling back to *preferred*."""
    parsed     = urlparse(endpoint_url)
    models_url = urlunparse(parsed._replace(path="/v1/models", query="", fragment=""))
    headers    = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        resp = requests.get(models_url, headers=headers, timeout=min(timeout, 10.0))
        resp.raise_for_status()
        models = resp.json().get("data", [])
    except Exception as exc:
        print(f"VLM model discovery skipped ({models_url}): {exc}")
        return preferred

    preferred_norm = _normalize(preferred)

    for m in models:
        mid   = str(m.get("id")   or "").strip()
        mroot = str(m.get("root") or "").strip()
        if not mid:
            continue
        if preferred in (mid, mroot) or preferred_norm in (_normalize(mid), _normalize(mroot)):
            return mid

    if len(models) == 1:
        sole = str(models[0].get("id") or "").strip()
        if sole:
            print(f"VLM: '{preferred}' not found in /v1/models; using sole model '{sole}'")
            return sole

    return preferred


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Return a JPEG data-URL for *img*, resizing proportionally if needed.

    JPEG is ~10× smaller than PNG for document content, keeping VLM payloads
    small. Transparent images are composited onto white before encoding.
    """
    max_dim = VLM_IMAGE_MAX_DIM
    w, h    = img.size

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img   = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.LANCZOS,
        )

    # JPEG does not support alpha – composite onto white
    if img.mode in ("RGBA", "LA", "P"):
        bg   = Image.new("RGB", img.size, (255, 255, 255))
        mask = img.split()[-1] if img.mode in ("RGBA", "LA") else None
        bg.paste(img.convert("RGB"), mask=mask)
        img  = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_vlm(config: VlmConfig, img: Image.Image, prompt: str) -> str:
    """Send *img* and *prompt* to the VLM and return the cleaned response text."""
    headers = {"Authorization": f"Bearer {config.token}"} if config.token else {}
    payload = {
        "model":      config.model,
        "max_tokens": config.max_tokens,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": encode_image_jpeg(img)}},
            ],
        }],
    }
    resp = requests.post(
        config.endpoint_url, json=payload, headers=headers, timeout=config.timeout
    )
    resp.raise_for_status()
    raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    return _clean_response(raw, prompt=prompt)


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------

def _clean_response(content: Any, prompt: str = "") -> str:
    """Strip fences, prompt echo, and runaway repetition from a VLM reply."""
    text = str(content or "").strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text  = "\n".join(inner).strip()

    # Strip verbatim prompt echo
    if prompt:
        stripped = prompt.strip()
        if text.startswith(stripped):
            text = text[len(stripped):].lstrip()

    return _truncate_repetition(text)


def _truncate_repetition(text: str, max_repeats: int = 3) -> str:
    """Detect and truncate two classes of generation loops.

    1. *Line-level*: the same line appears more than *max_repeats* times.
    2. *Character-level*: a single character repeated 20+ times consecutively.
    """
    # Collapse long single-char runs (e.g. "!!!!!!!" → "!!!")
    text = re.sub(r"(.)\1{19,}", lambda m: m.group(1) * 3, text)

    # Stop at the first line seen more than max_repeats times
    counts: Dict[str, int] = {}
    result: List[str]      = []
    for line in text.splitlines():
        key = line.strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > max_repeats:
                break
        result.append(line)

    return "\n".join(result).strip()
