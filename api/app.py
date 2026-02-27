import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile


@dataclass(frozen=True)
class Settings:
    triton_http_url: str
    triton_model: str
    blip_http_url: str
    blip_model: str
    status_url: str
    paddlex_status_url: str
    pipeline_config_path: str
    paddlex_config_pbtxt: str
    blip_config_pbtxt: str
    swagger_enabled: bool
    max_upload_mb: int
    triton_timeout_seconds: float
    caption_enabled_flag: bool


def get_settings() -> Settings:
    return Settings(
        triton_http_url=os.environ.get(
            "TRITON_HTTP_URL", "http://paddlex-server:8000"
        ),
        triton_model=os.environ.get("TRITON_MODEL", "layout-parsing"),
        blip_http_url=os.environ.get(
            "TRITON_BLIP_HTTP_URL", "http://blip-server:8003"
        ),
        blip_model=os.environ.get("TRITON_BLIP_MODEL", "blip-caption"),
        status_url=os.environ.get(
            "STATUS_URL", "http://paddlex-server:8081/instance_status"
        ),
        paddlex_status_url=os.environ.get(
            "PADDLEX_STATUS_URL",
            "http://paddlex-server:8081/paddlex_instance_status",
        ),
        pipeline_config_path=os.environ.get(
            "PADDLEX_HPS_PIPELINE_CONFIG_PATH", "server/pipeline_config.yaml"
        ),
        paddlex_config_pbtxt=os.environ.get(
            "PADDLEX_CONFIG_PBTXT", "config_gpu_paddlex.pbtxt"
        ),
        blip_config_pbtxt=os.environ.get(
            "BLIP_CONFIG_PBTXT", "config_gpu_blip.pbtxt"
        ),
        swagger_enabled=os.environ.get("SWAGGER_ENABLED", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "50")),
        triton_timeout_seconds=float(
            os.environ.get("TRITON_TIMEOUT_SECONDS", "30")
        ),
        caption_enabled_flag=os.environ.get(
            "IMAGE_CAPTIONING_ENABLED", "true"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
    )


def _pbtxt_count(path: str) -> Optional[int]:
    try:
        contents = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    matches = re.findall(r"\bcount\s*:\s*(\d+)", contents)
    try:
        return sum(int(m) for m in matches)
    except Exception:
        return None


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise HTTPException(500, f"Failed to read config {path}: {exc}") from exc


async def _call_triton_model(
    url: str,
    model: str,
    payload: Dict[str, Any],
    timeout: float,
    unavailable_detail: Optional[str] = None,
) -> Dict[str, Any]:
    full_url = url.rstrip("/") + f"/v2/models/{model}/infer"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(full_url, json=payload)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, resp.text)
            try:
                return resp.json()
            except json.JSONDecodeError as exc:
                raise HTTPException(502, f"Invalid Triton response: {exc}") from exc
    except httpx.RequestError as exc:
        detail = (
            unavailable_detail
            or f"Triton model endpoint is not reachable: {full_url}"
        )
        raise HTTPException(503, detail) from exc


def _decode_triton_string_output(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = resp_json["outputs"][0]["data"][0]
        return json.loads(data)
    except Exception as exc:
        raise HTTPException(502, f"Failed to decode Triton payload: {exc}") from exc


def _ensure_ok(parsed: Dict[str, Any]) -> Dict[str, Any]:
    code = parsed.get("errorCode", 0)
    if code and code != 0:
        msg = parsed.get("errorMsg") or "Triton error"
        raise HTTPException(422, msg)
    return parsed


def _extract_clean(result: Dict[str, Any]) -> Dict[str, Any]:
    layout_results = result.get("layoutParsingResults") or []
    text_blocks: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    pages = result.get("dataInfo", {}).get("pages") or []

    for page_idx, page in enumerate(layout_results, start=1):
        pruned = page.get("prunedResult") or {}
        parsing = pruned.get("parsing_res_list") or []
        for blk in parsing:
            text_blocks.append(
                {
                    "page": page_idx,
                    "label": blk.get("block_label"),
                    "text": blk.get("block_content"),
                    "bbox": blk.get("block_bbox"),
                }
            )
        md = page.get("markdown") or {}
        for key, val in (md.get("images") or {}).items():
            images.append({"page": page_idx, "key": str(key), "data": val})

    return {
        "textBlocks": text_blocks,
        "images": images,
        "pages": pages,
    }


def _extract_markdown(result: Dict[str, Any], split_pages: bool) -> Dict[str, Any]:
    layout_results = result.get("layoutParsingResults") or []
    images: Dict[str, Any] = {}
    pages_md: List[str] = []
    combined_parts: List[str] = []

    for idx, page in enumerate(layout_results, start=1):
        md = page.get("markdown") or {}
        md_text = md.get("text") or ""
        if split_pages:
            pages_md.append(md_text)
        else:
            heading = f"# Page {idx}" if idx == 1 else f"\n\n---\n# Page {idx}"
            combined_parts.append(f"{heading}\n\n{md_text}")
        for key, val in (md.get("images") or {}).items():
            images[f"page_{idx}_{key}"] = val

    payload: Dict[str, Any] = {"images": images}
    if split_pages:
        payload["pages"] = pages_md
    else:
        payload["markdown"] = "".join(combined_parts)
    return payload


def _parse_json_payload(payload: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"Invalid payload JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(422, "payload must be a JSON object")
    return parsed


def _infer_file_type_from_upload(upload: Optional[UploadFile]) -> Optional[int]:
    if upload is None:
        return None

    content_type = (upload.content_type or "").lower()
    if content_type == "application/pdf":
        return 0
    if content_type.startswith("image/"):
        return 1

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".pdf":
        return 0
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}:
        return 1
    return None


async def _prepare_file_content(
    upload: Optional[UploadFile], inline_b64: Optional[str], max_bytes: int
) -> str:
    if upload:
        data = await upload.read()
        if len(data) > max_bytes:
            raise HTTPException(413, "Uploaded file exceeds max_upload_mb limit")
        return base64.b64encode(data).decode("ascii")
    if inline_b64:
        return inline_b64
    raise HTTPException(422, "file is required (base64 or upload)")


async def _fetch_status(url: str, timeout: float) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url)
        except Exception:
            return {}
        if resp.status_code != 200:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}


async def _invoke_layout_raw(
    payload_dict: Dict[str, Any],
    upload: Optional[UploadFile],
    config: Settings,
) -> Dict[str, Any]:
    if "fileType" not in payload_dict or payload_dict.get("fileType") is None:
        inferred_file_type = _infer_file_type_from_upload(upload)
        if inferred_file_type is not None:
            payload_dict["fileType"] = inferred_file_type

    file_b64 = await _prepare_file_content(
        upload,
        payload_dict.get("file"),
        max_bytes=config.max_upload_mb * 1024 * 1024,
    )
    body = dict(payload_dict)
    body["file"] = file_b64
    triton_payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [json.dumps(body)],
            }
        ],
        "outputs": [{"name": "output"}],
    }
    resp_json = await _call_triton_model(
        config.triton_http_url,
        config.triton_model,
        triton_payload,
        config.triton_timeout_seconds,
    )
    parsed = _decode_triton_string_output(resp_json)
    return _ensure_ok(parsed)


async def _invoke_caption(
    image_b64: Optional[str],
    upload: Optional[UploadFile],
    max_length: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    config: Settings,
) -> Dict[str, Any]:
    if not config.caption_enabled_flag:
        raise HTTPException(503, "captioning disabled via IMAGE_CAPTIONING_ENABLED")
    status = await _fetch_status(config.status_url, config.triton_timeout_seconds)
    configured = (
        status.get("blip_caption", {})
        .get("configured_instances", 0)
        if isinstance(status, dict)
        else 0
    )
    if configured <= 0:
        raise HTTPException(503, "captioning backend not configured")

    final_b64 = await _prepare_file_content(
        upload,
        image_b64,
        max_bytes=config.max_upload_mb * 1024 * 1024,
    )
    body = {
        "image_b64": final_b64,
        "max_length": max_length,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    triton_payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [json.dumps(body)],
            }
        ],
        "outputs": [{"name": "output"}],
    }
    resp_json = await _call_triton_model(
        config.blip_http_url,
        config.blip_model,
        triton_payload,
        config.triton_timeout_seconds,
        unavailable_detail=(
            "captioning server is not reachable. "
            "Please start blip-server and retry."
        ),
    )
    parsed = _decode_triton_string_output(resp_json)
    if parsed.get("error"):
        raise HTTPException(422, parsed.get("error"))
    return {"enabled": True, **parsed}


def create_app(settings: Settings) -> FastAPI:
    docs_url = "/docs" if settings.swagger_enabled else None
    redoc_url = "/redoc" if settings.swagger_enabled else None
    app = FastAPI(
        title="PaddleX Layout API",
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        swagger_ui_parameters={"docExpansion": "none"},
    )

    @app.get("/healthz")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready() -> Dict[str, Any]:
        status = await _fetch_status(settings.status_url, settings.triton_timeout_seconds)
        return {"status": "ok", "triton": bool(status)}

    @app.get("/v1/instance_status")
    async def instance_status(
        config: Settings = Depends(get_settings),
    ) -> Dict[str, Any]:
        status = await _fetch_status(config.status_url, config.triton_timeout_seconds)
        if not status:
            raise HTTPException(
                503,
                "instance_status endpoint is not reachable",
            )
        return status

    @app.get("/v1/paddlex_instance_status")
    async def paddlex_instance_status(
        config: Settings = Depends(get_settings),
    ) -> Dict[str, Any]:
        status = await _fetch_status(
            config.paddlex_status_url, config.triton_timeout_seconds
        )
        if not status:
            raise HTTPException(
                503,
                "paddlex_instance_status endpoint is not reachable",
            )
        return status

    @app.post("/v1/layout/raw/upload")
    async def layout_raw_upload(
        payload: str = Form(
            "{}",
            description="JSON string payload (without file or with optional inline file)",
        ),
        upload: UploadFile = File(..., description="Document/image file"),
        config: Settings = Depends(get_settings),
    ):
        payload_dict = _parse_json_payload(payload)
        return await _invoke_layout_raw(payload_dict, upload=upload, config=config)

    @app.post("/v1/layout/clean/upload")
    async def layout_clean_upload(
        payload: str = Form(
            "{}",
            description="JSON string payload (without file or with optional inline file)",
        ),
        upload: UploadFile = File(..., description="Document/image file"),
        config: Settings = Depends(get_settings),
    ):
        raw = await _invoke_layout_raw(
            _parse_json_payload(payload), upload=upload, config=config
        )
        result = raw.get("result") or {}
        return {
            "errorCode": 0,
            "errorMsg": "",
            "result": _extract_clean(result),
        }

    @app.post("/v1/layout/markdown/upload")
    async def layout_markdown_upload(
        payload: str = Form(
            "{}",
            description="JSON string payload (without file or with optional inline file)",
        ),
        upload: UploadFile = File(..., description="Document/image file"),
        splitPages: bool = False,
        config: Settings = Depends(get_settings),
    ):
        raw = await _invoke_layout_raw(
            _parse_json_payload(payload), upload=upload, config=config
        )
        result = raw.get("result") or {}
        md_payload = _extract_markdown(result, split_pages=splitPages)
        return {
            "errorCode": 0,
            "errorMsg": "",
            "result": md_payload,
        }

    @app.post("/v1/caption/upload")
    async def caption_upload(
        max_length: int = Form(50),
        num_beams: int = Form(3),
        no_repeat_ngram_size: int = Form(2),
        upload: UploadFile = File(..., description="Image file"),
        config: Settings = Depends(get_settings),
    ):
        return await _invoke_caption(
            image_b64=None,
            upload=upload,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            config=config,
        )

    @app.get("/v1/config")
    async def config(
        config: Settings = Depends(get_settings),
    ):
        pipeline = _load_yaml(config.pipeline_config_path)

        paddlex_count = _pbtxt_count(config.paddlex_config_pbtxt)
        blip_count = _pbtxt_count(config.blip_config_pbtxt)
        status = await _fetch_status(config.status_url, config.triton_timeout_seconds)
        paddlex_status = await _fetch_status(
            config.paddlex_status_url, config.triton_timeout_seconds
        )

        return {
            "pipelineConfigPath": config.pipeline_config_path,
            "pipeline": {
                "pipeline_name": pipeline.get("pipeline_name"),
                "batch_size": pipeline.get("batch_size"),
                "use_doc_preprocessor": pipeline.get("use_doc_preprocessor"),
                "use_chart_recognition": pipeline.get("use_chart_recognition"),
                "use_table_recognition": pipeline.get("use_table_recognition"),
                "use_formula_recognition": pipeline.get("use_formula_recognition"),
                "use_region_detection": pipeline.get("use_region_detection"),
                "Serving": pipeline.get("Serving", {}),
            },
            "instances": {
                "pbtxt": {
                    "layout_parsing": paddlex_count,
                    "blip_caption": blip_count,
                },
                "status": status,
                "paddlex_status": paddlex_status,
            },
            "captioning": {
                "enabled": bool(config.caption_enabled_flag),
                "configured_instances": (
                    status.get("blip_caption", {}).get("configured_instances")
                    if isinstance(status, dict)
                    else None
                ),
            },
        }

    return app


settings = get_settings()
app = create_app(settings)


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8088,
        reload=False,
        workers=1,
    )
