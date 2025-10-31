import base64
import io
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

import numpy as np
from PIL import Image



def ensure_base64(img_data: Any) -> Optional[str]:
    """Convert various image formats to a base64 JPEG string."""
    if isinstance(img_data, str):
        return img_data

    try:
        if isinstance(img_data, (bytes, bytearray)):
            return base64.b64encode(img_data).decode("ascii")

        if isinstance(img_data, Image.Image):
            buf = io.BytesIO()
            img_data.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")

        if isinstance(img_data, np.ndarray):
            arr = img_data
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1]
            pil_img = Image.fromarray(arr.astype("uint8"))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None

    return None


def _decode_infer_output(output_array) -> Optional[bytes]:
    """Decode Triton inference output to raw bytes."""
    try:
        first = output_array.reshape(-1)[0]
        if hasattr(first, "tobytes"):
            return first.tobytes()
        if isinstance(first, (bytes, bytearray)):
            return bytes(first)
        return bytes(first)
    except Exception:
        return None


class TritonCaptionClient:
    def __init__(self, url: str, model_name: str, timeout_ms: int = 5000):
        self._url = url
        self._model = model_name
        try:
            _ms = int(timeout_ms)
        except Exception:
            _ms = 0
        self._timeout_ms = _ms if _ms > 0 else None
        self._timeout_s = (_ms / 1000.0) if _ms > 0 else None
        self._mode = "grpc" if url.startswith("grpc://") else "http"
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return

        if self._mode == "grpc":
            try:
                import tritonclient.grpc as grpcclient  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "tritonclient.grpc not available; install tritonclient"
                ) from exc
            url = (
                self._url.split("grpc://", 1)[1]
                if self._url.startswith("grpc://")
                else self._url
            )
            self._client = grpcclient.InferenceServerClient(url=url)
            self._grpcclient = grpcclient
        else:
            try:
                import tritonclient.http as httpclient  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "tritonclient.http not available; install tritonclient"
                ) from exc
            url = (
                self._url.split("http://", 1)[1]
                if self._url.startswith("http://")
                else self._url
            )
            url = "http://" + url if not self._url.startswith("http://") else self._url
            self._client = httpclient.InferenceServerClient(url=url)
            self._httpclient = httpclient

    def caption(
        self,
        image_b64: str,
        *,
        max_length: int,
        num_beams: int,
        no_repeat_ngram_size: int,
    ) -> Optional[str]:
        try:
            self._ensure_client()
        except Exception as exc:
            print(f"[LAYOUT-CAPTION] BLIP client init failed: {exc}")
            return None

        payload = json.dumps(
            dict(
                image_b64=image_b64,
                max_length=int(max_length),
                num_beams=int(num_beams),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
            )
        ).encode("utf-8")
        obj = np.asarray([[payload]], dtype=object)

        try:
            if self._mode == "grpc":
                infer_input = self._grpcclient.InferInput("input", [1, 1], "BYTES")
                infer_input.set_data_from_numpy(obj)
                output = self._grpcclient.InferRequestedOutput("output")
                resp = self._client.infer(
                    self._model,
                    [infer_input],
                    model_version="",
                    outputs=[output],
                    timeout=self._timeout_ms,
                )
                out = resp.as_numpy("output")
            else:
                infer_input = self._httpclient.InferInput("input", [1, 1], "BYTES")
                infer_input.set_data_from_numpy(obj)
                output = self._httpclient.InferRequestedOutput("output")
                resp = self._client.infer(
                    self._model,
                    inputs=[infer_input],
                    outputs=[output],
                    model_version="",
                    timeout=self._timeout_s,
                )
                out = resp.as_numpy("output")
        except Exception as call_exc:
            print(f"[LAYOUT-CAPTION] Caption request to {self._url} failed: {call_exc}")
            return None

        raw = _decode_infer_output(out)
        if not raw:
            return None

        try:
            res = json.loads(raw.decode("utf-8"))
            if isinstance(res, dict) and res.get("caption") and not res.get("error"):
                cap_text = str(res.get("caption"))
                return cap_text[:512]
        except Exception:
            return None
        return None


def load_caption_config(module_dir: str) -> Dict[str, Any]:
    """Load caption configuration YAML from the model directory."""
    cfg: Dict[str, Any] = {"captioning": {"enabled": True}}
    cfg_path = os.path.join(module_dir, "caption_config.yaml")
    if not os.path.isfile(cfg_path):
        return cfg
    try:
        if yaml is None:
            return cfg
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            cfg.update(data)
    except Exception:
        pass
    return cfg


def _parse_bbox_from_img_key(img_key: str) -> Optional[Tuple[int, int, int, int]]:
    m = re.search(r"img_in_image_box_(\d+)_(\d+)_(\d+)_(\d+)", str(img_key))
    if not m:
        return None
    try:
        return tuple(int(g) for g in m.groups())  # type: ignore
    except Exception:
        return None


def _should_caption_image(bbox: Tuple[int, int, int, int], config: Dict[str, Any]) -> bool:
    """Check if image meets minimum size thresholds for captioning."""
    try:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height

        min_width = int(config.get("min_image_width", 100))
        min_height = int(config.get("min_image_height", 100))
        min_area = int(config.get("min_image_area", 10000))

        if width < min_width or height < min_height:
            return False
        if area < min_area:
            return False
        return True
    except Exception:
        return True


def _collect_caption_requests(
    pruned_res: Dict[str, Any],
    md_images: Dict[str, str],
    caption_cfg: Dict[str, Any],
) -> List[Tuple[Tuple[int, int, int, int], str, Dict[str, Any]]]:
    """Collect caption requests for images that pass filtering."""
    requests = []

    parsing_list = pruned_res.get("parsing_res_list") or []
    if not isinstance(parsing_list, list) or not parsing_list:
        return requests

    by_bbox: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for blk in parsing_list:
        if not isinstance(blk, dict):
            continue
        if blk.get("block_label") != "image":
            continue
        bb = blk.get("block_bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            key = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        except Exception:
            continue
        by_bbox[key] = blk

    if not by_bbox:
        return requests

    for key, img_b64 in (md_images or {}).items():
        if "img_in_image_box" not in str(key):
            continue
        bbox = _parse_bbox_from_img_key(str(key))
        if not bbox:
            continue

        if not _should_caption_image(bbox, caption_cfg):
            continue

        blk = by_bbox.get(bbox)
        if blk is None or not img_b64:
            continue

        if not isinstance(img_b64, str):
            img_b64 = ensure_base64(img_b64)
            if not img_b64:
                continue

        requests.append((bbox, img_b64, blk))

    return requests


def _send_caption_requests_async(
    page_idx: int,
    pruned_res: Dict[str, Any],
    md_images: Dict[str, str],
    caption_cfg: Dict[str, Any],
    client: TritonCaptionClient,
    params: Dict[str, int],
) -> List[threading.Thread]:
    threads: List[threading.Thread] = []
    requests = _collect_caption_requests(pruned_res, md_images, caption_cfg)

    if not requests:
        return threads

    def _caption_worker(
        bbox: Tuple[int, int, int, int],
        img_b64: str,
        blk: Dict[str, Any],
    ):
        try:
            cap = client.caption(
                img_b64,
                max_length=params["max_length"],
                num_beams=params["num_beams"],
                no_repeat_ngram_size=params["no_repeat_ngram_size"],
            )
            if cap:
                blk["_pending_caption"] = cap
        except Exception as exc:
            print(f"[LAYOUT-CAPTION] Caption failed for bbox={bbox}: {exc}")

    for bbox, img_b64, blk in requests:
        thread = threading.Thread(
            target=_caption_worker,
            args=(bbox, img_b64, blk),
            daemon=True,
            name=f"caption-page{page_idx}-{bbox[0]},{bbox[1]}",
        )
        thread.start()
        threads.append(thread)

    return threads


def merge_pending_captions(layout_parsing_results: List[Dict[str, Any]]) -> None:
    """Merge pending captions into block content after all threads complete."""
    caption_count = 0
    for page_result in layout_parsing_results:
        pruned_res = page_result.get("prunedResult")
        if not pruned_res:
            continue
        parsing_list = pruned_res.get("parsing_res_list") or []
        for blk in parsing_list:
            if not isinstance(blk, dict):
                continue
            pending_cap = blk.pop("_pending_caption", None)
            if pending_cap:
                existing = blk.get("block_content")
                if existing:
                    blk["block_content"] = f"{existing} {pending_cap}".strip()
                else:
                    blk["block_content"] = pending_cap
                caption_count += 1

    if caption_count > 0:
        print(f"[LAYOUT-CAPTION] Merged {caption_count} captions into results")


class CaptionCoordinator:
    """Manage captioning requests across pages and merge the results."""

    def __init__(self, caption_cfg: Dict[str, Any]):
        self._cfg = caption_cfg or {}
        self._client: Optional[TritonCaptionClient] = None
        self._params: Optional[Dict[str, int]] = None
        self._threads: List[threading.Thread] = []
        self._initialize_client()

    def _initialize_client(self) -> None:
        if not (self._cfg.get("enabled") and self._cfg.get("provider") == "triton"):
            return
        try:
            url = self._cfg.get("triton_url") or "grpc://blip-server:8004"
            model = self._cfg.get("triton_model") or "blip-caption"
            timeout_ms = int(self._cfg.get("timeout_ms", 10000) or 10000)
            self._client = TritonCaptionClient(url, model, timeout_ms)
            self._params = dict(
                max_length=int(self._cfg.get("max_length", 50) or 50),
                num_beams=int(self._cfg.get("num_beams", 3) or 3),
                no_repeat_ngram_size=int(
                    self._cfg.get("no_repeat_ngram_size", 2) or 2
                ),
            )
            print(f"[LAYOUT-CAPTION] Async captioning enabled: {url}")
        except Exception as exc:
            print(f"[LAYOUT-CAPTION] Failed to init caption client: {exc}")
            self._client = None
            self._params = None

    @property
    def is_enabled(self) -> bool:
        return self._client is not None and self._params is not None

    def start_page(
        self,
        page_index: int,
        pruned_res: Dict[str, Any],
        md_images: Dict[str, Any],
        log_id: Optional[str] = None,
    ) -> None:
        if not self.is_enabled:
            return
        assert self._client is not None and self._params is not None
        try:
            threads = _send_caption_requests_async(
                page_index,
                pruned_res,
                md_images,
                self._cfg,
                self._client,
                self._params,
            )
            self._threads.extend(threads)
        except Exception as exc:
            prefix = f"[{log_id}] " if log_id else ""
            print(
                f"[LAYOUT-CAPTION] {prefix}Failed to start caption threads for page {page_index}: {exc}"
            )

    def finalize(self, layout_parsing_results: List[Dict[str, Any]]) -> None:
        if not self._threads:
            return

        start_wait = time.time()
        for thread in self._threads:
            thread.join(timeout=30.0)
        wait_time = time.time() - start_wait
        if self._threads:
            print(
                f"[LAYOUT-CAPTION] Caption threads completed in {wait_time:.2f}s ({len(self._threads)} thread(s))"
            )

        merge_pending_captions(layout_parsing_results)
        self._threads.clear()
