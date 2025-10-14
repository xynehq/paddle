import atexit
import errno
import gc
import io
import json
import logging
import multiprocessing
import os
import queue
import re
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import paddle
import socketserver
from PIL import Image, ImageSequence

from paddlex_hps_server import (
    BaseTritonPythonModel,
    app_common,
    protocol,
    schemas,
    utils,
)
from paddlex_hps_server.storage import SupportsGetURL, create_storage

_DEFAULT_MAX_NUM_INPUT_IMGS: Final[int] = 10
_DEFAULT_MAX_OUTPUT_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
_PDF_RENDER_ZOOM: Final[float] = 1.5
_MAX_RENDER_DIM: Final[int] = 2200
_IMAGE_QUEUE_MAXSIZE: Final[int] = 1
_MAX_INPUT_DIM: Final[int] = 6000
_HARD_MAX_NUM_INPUT_IMGS: Final[int] = 200
_POSTPROCESS_UPLOAD_TIMEOUT: Final[float] = float(
    os.environ.get("TRITON_POSTPROCESS_TIMEOUT_SECONDS", "30.0")
)
_STATUS_WRITE_INTERVAL: Final[float] = 0.5
_TRIM_CACHE_FREQUENCY: Final[int] = max(
    0, int(os.environ.get("PADDLE_TRIM_CACHE_EVERY_N", "0"))
)

try:
    paddle.disable_static()
except Exception:
    pass

_STATUS_SERVER_HOST: Final[str] = os.environ.get(
    "TRITON_INSTANCE_STATUS_HOST", "0.0.0.0"
)
_STATUS_SERVER_PORT: Final[int] = int(
    os.environ.get("TRITON_INSTANCE_STATUS_PORT", "8081")
)
_STATUS_ENDPOINT_PATH: Final[str] = os.environ.get(
    "TRITON_INSTANCE_STATUS_PATH", "/instance_status"
)

_LOGGER = logging.getLogger(__name__)

# Use threading instead of multiprocessing for better compatibility
_activity_lock = threading.Lock()
_active_request_counter = 0
_total_instance_counter = 0

# Use per-instance status files to avoid race conditions
_INSTANCE_ID = os.environ.get("TRITON_SERVER_INSTANCE_ID", str(os.getpid()))
_STATUS_FILE = f"/tmp/triton_instance_status_{_INSTANCE_ID}.json"
_last_status_write = 0.0
_status_heartbeat_stop = threading.Event()
_status_heartbeat_thread: Optional[threading.Thread] = None
_trim_cache_counter = 0


def _safe_unlink(path: str):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            _LOGGER.debug(f"unlink failed: {exc}")


def _write_status_to_file(force: bool = False):
    """Write current status to file for the standalone server to read."""
    global _last_status_write
    try:
        now = time.time()
        if not force and (now - _last_status_write) < _STATUS_WRITE_INTERVAL:
            return

        with _activity_lock:
            active = _active_request_counter
            total = _total_instance_counter

        status_data = {
            "active_instances": max(0, active),
            "configured_instances": max(0, total),
            "idle_instances": max(0, total - active),
            "last_updated": int(now),
            "instance_id": _INSTANCE_ID,
        }
        tmp_path = _STATUS_FILE + ".tmp"
        with open(tmp_path, "w") as file_obj:
            json.dump(status_data, file_obj, separators=(",", ":"))
        os.replace(tmp_path, _STATUS_FILE)
        _last_status_write = now
    except Exception as exc:
        _LOGGER.debug(f"Failed to write status file: {exc}")


def _status_heartbeat_loop():
    while not _status_heartbeat_stop.wait(_STATUS_WRITE_INTERVAL):
        _write_status_to_file()


def _start_status_heartbeat():
    global _status_heartbeat_thread
    if _status_heartbeat_thread is None:
        _status_heartbeat_thread = threading.Thread(
            target=_status_heartbeat_loop,
            name="status-heartbeat",
            daemon=True,
        )
        _status_heartbeat_thread.start()


@atexit.register
def _cleanup_status_file():
    _status_heartbeat_stop.set()
    if _status_heartbeat_thread and _status_heartbeat_thread.is_alive():
        _status_heartbeat_thread.join(timeout=1.0)
    _safe_unlink(_STATUS_FILE)
    _safe_unlink(_STATUS_FILE + ".tmp")


def _maybe_trim_gpu_cache():
    global _trim_cache_counter
    try:
        paddle.device.cuda.synchronize()
    except Exception:
        return
    gc.collect()
    if _TRIM_CACHE_FREQUENCY > 0:
        _trim_cache_counter = (_trim_cache_counter + 1) % _TRIM_CACHE_FREQUENCY
        if _trim_cache_counter == 0:
            try:
                paddle.device.cuda.empty_cache()
            except Exception:
                pass


def _postprocess_images_with_timeout(timeout: float, *args, **kwargs):
    if timeout is None or timeout <= 0:
        return app_common.postprocess_images(*args, **kwargs)

    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def _worker():
        try:
            result = app_common.postprocess_images(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - threads hard to cover
            result_queue.put(("error", exc))
        else:
            result_queue.put(("ok", result))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    try:
        status, payload = result_queue.get(timeout=timeout)
    except queue.Empty as exc:
        _LOGGER.error("Timed out waiting for postprocess_images after %.1fs", timeout)
        raise TimeoutError("postprocess_images timed out") from exc
    if status == "error":
        raise payload
    return payload


@contextmanager
def _track_active_requests():
    global _active_request_counter
    with _activity_lock:
        _active_request_counter += 1
    _write_status_to_file()
    try:
        yield
    finally:
        with _activity_lock:
            _active_request_counter = max(_active_request_counter - 1, 0)
        _write_status_to_file()


def _infer_configured_instance_count(model_config: Any) -> int:
    if model_config is None:
        return 0

    def _extract_groups(config_obj: Any) -> List[Any]:
        if config_obj is None:
            return []
        if isinstance(config_obj, dict):
            if "instance_group" in config_obj:
                return config_obj["instance_group"] or []
            if "config" in config_obj:
                return _extract_groups(config_obj["config"])
            return []
        groups = getattr(config_obj, "instance_group", None)
        if groups is not None:
            return list(groups)
        nested = getattr(config_obj, "config", None)
        if nested is not None:
            return _extract_groups(nested)
        return []

    groups = _extract_groups(model_config)
    total = 0
    for group in groups:
        if isinstance(group, dict):
            candidate = group.get("count", 0)
        else:
            candidate = getattr(group, "count", 0)
        try:
            total += int(candidate)
        except (TypeError, ValueError):
            continue
    return max(total, 0)


def _candidate_config_paths(args: Any) -> List[str]:
    candidates: List[str] = []

    def _add(path: Optional[str]):
        if path and path not in candidates:
            candidates.append(path)

    directories: List[str] = []

    def _add_directory(path: Optional[str]):
        if path and path not in directories:
            directories.append(path)

    if isinstance(args, dict):
        model_dir = args.get("model_directory")
        if model_dir:
            _add_directory(model_dir)
        repo_dir = args.get("model_repository")
        if repo_dir:
            _add_directory(repo_dir)
            model_version = args.get("model_version")
            if model_version:
                _add_directory(os.path.join(repo_dir, model_version))

    module_dir = os.path.dirname(__file__)
    _add_directory(module_dir)
    _add_directory(os.path.dirname(module_dir))

    for directory in directories:
        _add(os.path.join(directory, "config_gpu.pbtxt"))
        _add(os.path.join(directory, "config.pbtxt"))
        _add(os.path.join(directory, "config_cpu.pbtxt"))

    env_config_path = os.environ.get("TRITON_MODEL_CONFIG_PATH")
    _add(env_config_path)
    return candidates


def _infer_instance_count_from_config_files(args: Any) -> int:
    for path in _candidate_config_paths(args):
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                contents = config_file.read()
        except FileNotFoundError:
            continue
        except OSError as exc:
            _LOGGER.debug("Failed to read config file %s: %s", path, exc)
            continue
        matches = re.findall(r"\bcount\s*:\s*(\d+)", contents)
        candidate_total = sum(int(match) for match in matches)
        if candidate_total > 0:
            return candidate_total
    return 0


def _update_total_instance_counter(candidate_total: int, *, force: bool = False):
    global _total_instance_counter
    if candidate_total <= 0:
        return
    with _activity_lock:
        if force or _total_instance_counter <= 0:
            _total_instance_counter = candidate_total
            return
        # Keep the highest observed value to handle multiple worker initializations
        # racing to report their configured instance counts.
        _total_instance_counter = max(_total_instance_counter, candidate_total)


# Initialize status file with default values
_write_status_to_file(force=True)

_initial_instance_env = os.environ.get("TRITON_INSTANCE_COUNT")
if _initial_instance_env:
    try:
        _update_total_instance_counter(int(_initial_instance_env))
        _write_status_to_file(force=True)
    except ValueError:
        _LOGGER.warning(
            "Ignored invalid TRITON_INSTANCE_COUNT value: %s",
            _initial_instance_env,
        )

_initial_config_total = _infer_instance_count_from_config_files(
    {"model_directory": os.path.dirname(__file__)}
)
if _initial_config_total > 0:
    _update_total_instance_counter(_initial_config_total, force=True)
    _write_status_to_file(force=True)

_start_status_heartbeat()
_LOGGER.info("Started model instance %s (PID=%s)", _INSTANCE_ID, os.getpid())


class TritonPythonModel(BaseTritonPythonModel):
    def initialize(self, args):
        super().initialize(args)
        self.context = {}
        self.context["file_storage"] = None
        self.context["return_img_urls"] = False
        self.context["max_num_input_imgs"] = _DEFAULT_MAX_NUM_INPUT_IMGS
        self.context["max_output_img_size"] = _DEFAULT_MAX_OUTPUT_IMG_SIZE
        if self.app_config.extra:
            if "file_storage" in self.app_config.extra:
                self.context["file_storage"] = create_storage(
                    self.app_config.extra["file_storage"]
                )
            if "return_img_urls" in self.app_config.extra:
                self.context["return_img_urls"] = self.app_config.extra[
                    "return_img_urls"
                ]
            if "max_num_input_imgs" in self.app_config.extra:
                self.context["max_num_input_imgs"] = self.app_config.extra[
                    "max_num_input_imgs"
                ]
            if "max_output_img_size" in self.app_config.extra:
                self.context["max_output_img_size"] = self.app_config.extra[
                    "max_output_img_size"
                ]
        if self.context["return_img_urls"]:
            file_storage = self.context["file_storage"]
            if not file_storage:
                raise ValueError(
                    "The file storage must be properly configured when URLs need to be returned."
                )
            if not isinstance(file_storage, SupportsGetURL):
                raise TypeError(f"{type(file_storage)} does not support getting URLs.")

        configured_instances = _infer_configured_instance_count(
            getattr(self, "model_config", None)
        )
        if not configured_instances and isinstance(args, dict):
            model_config_json = args.get("model_config")
            if model_config_json:
                try:
                    maybe_configured = _infer_configured_instance_count(
                        json.loads(model_config_json)
                    )
                    configured_instances = max(configured_instances, maybe_configured)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    _LOGGER.debug(
                        "Failed to parse model_config for instance count: %s",
                        exc,
                    )
        env_override = os.environ.get("TRITON_INSTANCE_COUNT")
        if env_override:
            try:
                configured_instances = max(configured_instances, int(env_override))
            except ValueError:
                pass
        force_total_update = False
        file_override = _infer_instance_count_from_config_files(args)
        if file_override > 0:
            configured_instances = file_override
            force_total_update = True
        try:
            configured_instances = max(1, int(configured_instances or 0))
        except (TypeError, ValueError):
            configured_instances = 1
        _update_total_instance_counter(
            configured_instances, force=force_total_update
        )
        _write_status_to_file(force=True)

    def get_input_model_type(self):
        return schemas.pp_structurev3.InferRequest

    def get_result_model_type(self):
        return schemas.pp_structurev3.InferResult

    def _start_image_producer(
        self,
        file_bytes: bytes,
        file_type: str,
        max_num_imgs: Optional[int],
    ):
        output_queue: queue.Queue = queue.Queue(maxsize=_IMAGE_QUEUE_MAXSIZE)
        stop_evt = threading.Event()
        producer = threading.Thread(
            target=self._image_conversion_worker,
            args=(file_bytes, file_type, max_num_imgs, output_queue, stop_evt),
            daemon=True,
        )
        producer.start()
        return output_queue, producer, stop_evt

    def _image_conversion_worker(
        self,
        file_bytes: bytes,
        file_type: str,
        max_num_imgs: Optional[int],
        output_queue: queue.Queue,
        stop_evt: threading.Event,
    ):
        try:
            if file_type == "PDF":
                iterator = self._iterate_pdf_pages(file_bytes, max_num_imgs)
            else:
                iterator = self._iterate_image_frames(file_bytes, max_num_imgs)
            for payload in iterator:
                if stop_evt.is_set():
                    break
                while not stop_evt.is_set():
                    try:
                        output_queue.put(("data", payload), timeout=5)
                        break
                    except queue.Full:
                        if stop_evt.wait(0.1):
                            break
        except Exception as exc:
            try:
                output_queue.put(("error", exc), timeout=1)
            except queue.Full:
                _LOGGER.debug("Dropping conversion error due to full queue: %s", exc)
        finally:
            while True:
                try:
                    output_queue.put(("done", None), timeout=1)
                    break
                except queue.Full:
                    if stop_evt.wait(0.1):
                        break

    def _run_page_with_fallback(
        self,
        img: np.ndarray,
        pipeline_kwargs: Dict[str, Any],
        log_id: str,
        page_index: int,
    ):
        try:
            return self.pipeline([img], **pipeline_kwargs)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            _LOGGER.warning(
                "[%s] OOM on page %s, retrying with half-size input",
                log_id,
                page_index,
            )
            height, width = img.shape[:2]
            if height <= 1 or width <= 1:
                raise
            new_size = (max(1, width // 2), max(1, height // 2))
            try:
                resized = Image.fromarray(img[:, :, ::-1]).resize(
                    new_size, Image.BILINEAR
                )
                img_small = np.ascontiguousarray(
                    np.asarray(resized, dtype=np.uint8)[:, :, ::-1]
                )
                del resized
            except Exception as resize_exc:
                _LOGGER.warning(
                    "[%s] Failed to downscale page %s after OOM: %s",
                    log_id,
                    page_index,
                    resize_exc,
                )
                raise
            try:
                try:
                    gc.collect()
                    paddle.device.cuda.empty_cache()
                except Exception:
                    pass
                return self.pipeline([img_small], **pipeline_kwargs)
            finally:
                del img_small

    def _iterate_pdf_pages(
        self, file_bytes: bytes, max_num_imgs: Optional[int]
    ):
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF (fitz) is required to process PDF inputs."
            ) from exc

        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            total_pages = document.page_count
            if max_num_imgs is None or max_num_imgs < 0:
                allowed_pages = total_pages
            else:
                allowed_pages = min(total_pages, max_num_imgs)
            allowed_pages = min(allowed_pages, _HARD_MAX_NUM_INPUT_IMGS)

            for page_index in range(allowed_pages):
                page = document.load_page(page_index)
                width, height = page.rect.width, page.rect.height
                max_dim = max(width, height) or 1.0
                scale = min(_PDF_RENDER_ZOOM, _MAX_RENDER_DIM / max_dim)
                if scale <= 0:
                    scale = _PDF_RENDER_ZOOM
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 1:
                    image = np.repeat(image, 3, axis=2)
                elif pix.n == 4:
                    image = image[:, :, :3]
                image = np.ascontiguousarray(image[:, :, ::-1])
                yield page_index, image, {"width": pix.width, "height": pix.height}
                del image
                del pix

    def _iterate_image_frames(
        self, file_bytes: bytes, max_num_imgs: Optional[int]
    ):
        if max_num_imgs == 0:
            return

        if max_num_imgs is None or max_num_imgs < 0:
            effective_limit = _HARD_MAX_NUM_INPUT_IMGS
        else:
            effective_limit = min(max_num_imgs, _HARD_MAX_NUM_INPUT_IMGS)
        with Image.open(io.BytesIO(file_bytes)) as image:
            for frame_index, frame in enumerate(ImageSequence.Iterator(image)):
                if effective_limit is not None and frame_index >= effective_limit:
                    break
                rgb_frame = frame.convert("RGB")
                width, height = rgb_frame.size
                largest_dim = max(width, height)
                if largest_dim > _MAX_INPUT_DIM:
                    scale = _MAX_INPUT_DIM / float(largest_dim)
                    new_size = (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    )
                    rgb_frame = rgb_frame.resize(new_size, Image.BILINEAR)
                    width, height = rgb_frame.size
                np_img = np.ascontiguousarray(np.asarray(rgb_frame)[:, :, ::-1])
                height, width = np_img.shape[:2]
                yield frame_index, np_img, {"width": width, "height": height}
                del np_img
                del rgb_frame

    def run(self, input, log_id):
        with _track_active_requests():
            return self._run_impl(input, log_id)

    def _run_impl(self, input, log_id):
        if input.fileType is None:
            if utils.is_url(input.file):
                maybe_file_type = utils.infer_file_type(input.file)
                if maybe_file_type is None or not (
                    maybe_file_type == "PDF" or maybe_file_type == "IMAGE"
                ):
                    return protocol.create_aistudio_output_without_result(
                        422,
                        "Unsupported file type",
                        log_id=log_id,
                    )
                file_type = maybe_file_type
            else:
                return protocol.create_aistudio_output_without_result(
                    422,
                    "File type cannot be determined",
                    log_id=log_id,
                )
        else:
            file_type = "PDF" if input.fileType == 0 else "IMAGE"
        visualize_enabled = input.visualize if input.visualize is not None else self.app_config.visualize

        file_bytes = utils.get_raw_bytes(input.file)
        max_num_imgs = self.context["max_num_input_imgs"]

        pipeline_kwargs = dict(
            use_doc_orientation_classify=input.useDocOrientationClassify,
            use_doc_unwarping=input.useDocUnwarping,
            use_textline_orientation=input.useTextlineOrientation,
            use_seal_recognition=input.useSealRecognition,
            use_table_recognition=input.useTableRecognition,
            use_formula_recognition=input.useFormulaRecognition,
            use_chart_recognition=input.useChartRecognition,
            use_region_detection=input.useRegionDetection,
            layout_threshold=input.layoutThreshold,
            layout_nms=input.layoutNms,
            layout_unclip_ratio=input.layoutUnclipRatio,
            layout_merge_bboxes_mode=input.layoutMergeBboxesMode,
            text_det_limit_side_len=input.textDetLimitSideLen,
            text_det_limit_type=input.textDetLimitType,
            text_det_thresh=input.textDetThresh,
            text_det_box_thresh=input.textDetBoxThresh,
            text_det_unclip_ratio=input.textDetUnclipRatio,
            text_rec_score_thresh=input.textRecScoreThresh,
            seal_det_limit_side_len=input.sealDetLimitSideLen,
            seal_det_limit_type=input.sealDetLimitType,
            seal_det_thresh=input.sealDetThresh,
            seal_det_box_thresh=input.sealDetBoxThresh,
            seal_det_unclip_ratio=input.sealDetUnclipRatio,
            seal_rec_score_thresh=input.sealRecScoreThresh,
            use_wired_table_cells_trans_to_html=input.useWiredTableCellsTransToHtml,
            use_wireless_table_cells_trans_to_html=input.useWirelessTableCellsTransToHtml,
            use_table_orientation_classify=input.useTableOrientationClassify,
            use_ocr_results_with_table_cells=input.useOcrResultsWithTableCells,
            use_e2e_wired_table_rec_model=input.useE2eWiredTableRecModel,
            use_e2e_wireless_table_rec_model=input.useE2eWirelessTableRecModel,
        )

        image_queue, producer, stop_evt = self._start_image_producer(
            file_bytes=file_bytes,
            file_type=file_type,
            max_num_imgs=max_num_imgs,
        )

        layout_parsing_results: List[Dict[str, Any]] = []
        pages_meta: List[Dict[str, int]] = []
        conversion_error: Optional[Exception] = None

        try:
            while True:
                try:
                    message_type, payload = image_queue.get(timeout=10)
                except queue.Empty:
                    conversion_error = RuntimeError(
                        "Image producer stalled or died"
                    )
                    stop_evt.set()
                    break
                if message_type == "done":
                    break
                if message_type == "error":
                    if conversion_error is None:
                        conversion_error = payload
                    continue
                if message_type != "data":
                    continue
                page_index, img, page_info = payload
                pages_meta.append(page_info)

                item = None
                pipeline_output = None
                try:
                    pipeline_output = self._run_page_with_fallback(
                        img, pipeline_kwargs, log_id, page_index
                    )
                    pipeline_iter = iter(pipeline_output)
                    try:
                        item = next(pipeline_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"Pipeline produced no output for page index {page_index}"
                        ) from exc
                    finally:
                        for _ in pipeline_iter:
                            pass
                        del pipeline_iter

                    pruned_res = app_common.prune_result(item.json["res"])
                    md_data = item.markdown
                    md_text = md_data["markdown_texts"]
                    result_index = len(layout_parsing_results)
                    try:
                        md_imgs = _postprocess_images_with_timeout(
                            _POSTPROCESS_UPLOAD_TIMEOUT,
                            md_data["markdown_images"],
                            log_id,
                            filename_template=f"markdown_{result_index}/{{key}}",
                            file_storage=self.context["file_storage"],
                            return_urls=self.context["return_img_urls"],
                            max_img_size=self.context["max_output_img_size"],
                        )
                    except TimeoutError as exc:
                        conversion_error = exc
                        stop_evt.set()
                        break
                    md_flags = md_data["page_continuation_flags"]
                    if visualize_enabled:
                        imgs = {
                            "input_img": img,
                            **item.img,
                        }
                        try:
                            imgs = _postprocess_images_with_timeout(
                                _POSTPROCESS_UPLOAD_TIMEOUT,
                                imgs,
                                log_id,
                                filename_template=f"{{key}}_{result_index}.jpg",
                                file_storage=self.context["file_storage"],
                                return_urls=self.context["return_img_urls"],
                                max_img_size=self.context["max_output_img_size"],
                            )
                        except TimeoutError as exc:
                            conversion_error = exc
                            stop_evt.set()
                            break
                    else:
                        imgs = {}
                    layout_parsing_results.append(
                        dict(
                            prunedResult=pruned_res,
                            markdown=dict(
                                text=md_text,
                                images=md_imgs,
                                isStart=md_flags[0],
                                isEnd=md_flags[1],
                            ),
                            outputImages=(
                                {k: v for k, v in imgs.items() if k != "input_img"}
                                if imgs
                                else None
                            ),
                            inputImage=imgs.get("input_img"),
                        )
                    )
                finally:
                    if pipeline_output is not None:
                        del pipeline_output
                    if item is not None:
                        del item
                    del img
                    _maybe_trim_gpu_cache()
        finally:
            stop_evt.set()
            producer.join(timeout=5)
            while True:
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    break

        if conversion_error is not None:
            message = (
                f"Failed to convert the input file into images: {conversion_error}"
            )
            if isinstance(conversion_error, TimeoutError):
                message = (
                    "Timed out while uploading processed images. "
                    "Please retry later."
                )
            return protocol.create_aistudio_output_without_result(
                422,
                message,
                log_id=log_id,
            )

        if not layout_parsing_results:
            return protocol.create_aistudio_output_without_result(
                422,
                "No renderable pages found in the provided file.",
                log_id=log_id,
            )

        data_info = {
            "numPages": len(pages_meta),
            "pages": pages_meta,
            "type": file_type.lower(),
        }

        _LOGGER.info(
            "[%s] pages=%s file_type=%s visualize=%s",
            log_id,
            len(pages_meta),
            file_type,
            bool(visualize_enabled),
        )

        result_output = schemas.pp_structurev3.InferResult(
            layoutParsingResults=layout_parsing_results,
            dataInfo=data_info,
        )

        # Explicit memory cleanup to prevent GPU memory persistence
        del file_bytes, pages_meta, layout_parsing_results, pipeline_kwargs
        gc.collect()

        # Clear PaddlePaddle GPU cache
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass  # Ignore if not using GPU or paddle not available

        return result_output
