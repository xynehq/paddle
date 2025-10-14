import atexit
import gc
import io
import json
import logging
import multiprocessing
import os
import queue
import re
import threading
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
_IMAGE_QUEUE_MAXSIZE: Final[int] = 1

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
_status_server_lock = threading.Lock()
_status_server_started = False
_status_httpd: Optional[HTTPServer] = None


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _InstanceStatusRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            normalized_path = self.path.rstrip("/") or "/"
            normalized_endpoint = _STATUS_ENDPOINT_PATH.rstrip("/") or "/"
            if normalized_path != normalized_endpoint:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")
                return

            # Use a timeout on the lock to prevent hanging
            lock_acquired = _activity_lock.acquire(timeout=1.0)
            if not lock_acquired:
                # If we can't acquire the lock, return current best-effort values
                active = 0
                total = 0
            else:
                try:
                    active = _active_request_counter
                    total = _total_instance_counter
                finally:
                    _activity_lock.release()

            idle = max(total - active, 0) if total > 0 else None
            payload = {
                "active_instances": active,
                "configured_instances": total,
            }
            if idle is not None:
                payload["idle_instances"] = idle
            response = json.dumps(payload, separators=(",", ":")).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            self.wfile.flush()
        except Exception as e:
            _LOGGER.error(f"Error in status handler: {e}")
            try:
                self.send_error(500, f"Internal error: {e}")
            except:
                pass

    def log_message(self, format, *args):
        return


def _shutdown_status_server():
    global _status_httpd, _status_server_started
    if _status_httpd is not None:
        try:
            _status_httpd.shutdown()
            _status_httpd.server_close()
        except Exception:
            pass
        finally:
            _status_httpd = None
    with _status_server_lock:
        _status_server_started = False


def _start_status_server_if_needed():
    global _status_server_started

    with _status_server_lock:
        if _status_server_started:
            return
        _status_server_started = True

    def _serve():
        global _status_httpd
        try:
            httpd = _ThreadedHTTPServer(
                (_STATUS_SERVER_HOST, _STATUS_SERVER_PORT),
                _InstanceStatusRequestHandler,
            )
        except OSError as exc:
            _LOGGER.warning(
                "Instance status server failed to bind %s:%s (%s)",
                _STATUS_SERVER_HOST,
                _STATUS_SERVER_PORT,
                exc,
            )
            return

        _status_httpd = httpd
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
            _status_httpd = None

    threading.Thread(target=_serve, daemon=True).start()


@contextmanager
def _track_active_requests():
    global _active_request_counter
    with _activity_lock:
        _active_request_counter += 1
    try:
        yield
    finally:
        with _activity_lock:
            _active_request_counter = max(_active_request_counter - 1, 0)


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


atexit.register(_shutdown_status_server)

_initial_instance_env = os.environ.get("TRITON_INSTANCE_COUNT")
if _initial_instance_env:
    try:
        _update_total_instance_counter(int(_initial_instance_env))
    except ValueError:
        _LOGGER.warning(
            "Ignored invalid TRITON_INSTANCE_COUNT value: %s",
            _initial_instance_env,
        )

try:
    _start_status_server_if_needed()
except Exception as exc:  # pragma: no cover - defensive guard
    _LOGGER.exception("Failed to start instance status server: %s", exc)

_initial_config_total = _infer_instance_count_from_config_files(
    {"model_directory": os.path.dirname(__file__)}
)
if _initial_config_total > 0:
    _update_total_instance_counter(_initial_config_total, force=True)


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
        _update_total_instance_counter(
            configured_instances, force=force_total_update
        )
        _start_status_server_if_needed()

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

        producer = threading.Thread(
            target=self._image_conversion_worker,
            args=(file_bytes, file_type, max_num_imgs, output_queue),
            daemon=True,
        )
        producer.start()
        return output_queue, producer

    def _image_conversion_worker(
        self,
        file_bytes: bytes,
        file_type: str,
        max_num_imgs: Optional[int],
        output_queue: queue.Queue,
    ):
        try:
            if file_type == "PDF":
                iterator = self._iterate_pdf_pages(file_bytes, max_num_imgs)
            else:
                iterator = self._iterate_image_frames(file_bytes, max_num_imgs)
            for payload in iterator:
                output_queue.put(("data", payload))
        except Exception as exc:
            output_queue.put(("error", exc))
        finally:
            output_queue.put(("done", None))

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

            for page_index in range(allowed_pages):
                page = document.load_page(page_index)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(_PDF_RENDER_ZOOM, _PDF_RENDER_ZOOM),
                    alpha=False,
                )
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

        max_allowed = (
            None if max_num_imgs is None or max_num_imgs < 0 else max_num_imgs
        )
        with Image.open(io.BytesIO(file_bytes)) as image:
            for frame_index, frame in enumerate(ImageSequence.Iterator(image)):
                if max_allowed is not None and frame_index >= max_allowed:
                    break
                rgb_frame = frame.convert("RGB")
                np_img = np.asarray(rgb_frame)
                np_img = np.ascontiguousarray(np_img[:, :, ::-1])
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

        image_queue, producer = self._start_image_producer(
            file_bytes=file_bytes,
            file_type=file_type,
            max_num_imgs=max_num_imgs,
        )

        layout_parsing_results: List[Dict[str, Any]] = []
        pages_meta: List[Dict[str, int]] = []
        conversion_error: Optional[Exception] = None

        try:
            while True:
                message_type, payload = image_queue.get()
                if message_type == "done":
                    break
                if message_type == "error":
                    conversion_error = payload
                    continue
                page_index, img, page_info = payload
                pages_meta.append(page_info)

                pipeline_output = self.pipeline([img], **pipeline_kwargs)
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
                del pipeline_output

                pruned_res = app_common.prune_result(item.json["res"])
                md_data = item.markdown
                md_text = md_data["markdown_texts"]
                result_index = len(layout_parsing_results)
                md_imgs = app_common.postprocess_images(
                    md_data["markdown_images"],
                    log_id,
                    filename_template=f"markdown_{result_index}/{{key}}",
                    file_storage=self.context["file_storage"],
                    return_urls=self.context["return_img_urls"],
                    max_img_size=self.context["max_output_img_size"],
                )
                md_flags = md_data["page_continuation_flags"]
                if visualize_enabled:
                    imgs = {
                        "input_img": img,
                        **item.img,
                    }
                    imgs = app_common.postprocess_images(
                        imgs,
                        log_id,
                        filename_template=f"{{key}}_{result_index}.jpg",
                        file_storage=self.context["file_storage"],
                        return_urls=self.context["return_img_urls"],
                        max_img_size=self.context["max_output_img_size"],
                    )
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

                del item
                del img
                
                # Periodic memory cleanup after each page
                gc.collect()
                
                # Clear PaddlePaddle GPU cache after each page
                try:
                    paddle.device.cuda.empty_cache()
                except Exception:
                    pass  # Ignore if not using GPU or paddle not available
        finally:
            producer.join()

        if conversion_error is not None:
            return protocol.create_aistudio_output_without_result(
                422,
                f"Failed to convert the input file into images: {conversion_error}",
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
