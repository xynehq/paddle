import gc
import io
import json
import os
import queue
import threading
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import paddle
from PIL import Image, ImageSequence

from layout_captioning import CaptionCoordinator, load_caption_config
from layout_status import (
    InstanceStatusTracker,
    infer_configured_instance_count,
    infer_instance_count_from_config_files,
)

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

_INSTANCE_ID = os.environ.get("TRITON_SERVER_INSTANCE_ID", str(os.getpid()))
_STATUS_FILE = f"/tmp/triton_instance_status_{_INSTANCE_ID}.json"

_STATUS_TRACKER = InstanceStatusTracker(
    _STATUS_FILE,
    _INSTANCE_ID,
    _STATUS_WRITE_INTERVAL,
)

_trim_cache_counter = 0


# ---------------- Helper Functions -----------------

def _read_bool_env(name: str) -> Optional[bool]:
    """Return boolean value for an environment flag if it is well-formed."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return None


def _maybe_trim_gpu_cache():
    global _trim_cache_counter
    try:
        sync = getattr(paddle.device, "synchronize", None)
        if callable(sync):
            sync()
        else:
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
        print(f"[LAYOUT-PARSING] postprocess_images timed out after {timeout:.1f}s")
        raise TimeoutError("postprocess_images timed out") from exc
    if status == "error":
        raise payload
    return payload


# Initialize status tracker with default values
_STATUS_TRACKER.write_status(force=True)

_initial_instance_env = os.environ.get("TRITON_INSTANCE_COUNT")
if _initial_instance_env:
    try:
        _STATUS_TRACKER.update_total_instances(
            int(_initial_instance_env), force=True
        )
    except ValueError:
        print(
            f"[LAYOUT-PARSING] Ignored invalid TRITON_INSTANCE_COUNT value: {_initial_instance_env}"
        )

_initial_config_total = infer_instance_count_from_config_files(
    {"model_directory": os.path.dirname(__file__)}
)
if _initial_config_total > 0:
    _STATUS_TRACKER.update_total_instances(_initial_config_total, force=True)

_STATUS_TRACKER.start()
print(f"[LAYOUT-PARSING] Started model instance {_INSTANCE_ID} (PID={os.getpid()})")
class TritonPythonModel(BaseTritonPythonModel):
    def initialize(self, args):
        super().initialize(args)
        self.context = {}
        self.context["file_storage"] = None
        self.context["return_img_urls"] = False
        self.context["max_num_input_imgs"] = _DEFAULT_MAX_NUM_INPUT_IMGS
        self.context["max_output_img_size"] = _DEFAULT_MAX_OUTPUT_IMG_SIZE
        # Load caption config from colocated YAML (no env, no fallback)
        try:
            cfg = load_caption_config(os.path.dirname(__file__))
            self.context["captioning"] = cfg.get("captioning", {})
        except Exception:
            self.context["captioning"] = {"enabled": True}

        # Global override via environment variable IMAGE_CAPTIONING_ENABLED (default: true)
        # Accept truthy: 1,true,yes,on | falsy: 0,false,no,off
        try:
            override = _read_bool_env("IMAGE_CAPTIONING_ENABLED")
            if override is not None:
                cap_cfg = dict(self.context.get("captioning") or {})
                cap_cfg["enabled"] = override
                self.context["captioning"] = cap_cfg
        except Exception:
            pass
        try:
            cap_cfg = self.context.get("captioning") or {}
            print(
                "[LAYOUT-PARSING] Captioning cfg: "
                f"enabled={bool(cap_cfg.get('enabled'))} "
                f"provider={cap_cfg.get('provider')} "
                f"url={cap_cfg.get('triton_url')} "
                f"model={cap_cfg.get('triton_model')}"
            )
        except Exception:
            pass
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

        configured_instances = infer_configured_instance_count(
            getattr(self, "model_config", None)
        )
        if not configured_instances and isinstance(args, dict):
            model_config_json = args.get("model_config")
            if model_config_json:
                try:
                    maybe_configured = infer_configured_instance_count(
                        json.loads(model_config_json)
                    )
                    configured_instances = max(configured_instances, maybe_configured)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    # Intentionally silent: noisy debug message removed for production.
                    pass
        env_override = os.environ.get("TRITON_INSTANCE_COUNT")
        if env_override:
            try:
                configured_instances = max(configured_instances, int(env_override))
            except ValueError:
                pass
        force_total_update = False
        file_override = infer_instance_count_from_config_files(args)
        if file_override > 0:
            configured_instances = file_override
            force_total_update = True
        try:
            configured_instances = max(1, int(configured_instances or 0))
        except (TypeError, ValueError):
            configured_instances = 1
        _STATUS_TRACKER.update_total_instances(
            configured_instances, force=force_total_update
        )
        _STATUS_TRACKER.write_status(force=True)

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
                # Intentionally silent: noisy debug message removed for production.
                pass
        finally:
            while True:
                try:
                    output_queue.put(("done", None), timeout=1)
                    break
                except queue.Full:
                    if stop_evt.wait(0.1):
                        break

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
        with _STATUS_TRACKER.track_active():
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
        caption_cfg = self.context.get("captioning") or {}
        caption_coordinator = CaptionCoordinator(caption_cfg)

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

                    pruned_res = app_common.prune_result(item.json["res"])
                    md_data = item.markdown
                    md_text = md_data["markdown_texts"]
                    result_index = len(layout_parsing_results)
                    
                    # Trigger captioning before markdown images are mutated
                    if caption_coordinator.is_enabled:
                        original_md_images = dict(
                            md_data.get("markdown_images") or {}
                        )
                        caption_coordinator.start_page(
                            page_index,
                            pruned_res,
                            original_md_images,
                            log_id,
                        )
                    
                    # Process images (this modifies md_data["markdown_images"] in place, converting to URLs)
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

        print(
            f"[LAYOUT-PARSING] log_id={log_id} pages={len(pages_meta)} "
            f"file_type={file_type} visualize={bool(visualize_enabled)}"
        )
        caption_coordinator.finalize(layout_parsing_results)

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
