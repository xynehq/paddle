import json
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import triton_python_backend_utils as pb_utils

from status_tracker import (
    InstanceStatusTracker,
    infer_configured_instance_count,
    infer_instance_count_from_config_files,
)


# Defaults for BLIP config (no env required)
DEFAULT_BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"
# Set to None for auto (cuda if available else cpu)
DEFAULT_BLIP_DEVICE = None
# Local cache directory for model weights
BLIP_MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "blip_model_cache")

_STATUS_WRITE_INTERVAL = 0.5
_INSTANCE_ID = os.environ.get("TRITON_SERVER_INSTANCE_ID", str(os.getpid()))
_STATUS_FILE = f"/tmp/blip_instance_status_{_INSTANCE_ID}.json"
_STATUS_TRACKER = InstanceStatusTracker(
    _STATUS_FILE,
    _INSTANCE_ID,
    _STATUS_WRITE_INTERVAL,
)
_STATUS_TRACKER.write_status(force=True)
_initial_instance_env = os.environ.get("TRITON_INSTANCE_COUNT")
if _initial_instance_env:
    try:
        _STATUS_TRACKER.update_total_instances(int(_initial_instance_env), force=True)
    except ValueError:
        pass

_initial_config_total = infer_instance_count_from_config_files(
    {"model_directory": os.path.dirname(__file__)}
)
if _initial_config_total > 0:
    _STATUS_TRACKER.update_total_instances(_initial_config_total, force=True)

_STATUS_TRACKER.start()


class TritonPythonModel:
    """Triton Python backend for BLIP image captioning.

    Input:  name "input", TYPE_STRING, dims [1]
            Each element is a JSON string: {
              "image_b64": "...",  # base64 image, optional data: prefix
              "max_length": 50,
              "num_beams": 3,
              "no_repeat_ngram_size": 2
            }
    Output: name "output", TYPE_STRING, dims [1]
            Each element is a JSON string: {"caption": str|null, "error": str|null, "params": {...}}
    """

    def _is_model_cached(self, cache_path: Path) -> bool:
        """Check if model is already downloaded in cache directory."""
        if not cache_path.exists():
            return False
        
        # Check for essential model files
        # Model weights can be either pytorch_model.bin or model.safetensors
        has_config = (cache_path / "config.json").exists()
        has_preprocessor = (cache_path / "preprocessor_config.json").exists()
        has_model_weights = (cache_path / "pytorch_model.bin").exists() or (cache_path / "model.safetensors").exists()
        
        return has_config and has_preprocessor and has_model_weights
    
    def _download_and_cache_model(self, model_name: str, cache_path: Path, torch, BlipForConditionalGeneration, BlipProcessor):
        """Download model and save to cache directory."""
        print(
            f"[BLIP-CAPTION] Downloading model '{model_name}' to cache at {cache_path}. "
            "This may take several minutes..."
        )
        cache_path.mkdir(parents=True, exist_ok=True)

        processor = BlipProcessor.from_pretrained(model_name)
        processor.save_pretrained(cache_path)

        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.save_pretrained(cache_path)

        metadata = {
            "model_name": model_name,
            "download_complete": True,
            "cache_dir": str(cache_path),
        }
        with open(cache_path / "download_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("[BLIP-CAPTION] Model download complete.")
    
    def initialize(self, args: Any):
        # Lazy import heavy deps only here
        try:
            import torch
            from PIL import Image
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "BLIP model requires torch, transformers, and pillow to be installed"
            ) from exc

        self._torch = torch
        self._Image = Image
        self._BlipForConditionalGeneration = BlipForConditionalGeneration
        self._BlipProcessor = BlipProcessor

        # Use constants instead of environment variables
        model_name = DEFAULT_BLIP_MODEL_NAME
        device = DEFAULT_BLIP_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup cache directory
        cache_path = Path(BLIP_MODEL_CACHE_DIR)
        
        # Check if model is cached, if not download it
        if not self._is_model_cached(cache_path):
            print(f"[BLIP-CAPTION] Cache miss detected. Fetching model '{model_name}'.")
            self._download_and_cache_model(
                model_name, 
                cache_path, 
                torch, 
                BlipForConditionalGeneration, 
                BlipProcessor
            )
        
        model_path = str(cache_path)
        print(f"[BLIP-CAPTION] Loading model from cache at {model_path} on device {device}.")
        self._device = device
        
        # Load processor and model from cache
        self._processor = self._BlipProcessor.from_pretrained(model_path)
        
        self._model = (
            self._BlipForConditionalGeneration.from_pretrained(model_path).to(device)
        )
        self._model.eval()
        
        print(f"[BLIP-CAPTION] Model initialized successfully on {device}.")

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
                except (json.JSONDecodeError, TypeError, ValueError):
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

    def _decode_image(self, image_b64: str):
        import base64
        import io

        if image_b64.startswith("data:"):
            try:
                image_b64 = image_b64.split(",", 1)[1]
            except Exception:
                pass
        image_bytes = base64.b64decode(image_b64)
        return self._Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _generate_caption(self, pil_img, *, max_length: int, num_beams: int, no_repeat_ngram_size: int) -> str:
        torch = self._torch
        inputs = self._processor(pil_img, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_length=int(max_length),
                num_beams=int(num_beams),
                early_stopping=False,
                no_repeat_ngram_size=int(no_repeat_ngram_size),
            )
        return self._processor.decode(out[0], skip_special_tokens=True)

    def _as_bytes(self, elem) -> bytes:
        """Convert element to bytes, handling numpy.bytes_, bytes, or objects with .tobytes()."""
        if hasattr(elem, "tobytes"):
            return elem.tobytes()
        elif isinstance(elem, (bytes, bytearray)):
            return bytes(elem)
        else:
            return bytes(elem)
    
    def _error_payload(self, error_msg: str, params: dict = None) -> bytes:
        """Build a JSON error response payload."""
        return json.dumps({
            "caption": None,
            "error": error_msg,
            "params": params or {}
        }).encode("utf-8")

    def execute(self, requests):
        # print(f"[BLIP-CAPTION] Handling {len(requests)} inference request(s).")
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "input")
            if in_tensor is None:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "output",
                                np.array(
                                    [self._error_payload("missing input tensor")],
                                    dtype=object,
                                ).reshape(1, 1),
                            )
                        ]
                    )
                )
                continue

            batch = in_tensor.as_numpy().reshape(-1)
            outs: List[bytes] = []
            for elem in batch:
                try:
                    raw = self._as_bytes(elem)
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    outs.append(self._error_payload(f"invalid json: {exc}"))
                    continue
                image_b64 = payload.get("image_b64") or payload.get("image")
                max_length = int(payload.get("max_length", 50))
                num_beams = int(payload.get("num_beams", 3))
                no_repeat_ngram_size = int(payload.get("no_repeat_ngram_size", 2))
                if not image_b64 or not isinstance(image_b64, str):
                    outs.append(self._error_payload("missing image_b64"))
                    continue
                try:
                    pil_img = self._decode_image(image_b64)
                    caption = self._generate_caption(
                        pil_img,
                        max_length=max_length,
                        num_beams=num_beams,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    outs.append(
                        json.dumps(
                            {
                                "caption": caption,
                                "error": None,
                                "params": {
                                    "max_length": max_length,
                                    "num_beams": num_beams,
                                    "no_repeat_ngram_size": no_repeat_ngram_size,
                                },
                            }
                        ).encode("utf-8")
                    )
                except Exception as exc:
                    outs.append(self._error_payload(str(exc)))

            out_np = np.array(outs, dtype=object).reshape((-1, 1))
            out_tensor = pb_utils.Tensor("output", out_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        _STATUS_TRACKER.write_status()
        return responses

    def finalize(self):
        try:
            del self._model
        except Exception:
            pass
