import json
import os
from typing import Any, List

import numpy as np
import triton_python_backend_utils as pb_utils


# Defaults for BLIP config (no env required)
DEFAULT_BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"
# Set to None for auto (cuda if available else cpu)
DEFAULT_BLIP_DEVICE = None


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

        self._device = device
        self._processor = BlipProcessor.from_pretrained(model_name)
        self._model = (
            BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        )
        self._model.eval()

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

    def execute(self, requests):
        print(f"[BLIP-CAPTION] Received {len(requests)} request(s)")
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
                                    [
                                        json.dumps(
                                            {
                                                "caption": None,
                                                "error": "missing input tensor",
                                                "params": {},
                                            }
                                        ).encode("utf-8")
                                    ],
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
                    # elem can be numpy.bytes_, bytes, or an object scalar with .tobytes()
                    if hasattr(elem, "tobytes"):
                        raw = elem.tobytes()
                    elif isinstance(elem, (bytes, bytearray)):
                        raw = bytes(elem)
                    else:
                        raw = bytes(elem)
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    outs.append(
                        json.dumps({"caption": None, "error": f"invalid json: {exc}", "params": {}}).encode("utf-8")
                    )
                    continue
                image_b64 = payload.get("image_b64") or payload.get("image")
                max_length = int(payload.get("max_length", 50))
                num_beams = int(payload.get("num_beams", 3))
                no_repeat_ngram_size = int(payload.get("no_repeat_ngram_size", 2))
                if not image_b64 or not isinstance(image_b64, str):
                    outs.append(
                        json.dumps({"caption": None, "error": "missing image_b64", "params": {}}).encode("utf-8")
                    )
                    continue
                try:
                    print(f"[BLIP-CAPTION] Decoding image and generating caption...")
                    pil_img = self._decode_image(image_b64)
                    caption = self._generate_caption(
                        pil_img,
                        max_length=max_length,
                        num_beams=num_beams,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    print(f"[BLIP-CAPTION] Generated caption: {caption[:100]}...")
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
                    outs.append(
                        json.dumps({"caption": None, "error": str(exc), "params": {}}).encode("utf-8")
                    )

            out_np = np.array(outs, dtype=object).reshape((-1, 1))
            out_tensor = pb_utils.Tensor("output", out_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        try:
            del self._model
        except Exception:
            pass
