import base64
import io
import os
from typing import Optional

from flask import Flask, jsonify, request
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


# --- App and model setup ---
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load once at startup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
model.eval()


def _decode_image_b64(image_b64: str) -> Image.Image:
    """Decode a base64 image string (optionally with data URL prefix) to a PIL Image."""
    # Strip data URL prefix if present
    if image_b64.startswith("data:"):
        try:
            image_b64 = image_b64.split(",", 1)[1]
        except Exception:
            pass
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img


def generate_caption_pil(img: Image.Image,
                         max_length: int = 50,
                         num_beams: int = 3,
                         no_repeat_ngram_size: int = 2) -> str:
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": device,
        "model": "Salesforce/blip-image-captioning-large"
    })


@app.post("/caption")
def caption_endpoint():
    """Accepts JSON: {"image_b64": "..."} and returns {"caption": "..."}.

    Optional params: max_length, num_beams, no_repeat_ngram_size
    """
    if not request.is_json:
        return jsonify({"error": "Expected application/json body"}), 400

    data = request.get_json(silent=True) or {}
    image_b64: Optional[str] = data.get("image_b64") or data.get("image")
    if not image_b64 or not isinstance(image_b64, str):
        return jsonify({"error": "Missing image_b64 (base64-encoded image)"}), 400

    try:
        img = _decode_image_b64(image_b64)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {e}"}), 400

    # Optional generation params
    max_length = int(data.get("max_length", 50))
    num_beams = int(data.get("num_beams", 3))
    no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", 2))

    try:
        caption = generate_caption_pil(
            img,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    except Exception as e:
        return jsonify({"error": f"Caption generation failed: {e}"}), 500

    return jsonify({
        "caption": caption,
        "params": {
            "max_length": max_length,
            "num_beams": num_beams,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
    })


if __name__ == "__main__":
    # Configure host/port via env if desired
    host = os.environ.get("CAPTION_HOST", "0.0.0.0")
    port = int(os.environ.get("CAPTION_PORT", "5001"))
    # Disable threaded to avoid multiple concurrent initializations; model is global anyway
    app.run(host=host, port=port, debug=False, threaded=False)

