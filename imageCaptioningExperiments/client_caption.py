import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def encode_image_to_b64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description="Call BLIP caption server with a local image")
    parser.add_argument("image", type=Path, help="Path to image (jpg/png)")
    parser.add_argument("--server", default="http://localhost:5001", help="Server base URL")
    parser.add_argument("--endpoint", default="/caption", help="Endpoint path")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        sys.exit(1)

    image_b64 = encode_image_to_b64(args.image)

    payload = {
        "image_b64": image_b64,
        "max_length": args.max_length,
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    url = args.server.rstrip("/") + args.endpoint
    try:
        resp = requests.post(url, json=payload, timeout=300)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(2)

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(3)

    try:
        data = resp.json()
    except Exception as e:
        print(f"Failed to parse JSON: {e}\n{resp.text}")
        sys.exit(4)

    print(json.dumps(data, indent=2, ensure_ascii=False))
    caption = data.get("caption")
    if caption:
        print(f"\nCaption: {caption}")


if __name__ == "__main__":
    main()

