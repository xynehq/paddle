# XYNE PaddleX High-Stability Serving

Production deployment of PP-StructureV3 on Triton Inference Server, extended for XYNE/Juspay workloads. This project starts from the PaddleX High-Stability Serving (HPS) SDK for PP-StructureV3[^pp-structure-sdk][^paddlex-docs] and layers operational tooling, performance tuning, and optional multimodal captioning that we run in production at [xynehq/xyne](https://github.com/xynehq/xyne/) and across other Juspay services.

## Architecture Overview
- **paddlex-server** container packages the PP-StructureV3 layout pipeline as a Triton Python backend (model `layout-parsing`) and ships the pipeline configs from `server/`.
- **blip-server** (optional) hosts a separate Triton instance serving BLIP image captioning (`blip-caption`) that model.py can call asynchronously for richer markdown output.
- **status_server.py** exposes `/instance_status` and `/paddlex_instance_status` on port `8081`, aggregating per-instance JSON heartbeats written by each Triton backend.
- All services inherit the PaddleX HPS runtime (HTTP :8000, gRPC :8001, Metrics :8002) from `paddle/hps:paddlex3.1-gpu`, with additional Python dependencies for PyMuPDF, BLIP, and Triton clients installed in the `Dockerfile`.

```
+--------------+        gRPC/HTTP         +------------+
| paddlex-server|  <--------------------> | clients    |
|  layout-parsing  |                      +------------+
|  status_server   |                            ^
+---------+--------+                            |
          | async captions                      |
          v                                      |
+---------+--------+   gRPC:8004 / HTTP:8003    |
|   blip-server    |<----------------------------+
|   blip-caption   |
+------------------+
```

## Key Enhancements Beyond the Base SDK
- **Instance-aware scheduling:** `server/model_repo/layout-parsing/1/model.py` integrates `InstanceStatusTracker` with `/tmp/triton_instance_status_*.json` heartbeats, exposes configured/active counts, and honours overrides from `TRITON_INSTANCE_COUNT` and `config_*.pbtxt`[^layout-model].
- **Async image captioning pipeline:** `CaptionCoordinator` in `layout_captioning.py` streams markdown images to the BLIP Triton model, merges captions back into `parsing_res_list`, and is gated by `caption_config.yaml` plus the global `IMAGE_CAPTIONING_ENABLED` env flag[^layout-caption].
- **Memory hygiene for long PDFs:** Page rendering streams via bounded queues, `gc.collect()` is invoked per page, `paddle.device.cuda.empty_cache()` is called after each request, and a `PADDLE_TRIM_CACHE_EVERY_N` knob controls periodic GPU cache trimming[^layout-model].
- **Resilient media uploads:** `app_common.postprocess_images` is wrapped with `_POSTPROCESS_UPLOAD_TIMEOUT_SECONDS`, giving fast failures and actionable logs when downstream storage stalls.
- **Consolidated configuration:** Model/backend matrices live in `configFiles/modelSupport.json` and `configFiles/modeSupportGpu.json`, while pipeline variants for Paddle, Paddle+HP, and ONNX Runtime runtimes sit alongside the shipping `server/pipeline_config.yaml`.

## Repository Layout
- `server/model_repo/layout-parsing/1/` – Triton Python backend using PaddleX pipeline, caption orchestration, and active-instance tracking.
- `server/model_repo/blip-caption/1/` – Triton Python backend that lazy-loads and caches `Salesforce/blip-image-captioning-large` with JSON IO.
- `server/status_server.py` – threadsafe HTTP server that aggregates `/tmp/*_instance_status*.json`.
- `configFiles/` – curated runtime configs (`pipeline_config *.yaml`) and backend capability matrices (`modelSupport.json`, `modeSupportGpu.json`).
- `Dockerfile`, `docker-compose*.yml` – container build/test orchestration for local dev and production.
- `client/` – lightweight gRPC helper (`client.py`) and requirements for smoke tests against :8001.

## Pipeline Runtime Configurations
Choose the right pipeline manifest and set `PADDLEX_HPS_PIPELINE_CONFIG_PATH` before launching `server.sh`:

| File | Purpose |
| ---- | ------- |
| `server/pipeline_config.yaml` | Default shipping config with Paddle backend and tuned thresholds for production. |
| `configFiles/pipeline_config Paddle.yaml` | Legacy pure-Paddle baseline (HP Inference disabled) for compatibility testing. |
| `configFiles/pipeline_config paddleHpiTrue.yaml` | Paddle backend with `use_hpip: True` to enable High Performance Inference acceleration on supported modules. |
| `configFiles/pipeline_config onnxRuntime.yaml` | Hybrid config pushing layout detection and OCR classifiers through ONNX Runtime for lower latency on NVIDIA GPUs. |

`server/server.sh` copies the chosen manifest into the runtime image, resolves `config_{gpu,cpu}.pbtxt` symlinks, and boots Triton with `--model-repository` set to `/paddlex/var/paddlex_model_repo`.

## Triton Model Configuration
- Layout parsing uses `config_gpu_paddlex.pbtxt` (`instance_group.count: 6`) for GPU deployments and `server/model_repo/layout-parsing/config_cpu.pbtxt` for CPU fallbacks.
- BLIP captioning relies on `config_gpu_blip.pbtxt` / `server/model_repo/blip-caption/config_gpu.pbtxt` (`instance_group.count: 4`, dynamic batching up to 16) and shares the same image.
- Update `TRITON_INSTANCE_COUNT` to ensure the status tracker reflects manual scaling, or edit the `count:` fields and reload the model.

### Backend Capability Matrices
We expose PaddleX model-to-backend compatibility as JSON so downstream services can reason about runtime choices:

```jsonc
// configFiles/modeSupportGpu.json (excerpt)
{
  "gpu_cuda118_cudnn89": {
    "PP-DocLayout_plus-L": ["tensorrt_fp16", "paddle", "onnxruntime"],
    "PP-DocBlockLayout": ["tensorrt", "paddle", "onnxruntime"],
    "PP-OCRv5_server_det": ["tensorrt", "paddle"],
    "PP-OCRv5_server_rec": ["paddle_tensorrt_fp16", "tensorrt_fp16", "onnxruntime", "paddle"],
    "SLANet_plus": ["paddle"],
    "SLANeXt_wired": ["paddle"]
  }
}
```

See the full JSON files for the exhaustive matrix.

## Building the Image
Prerequisites: Docker Engine with NVIDIA Container Toolkit, CUDA-capable GPU (≥12 GB recommended), and access to the PaddleX3.1 base image registry.

```bash
# Build the Triton-serving image
docker build -t paddlex/app:latest .
```

## Running in Development
```bash
# Start layout parsing only (captioning disabled, lighter footprint)
IMAGE_CAPTIONING_ENABLED=false docker compose up paddlex-server

# Start both layout parsing and blip-caption (captioning enabled)
COMPOSE_PROFILES=image-captioning IMAGE_CAPTIONING_ENABLED=true \
  docker compose up

# Tail logs
docker logs -f paddlex-server
docker logs -f blip-server   # only if profile enabled
```

- `docker-compose.yml` mounts the repo into `/app`, reuses `config_gpu_paddlex.pbtxt`, and exposes ports `8000/8001/8002/8081`.
- Captioning defaults on; flip `IMAGE_CAPTIONING_ENABLED` or remove the `image-captioning` profile to opt out.

## Running in Production
```bash
# Build once, then deploy with production compose
docker compose -f docker-compose.prod.yml build

# Keep captioning enabled
COMPOSE_PROFILES=image-captioning IMAGE_CAPTIONING_ENABLED=true \
  docker compose -f docker-compose.prod.yml up -d

# Or disable captioning globally while keeping layout parsing online
IMAGE_CAPTIONING_ENABLED=false \
  docker compose -f docker-compose.prod.yml up -d paddlex-server
```

Production compose avoids bind-mounting the workspace, instead mounting curated model caches (`paddlex/official_models`) and optional BLIP cache directories. Ensure the external Docker network `xyne` exists (`docker network create xyne`) before bringing up the stack.

### Bare-Metal Launch
For on-host Triton, execute `server/server.sh` after setting:

```bash
export PADDLEX_HPS_PIPELINE_CONFIG_PATH=/path/to/pipeline_config.yaml
export IMAGE_CAPTIONING_ENABLED=true
./server/server.sh
```

The script mirrors the repo to `/paddlex/var/paddlex_model_repo`, boots the status server, and launches Triton with explicit model loads.

## Monitoring & Operations
- Health metrics live at `/paddlex_instance_status` (layout-parsing active/idle counts) and `/instance_status` (layout + BLIP configured instances) on port `8081`.
- Runtime tuning knobs:
  - `TRITON_POSTPROCESS_TIMEOUT_SECONDS` – fail fast when image uploads hang.
  - `PADDLE_TRIM_CACHE_EVERY_N` – force `paddle.device.cuda.empty_cache()` periodically for long-running jobs.
  - `IMAGE_CAPTIONING_ENABLED` – global kill-switch for caption orchestration.
  - `TRITON_INSTANCE_COUNT` – advertises pre-scaled pool sizes to the status tracker.

## Using the gRPC Client
The `client/` folder includes a minimal smoke-test harness:

```bash
pip install -r client/requirements.txt
python client/client.py --file path/to/document.pdf --file-type 0 --url localhost:8001
```

Generated markdown and images are written to `./markdown_<page>/`, mirroring how downstream services consume the layout parsing output.

## Benchmark Highlights
We evaluated representative PDFs (Juspay pitch deck, LIC forms) across backends and GPUs. Times are wall-clock per document with a single instance unless noted.

| Mode | GPU / Machine | Document | Pages | Size | Latency | Peak GPU Mem | Quality | Notes |
| ---- | ------------- | -------- | ----- | ---- | ------- | ------------ | ------- | ----- |
| Paddle basic (UAT) | RTX 4090 (local) | juspay.pdf | 41 | 22 MB | 28 s | 6 GB + spike | Excellent | Baseline profile currently in UAT. |
| Paddle basic (UAT) | L4 (UAT) | juspay.pdf | 41 | 22 MB | 112 s | 6 GB + spike | Excellent | Longer wall-time on L4 due to lower clocks. |
| Paddle basic (UAT) | RTX 4090 (local) | LIC.pdf | 93 | 12 MB | 65 s | 6 GB + spike | Excellent | — |
| Paddle basic (UAT) | L4 (UAT) | LIC.pdf | 93 | 12 MB | 150–180 s | 6 GB + spike | Excellent | — |
| TensorRT fp16 | RTX 4090 (local) | juspay.pdf | 41 | 22 MB | 13 s | 9 GB + 4 GB | Poor | Conversion skipped layers → unacceptable accuracy. |
| TensorRT fp32 | RTX 4090 (local) | juspay.pdf | 41 | 22 MB | 18 s | 12 GB + 4 GB | Poor | Same degradation as fp16. |
| ONNX Runtime | RTX 4090 (local) | juspay.pdf | 41 | 22 MB | 19 s | 12 GB + 6 GB | Excellent | Competitive alternative. |
| Paddle + TensorRT subgraph fp32 | RTX 4090 (local) | juspay.pdf | 41 | 22 MB | 17 s | 6 GB + 2.5 GB | Excellent | Selected for production rollout. |
| Paddle + TensorRT subgraph fp32 | A100 | juspay.pdf | 41 | 22 MB | 53 s | 6 GB + 2.5 GB | Excellent | Multi-instance friendly. |
| Paddle + TensorRT subgraph fp32 | RTX 4090 (local) | LIC.pdf | 93 | 12 MB | 35 s | 6 GB + 2.5 GB | Excellent | 40 % faster than UAT baseline. |
| Paddle + TensorRT subgraph fp32 | A100 | LIC.pdf | 93 | 12 MB | 92 s | 6 GB + 2.5 GB | Excellent | Up to 7 instances per A100 feasible. |

Notes:
- Latency grows ~30 % when scaling out instances; the status tracker helps right-size pools.
- Formula recognition and seal detection remain disabled to prioritise throughput (adds 0.5–1 s per page when enabled).
- Horizontal scaling is VRAM-bound; A100 deployments comfortably host seven concurrent instances (vs three on UAT).

## Licensing & Attribution
- PaddleOCR, PP-StructureV3, and PaddleX assets are Apache License 2.0. Review upstream license text at [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
- BLIP weights (`Salesforce/blip-image-captioning-large`) inherit their respective Hugging Face licensing terms.
- Cite the upstream research when publishing results built on this stack:
  - PaddleOCR 3.0 Technical Report[^cui2025a]
  - PaddleOCR-VL[^cui2025b]

## References
[^pp-structure-sdk]: PaddleX HPS PP-StructureV3 SDK download – https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.3/paddlex_hps_PP-StructureV3_sdk.tar.gz
[^paddlex-docs]: PaddleX High-Stability Serving documentation – https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/serving.html#13-invoke-the-service
[^layout-model]: `server/model_repo/layout-parsing/1/model.py`
[^layout-caption]: `server/model_repo/layout-parsing/1/layout_captioning.py`
[^cui2025a]: Cheng Cui et al. *PaddleOCR 3.0 Technical Report*. arXiv:2507.05595, 2025.
[^cui2025b]: Cheng Cui et al. *PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model*. arXiv:2510.14528, 2025.
