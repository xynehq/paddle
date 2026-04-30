# Docling Server - Docker Image
#
# BUILD (on a machine with internet):
#   docker build --platform linux/amd64 -t docling-server .
#   docker save docling-server | gzip > docling-server.tar.gz
#
# DEPLOY (air-gapped machine):
#   docker load -i docling-server.tar.gz
#   docker run -p 8000:8000 \
#     -e VLM_PRESET=lightonocr \
#     -e VLM_URL=https://<vlm-endpoint>/v1/chat/completions \
#     -e VLM_MODEL=lightonai/LightOnOCR-2-1B \
#     -e VLM_ACCESS_TOKEN=<token> \
#     docling-server

FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies required by docling / PIL / OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Where docling will find its models — both at build time and runtime.
# Docling looks for: $DOCLING_ARTIFACTS_PATH/<org>--<repo>/
ENV DOCLING_ARTIFACTS_PATH=/models

# Download docling models at build time so the image is fully self-contained.
COPY download_models.py ./
RUN python3 download_models.py && rm download_models.py

# Disable HuggingFace network access at runtime — everything is baked in.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy application code
COPY main.py .
COPY jina-tokenizer ./jina-tokenizer

RUN python3 -c "from pathlib import Path; assert Path('/app/jina-tokenizer/tokenizer.json').exists(), 'tokenizer.json missing'"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

