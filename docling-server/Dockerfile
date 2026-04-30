# Docling Server - Docker Image
#
# BUILD (on a machine with internet):
#   docker build --platform linux/amd64 -t docling-server .
#   docker save docling-server | gzip > docling-server.tar.gz
#
# DEPLOY (air-gapped machine):
#   docker load -i docling-server.tar.gz
#   docker run -p 8000:8000 \
#     -e REMOTE_VLM_PRESET=lightonocr \
#     -e GPU_INSTANCE_URL=http://<gpu-host> \
#     -e VLM_PORT=8001 \
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

# Fix HuggingFace cache to a known path so runtime can find what was
# downloaded at build time.
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Download docling models at build time so the image is fully self-contained.
# This adds ~500 MB to the image but means zero network calls at runtime.
COPY download_models.py .
RUN python3 download_models.py && rm download_models.py

# Lock HuggingFace into offline mode — no network calls ever at runtime.
# The models are already in the image from the RUN step above.
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

