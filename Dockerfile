# Start from the PaddleX HPS GPU base image
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.1-gpu

# Set environment variables
ENV PADDLEX_HPS_DEVICE_TYPE=gpu

# Install python dependencies needed by the Triton Python backend
RUN python3 -m pip install --no-cache-dir \
    "pymupdf>=1.24" \
    "pillow>=10.4,<11.0" \
    "numpy>=1.24" \
    "PyYAML>=6.0" \
    "safetensors>=0.4" \
    "sentencepiece>=0.1.99" \
    "transformers>=4.40,<5" \
    "tritonclient[all]>=2.41,<3" \
    && python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch \
    && rm -rf /root/.cache/pip

# # Install PyTorch (CUDA build). Adjust cu version if needed
# RUN python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
#     torch



WORKDIR /app

COPY server/ /app/server

WORKDIR /app/server

# Default shared memory size can’t be set in Dockerfile — pass via `--shm-size` at runtime
# Same for `--gpus all`, `--network host`, and `--init` — these are runtime options.

# Run your script
CMD ["/bin/bash", "server.sh"]
EXPOSE 8000
EXPOSE 8001
EXPOSE 8004
