# Start from the PaddleX HPS GPU base image
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.1-gpu

# Set environment variables
ENV PADDLEX_HPS_DEVICE_TYPE=gpu

# Install python dependencies needed by the Triton Python backend
RUN python3 -m pip install --no-cache-dir \
    "pymupdf>=1.24" \
    "pillow>=10.4,<11.0" \
    "numpy>=1.24"

# Set working directory inside the container
WORKDIR /app

# Copy your application files into the container
# (adjust the COPY path if server.sh and code are not in current dir)
COPY . /app

WORKDIR /app/server

# Default shared memory size can’t be set in Dockerfile — pass via `--shm-size` at runtime
# Same for `--gpus all`, `--network host`, and `--init` — these are runtime options.

# Run your script
CMD ["/bin/bash", "server.sh"]
EXPOSE 8000
EXPOSE 8001
