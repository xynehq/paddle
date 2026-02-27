#!/usr/bin/env bash

set -e

export LANG='C.UTF-8'
export PADDLEX_HPS_LOGGING_LEVEL='INFO'

export PADDLEX_HPS_PIPELINE_CONFIG_PATH="${PADDLEX_HPS_PIPELINE_CONFIG_PATH:-$(realpath pipeline_config.yaml)}"

# Do we need a unique directory?
readonly MODEL_REPO_DIR=/paddlex/var/paddlex_model_repo

rm -rf "${MODEL_REPO_DIR}"

cp -r model_repo "${MODEL_REPO_DIR}"

# TODO: Check if environment variables are set
find "${MODEL_REPO_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' dir_; do
    if [ -f "${dir_}/config_${PADDLEX_HPS_DEVICE_TYPE}.pbtxt" ]; then
        cp -f "${dir_}/config_${PADDLEX_HPS_DEVICE_TYPE}.pbtxt" "${dir_}/config.pbtxt"
    fi
done

if [ -d shared_mods ]; then
    export PYTHONPATH="$(realpath shared_mods):${PYTHONPATH}"
fi

# Resolve model_dir entries dynamically based on available local models
python3 "$(dirname "$0")/resolve_model_dirs.py" \
    --config "${PADDLEX_HPS_PIPELINE_CONFIG_PATH}" \
    --base-dir "/root/.paddlex/official_models"

# Start the standalone instance status server in the background
python3 "$(dirname "$0")/status_server.py" &
STATUS_SERVER_PID=$!

# Start Swagger API in the background (same container) when enabled
SWAGGER_ENABLED_RAW="${SWAGGER_ENABLED:-true}"
SWAGGER_ENABLED_NORMALIZED="$(printf '%s' "${SWAGGER_ENABLED_RAW}" | tr '[:upper:]' '[:lower:]')"
SWAGGER_API_PID=""
if [ "${SWAGGER_ENABLED_NORMALIZED}" = "1" ] || [ "${SWAGGER_ENABLED_NORMALIZED}" = "true" ] || [ "${SWAGGER_ENABLED_NORMALIZED}" = "yes" ] || [ "${SWAGGER_ENABLED_NORMALIZED}" = "on" ]; then
    export TRITON_HTTP_URL="${TRITON_HTTP_URL:-http://localhost:8000}"
    export STATUS_URL="${STATUS_URL:-http://localhost:8081/instance_status}"
    export PADDLEX_STATUS_URL="${PADDLEX_STATUS_URL:-http://localhost:8081/paddlex_instance_status}"
    export PADDLEX_HPS_PIPELINE_CONFIG_PATH="${PADDLEX_HPS_PIPELINE_CONFIG_PATH:-$(realpath pipeline_config.yaml)}"
    uvicorn api.app:app --app-dir /app --host 0.0.0.0 --port "${SWAGGER_PORT:-8088}" &
    SWAGGER_API_PID=$!
fi

# Trap to ensure background services are killed when tritonserver exits
trap 'kill $STATUS_SERVER_PID 2>/dev/null; if [ -n "$SWAGGER_API_PID" ]; then kill $SWAGGER_API_PID 2>/dev/null; fi' EXIT

# Start tritonserver in the background
tritonserver --model-repository="${MODEL_REPO_DIR}" --model-control-mode=explicit --load-model=layout-parsing --backend-config=python,shm-default-byte-size=104857600,shm-growth-byte-size=10485760 --http-port=8000 --grpc-port=8001 --metrics-port=8002 --log-info=1 --log-warning=1 --log-error=1
