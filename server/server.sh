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

# Align the Triton instance count with what the model configs request.
instance_count="$(
    MODEL_REPO_DIR="${MODEL_REPO_DIR}" python3 <<'PY'
import os
import pathlib
import re

model_repo = pathlib.Path(os.environ["MODEL_REPO_DIR"])
device = os.environ.get("PADDLEX_HPS_DEVICE_TYPE")

config_paths = sorted(model_repo.rglob("config.pbtxt"))
if not config_paths and device:
    config_paths = sorted(model_repo.rglob(f"config_{device}.pbtxt"))

total = 0
for path in config_paths:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        continue
    for match in re.finditer(r"count\s*:\s*(\d+)", text):
        total += int(match.group(1))

print(total)
PY
)"
instance_count="${instance_count//[[:space:]]/}"
if [ -n "${instance_count}" ]; then
    export TRITON_INSTANCE_COUNT="${instance_count}"
    printf 'Resolved TRITON_INSTANCE_COUNT=%s from model configs\n' "${instance_count}"
fi

if [ -d shared_mods ]; then
    export PYTHONPATH="$(realpath shared_mods):${PYTHONPATH}"
fi

# Resolve model_dir entries dynamically based on available local models
# python3 "$(dirname "$0")/resolve_model_dirs.py" \
#     --config "${PADDLEX_HPS_PIPELINE_CONFIG_PATH}" \
#     --base-dir "/root/.paddlex/official_models"

exec tritonserver --model-repository="${MODEL_REPO_DIR}" --backend-config=python,shm-default-byte-size=104857600,shm-growth-byte-size=10485760 --log-info=1 --log-warning=1 --log-error=1
