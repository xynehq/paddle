"""Download docling models directly from HuggingFace at build time.

Docling resolves models from DOCLING_ARTIFACTS_PATH using the folder naming
convention: <org>--<repo-name>  (slashes replaced with double-dash).

So DOCLING_ARTIFACTS_PATH=/models and repo_id=docling-project/docling-layout-heron
→ docling looks in /models/docling-project--docling-layout-heron/
"""
from pathlib import Path
from huggingface_hub import snapshot_download

ARTIFACTS_PATH = Path("/models")
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

MODELS = [
    # Layout analysis model (object detection)
    "docling-project/docling-layout-heron",
    # TableFormer (table structure recognition)
    "docling-project/docling-models",
]

for repo_id in MODELS:
    # docling's naming convention: org/repo → org--repo
    local_dir = ARTIFACTS_PATH / repo_id.replace("/", "--")
    print(f"Downloading {repo_id} → {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"  Done: {repo_id}")

print("All models downloaded.")
print("Contents of /models:")
for p in sorted(ARTIFACTS_PATH.iterdir()):
    print(f"  {p.name}/")
