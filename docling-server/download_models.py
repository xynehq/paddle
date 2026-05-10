"""Download docling models to DOCLING_ARTIFACTS_PATH for offline use."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ARTIFACTS_PATH = Path(os.environ.get("DOCLING_ARTIFACTS_PATH", "/models"))
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# Docling uses naming convention: org/repo → org--repo
REPOS = [
    ("docling-project/docling-layout-heron", "main"),
    ("docling-project/docling-models", "v2.3.0"),
]

for repo_id, revision in REPOS:
    # Convert org/repo to org--repo as docling expects
    local_name = repo_id.replace("/", "--")
    local_dir = ARTIFACTS_PATH / local_name
    
    print(f"Downloading {repo_id}@{revision} → {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir)
    )
    print(f"  Done: {local_name}/")

print(f"All models downloaded to {ARTIFACTS_PATH}")
print("Contents:")
for p in sorted(ARTIFACTS_PATH.iterdir()):
    print(f"  {p.name}/")
