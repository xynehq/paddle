"""Run at docker build time to download docling models into the HuggingFace
hub cache where docling's runtime expects to find them.

Revisions match docling's hardcoded constants:
  - docling-project/docling-layout-heron  → main  (DOCLING_LAYOUT_HERON.revision)
  - docling-project/docling-models        → v2.3.0 (TableStructureModel)

Mismatched revisions cause LocalEntryNotFoundError at runtime even when the
repo is cached, because snapshot_download(revision=...) checks for the exact
tagged snapshot.
"""
from huggingface_hub import snapshot_download

REPOS = [
    ("docling-project/docling-layout-heron", "main"),
    ("docling-project/docling-models", "v2.3.0"),
]

for repo, revision in REPOS:
    print(f"Downloading {repo}@{revision} ...")
    snapshot_download(repo_id=repo, revision=revision)
print("Models downloaded and cached.")
