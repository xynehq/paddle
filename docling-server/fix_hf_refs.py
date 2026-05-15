"""
Run after download_models.py to fix HuggingFace hub ref files.

Cross-platform builds (linux/amd64 on Apple Silicon via QEMU) sometimes fail
to write the refs/main pointer file that maps the branch name to a snapshot
commit hash.  Without it, HF_HUB_OFFLINE=1 cannot locate the cached model.

This script scans every model in the HF cache and writes refs/main if missing.
"""
from pathlib import Path

hub_cache = Path("/root/.cache/huggingface/hub")

if not hub_cache.exists():
    print("HF hub cache not found — nothing to fix.")
    raise SystemExit(0)

for model_dir in sorted(hub_cache.iterdir()):
    if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
        continue

    snapshots_dir = model_dir / "snapshots"
    refs_dir = model_dir / "refs"
    refs_main = refs_dir / "main"

    if not snapshots_dir.exists():
        continue

    snapshots = [s for s in snapshots_dir.iterdir() if s.is_dir()]
    if not snapshots:
        continue

    commit_hash = snapshots[0].name

    if refs_main.exists():
        existing = refs_main.read_text().strip()
        if existing == commit_hash:
            print(f"  OK  {model_dir.name}  refs/main → {commit_hash[:12]}")
        else:
            print(f"  ??  {model_dir.name}  refs/main points to {existing[:12]}, snapshot is {commit_hash[:12]}")
        continue

    refs_dir.mkdir(exist_ok=True)
    refs_main.write_text(commit_hash)
    print(f"  FIX {model_dir.name}  wrote refs/main → {commit_hash[:12]}")

print("Done.")
