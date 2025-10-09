#!/usr/bin/env python3
"""
Resolve model_dir entries in a PaddleX pipeline config.

For each occurrence of:
  model_name: <NAME>
  model_dir: <...>

Replace model_dir with the first existing candidate path:
  - <BASE_DIR>/<NAME>

If none exist, set model_dir to null so that the runtime can
download by model_name from the official repository.

Usage:
  python resolve_model_dirs.py --config ../pipeline_config.yaml \
      --base-dir /root/.paddlex/official_models

Multiple base dirs can be provided by repeating --base-dir.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List


def resolve_model_dirs(
    config_path: str, base_dirs: List[str], dry_run: bool = False
) -> int:
    # Normalize and filter base_dirs
    bases = [os.path.abspath(b) for b in base_dirs if b]

    # Read file preserving line endings
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)

    name_re = re.compile(r"^\s*model_name:\s*[\"']?([^\"']+?)[\"']?\s*$")
    dir_re = re.compile(r"^(\s*model_dir:\s*)(.*?)(\s*)(#.*)?$")

    pending_name: str | None = None
    changed = False
    changes: List[str] = []

    for i, line in enumerate(lines):
        mname = name_re.match(line)
        if mname:
            pending_name = mname.group(1).strip()
            continue

        mdir = dir_re.match(line)
        if mdir and pending_name:
            prefix, _value, ws, comment = (
                mdir.group(1),
                mdir.group(2),
                mdir.group(3) or "",
                mdir.group(4) or "",
            )

            chosen = None
            for base in bases:
                cand = os.path.join(base, pending_name)
                if os.path.isdir(cand):
                    chosen = cand
                    break

            new_value = "null" if chosen is None else chosen
            new_line = f"{prefix}{new_value}{ws}{comment}\n"

            if new_line != line:
                lines[i] = new_line
                changed = True
                changes.append(
                    f"{pending_name}: model_dir -> {new_value}"
                )

            # Reset after applying to the corresponding model_dir
            pending_name = None

    if dry_run:
        for c in changes:
            print(c)
        return 0

    if changed:
        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Updated {config_path} with {len(changes)} model_dir entries.")
        for c in changes:
            print(f" - {c}")
    else:
        print("No changes needed. All model_dir entries already correct.")

    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline_config.yaml",
    )
    parser.add_argument(
        "--base-dir",
        action="append",
        default=["/root/.paddlex/official_models"],
        help="Base directory to check for model folders (can be repeated).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the changes without modifying the file.",
    )
    args = parser.parse_args(argv)

    return resolve_model_dirs(args.config, args.base_dir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
