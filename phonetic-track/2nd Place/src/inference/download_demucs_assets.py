#!/usr/bin/env python3
"""
Download Demucs pretrained assets for offline runtime use.

This script creates a local Demucs model repo (default: offline_wheels/demucs_repo)
that can be loaded with:
    demucs.pretrained.get_model(<model_name>, repo=<repo_dir>)

Example:
    uv run src/inference/download_demucs_assets.py --model mdx_extra
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

from demucs import pretrained


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def parse_remote_files(files_txt_path: Path) -> tuple[str, dict[str, str]]:
    root = ""
    models: dict[str, str] = {}
    for raw_line in files_txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("root:"):
            root = line.split(":", 1)[1].strip()
            continue
        sig = line.split("-", 1)[0]
        models[sig] = pretrained.ROOT_URL + root + line
    return root, models


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] {dst.name} already exists")
        return
    print(f"[download] {dst.name}")
    urllib.request.urlretrieve(url, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mdx_extra", help="Demucs model bag name (e.g. mdx_extra)")
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "offline_wheels" / "demucs_repo",
        help="Directory to store Demucs .yaml and .th files",
    )
    parser.add_argument(
        "--force-yaml",
        action="store_true",
        help="Overwrite local yaml file if it already exists",
    )
    args = parser.parse_args()

    remote_dir = Path(pretrained.__file__).resolve().parent / "remote"
    files_txt = remote_dir / "files.txt"
    bag_yaml = remote_dir / f"{args.model}.yaml"

    if not files_txt.exists():
        print(f"[ERROR] Missing Demucs metadata file: {files_txt}", file=sys.stderr)
        sys.exit(1)
    if not bag_yaml.exists():
        print(f"[ERROR] Unknown Demucs model '{args.model}'. Missing yaml: {bag_yaml}", file=sys.stderr)
        sys.exit(1)

    _, sig_to_url = parse_remote_files(files_txt)

    bag_cfg = yaml.safe_load(bag_yaml.read_text(encoding="utf-8"))
    signatures = bag_cfg.get("models", [])
    if not signatures:
        print(f"[ERROR] No model signatures found in {bag_yaml.name}", file=sys.stderr)
        sys.exit(1)

    args.dest.mkdir(parents=True, exist_ok=True)

    dst_yaml = args.dest / bag_yaml.name
    if args.force_yaml and dst_yaml.exists():
        dst_yaml.unlink()
    if not dst_yaml.exists():
        shutil.copy2(bag_yaml, dst_yaml)
        print(f"[copy] {dst_yaml.name}")
    else:
        print(f"[skip] {dst_yaml.name} already exists")

    missing: list[str] = []
    for sig in signatures:
        url = sig_to_url.get(sig)
        if url is None:
            missing.append(sig)
            continue
        filename = url.rsplit("/", 1)[-1]
        download_file(url, args.dest / filename)

    if missing:
        print(f"[ERROR] Missing signatures in files.txt: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Local Demucs repo ready at: {args.dest}")
    print("Use this at runtime with get_model(model_name, repo=<that_dir>).")


if __name__ == "__main__":
    main()
