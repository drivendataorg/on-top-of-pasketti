#!/usr/bin/env python3
"""
Compare our pyproject.toml against the base competition runtime pyproject.toml,
find the extra top-level dependencies we added, and download their full
transitive closure into offline_wheels/ so the submission can install them
without internet access.

Usage (from anywhere in the repo):
    python external/phonetic/parakeet-cmudict/download_wheels.py

Optional overrides:
    --base   path to the base runtime pyproject.toml   (default: runtime/pyproject.toml)
    --ours   path to our pyproject.toml                (default: pyproject.toml)
    --dest   path to the offline_wheels directory      (default: <script_dir>/offline_wheels)
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
import tarfile

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parents[1]  # .../external/phonetic/parakeet-cmudict -> repo root

PINNED_VERSIONS = {
    "torchcodec": "0.9.0",
    'numpy': '1.26.4',
}

def normalize(name: str) -> str:
    """PEP 503 normalization: lowercase, collapse runs of [-_.] to '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def extract_dep_names(deps: list[str]) -> set[str]:
    """
    Extract normalized package names from a list of PEP 508 dependency strings.
    Handles 'pkg>=1.0', 'pkg[extra]>=1.0', 'pkg @ git+https://...' etc.
    """
    names: set[str] = set()
    for dep in deps:
        # Strip inline comments
        dep = dep.split("#")[0].strip()
        if not dep:
            continue
        # Split on first version specifier, URL marker, or extra bracket
        name = re.split(r"[\s>=<!;@\[]", dep, maxsplit=1)[0].strip()
        if name:
            names.add(normalize(name))
    return names


def parse_pyproject_deps(path: Path) -> set[str]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    return extract_dep_names(deps)


def parse_pyproject_requirements(path: Path) -> dict[str, Requirement]:
    """Return {normalized-name: Requirement} parsed from project dependencies."""
    with path.open("rb") as f:
        data = tomllib.load(f)

    reqs: dict[str, Requirement] = {}
    deps = data.get("project", {}).get("dependencies", [])
    for dep in deps:
        dep = dep.split("#")[0].strip()
        if not dep:
            continue
        try:
            req = Requirement(dep)
            reqs[normalize(req.name)] = req
        except Exception:
            # Skip malformed/non-standard entries and rely on name-only parsing elsewhere.
            continue
    return reqs


def resolve_transitive_versions(requested_specs: list[str]) -> dict[str, str]:
    """
    Resolve transitive dependency versions using pip's dry-run report.
    Returns {normalized-name: resolved-version}.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "pip_report.json"
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--report",
            str(report_path),
            *requested_specs,
        ]

        print("Resolving dependency graph (dry-run)...")
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0 or not report_path.exists():
            raise RuntimeError("Failed to resolve dependencies using pip dry-run report")

        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

    resolved: dict[str, str] = {}
    for item in report.get("install", []):
        metadata = item.get("metadata", {})
        name = metadata.get("name")
        version = metadata.get("version")
        if name and version:
            resolved[normalize(name)] = str(version)
    return resolved


def is_version_covered_by_base(base_req: Requirement, resolved_version: str) -> bool:
    """True if resolved_version is acceptable under base runtime requirement."""
    spec = base_req.specifier
    if not str(spec):
        # Base runtime already has this package name with no explicit version bound.
        return True
    try:
        return Version(resolved_version) in spec
    except InvalidVersion:
        return False


def is_runtime_managed_torch_stack(dep_name: str) -> bool:
    """
    Heavy CUDA/Triton packages are typically shipped alongside runtime torch.
    Skip re-downloading these for offline bundle generation when torch is already
    available in the base runtime.
    """
    if dep_name == "triton":
        return True
    return dep_name.startswith("nvidia-") or dep_name.startswith("cuda-")


def get_existing_wheels(wheel_dir: Path) -> dict[str, str]:
    """Return {normalized-name: version} for each .whl file present."""
    existing: dict[str, str] = {}
    for whl in wheel_dir.glob("*.whl"):
        parts = whl.stem.split("-")
        if len(parts) >= 2:
            name = normalize(parts[0])
            version = parts[1]
            existing[name] = version
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=REPO_ROOT / "external" / "runtime" / "pyproject.toml",
                        help="Base competition runtime pyproject.toml")
    parser.add_argument("--ours", type=Path, default=REPO_ROOT / "pyproject.toml",
                        help="Our (submission) pyproject.toml")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "offline_wheels",
                        help="Destination directory for wheels")
    args = parser.parse_args()

    base_path: Path = args.base
    ours_path: Path = args.ours
    wheel_dir: Path = args.dest

    # ------------------------------------------------------------------ #
    # 1. Diff the two pyproject.toml files
    # ------------------------------------------------------------------ #
    if not base_path.exists():
        print(f"[ERROR] Base pyproject.toml not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    if not ours_path.exists():
        print(f"[ERROR] Our pyproject.toml not found: {ours_path}", file=sys.stderr)
        sys.exit(1)

    base_deps = parse_pyproject_deps(base_path)
    base_reqs = parse_pyproject_requirements(base_path)
    our_deps  = parse_pyproject_deps(ours_path)
    extra_deps = our_deps - base_deps

    #remove pyrfr from extra_deps since it is only needed for training and not inference
    extra_deps.discard("pyrfr")
    extra_deps.discard("smac")
    extra_deps.discard("dask")
    extra_deps.discard("wandb")
    extra_deps.discard("ema-pytorch")


    # Always include packages with pinned versions (e.g. torchcodec==0.9.0)
    extra_deps.update(PINNED_VERSIONS.keys())

    print(f"Base pyproject : {base_path.relative_to(REPO_ROOT)}")
    print(f"Our pyproject  : {ours_path.relative_to(REPO_ROOT)}")
    print(f"\nBase top-level deps  ({len(base_deps)}): {', '.join(sorted(base_deps))}")
    print(f"Our  top-level deps  ({len(our_deps)}): {', '.join(sorted(our_deps))}")
    print(f"\nExtra deps we added  ({len(extra_deps)}): {', '.join(sorted(extra_deps)) or '(none)'}")

    if not extra_deps:
        print("\nNo extra dependencies detected. Nothing to download.")
        return

    requested_specs = sorted(
        f"{dep}=={PINNED_VERSIONS[dep]}" if dep in PINNED_VERSIONS else dep
        for dep in extra_deps
    )

    try:
        resolved_versions = resolve_transitive_versions(requested_specs)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    keep_specs: list[str] = []
    skipped_specs: list[str] = []
    base_has_torch = "torch" in base_reqs
    torch_is_extra = "torch" in extra_deps

    for dep_name in sorted(resolved_versions):
        dep_version = resolved_versions[dep_name]

        if dep_name in extra_deps:
            keep_specs.append(f"{dep_name}=={dep_version}")
            continue

        base_req = base_reqs.get(dep_name)

        if base_has_torch and not torch_is_extra and is_runtime_managed_torch_stack(dep_name):
            skipped_specs.append(f"{dep_name}=={dep_version}")
            continue

        if base_req is None:
            keep_specs.append(f"{dep_name}=={dep_version}")
            continue

        if is_version_covered_by_base(base_req, dep_version):
            skipped_specs.append(f"{dep_name}=={dep_version}")
        else:
            # Runtime cannot satisfy this resolved version, so include it.
            keep_specs.append(f"{dep_name}=={dep_version}")

    print(f"\nResolved packages: {len(resolved_versions)}")
    print(f"Will download      : {len(keep_specs)}")
    print(f"Skipped (runtime)  : {len(skipped_specs)}")
    if skipped_specs:
        preview = ", ".join(sorted(skipped_specs)[:20])
        suffix = " ..." if len(skipped_specs) > 20 else ""
        print(f"Skipped examples   : {preview}{suffix}")

    if not keep_specs:
        print("\nAll resolved dependencies are already covered by runtime constraints.")
        return

    # ------------------------------------------------------------------ #
    # 2. Download filtered closure of the extra packages
    # ------------------------------------------------------------------ #
    temp_wheel_dir = wheel_dir / "tmp_wheels"
    temp_wheel_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pip", "download",
        "--no-deps",
        "--dest", str(temp_wheel_dir),
        *sorted(keep_specs),
    ]
    
    print(f"Downloading wheels to temporary folder...")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        sys.exit(1)

    # 3. Create the .tar.gz archive
    archive_path = wheel_dir / "dependencies.tar.gz"
    print(f"\nCompressing wheels into {archive_path.name}...")
    
    with tarfile.open(archive_path, "w:gz") as tar:
        for whl in temp_wheel_dir.glob("*.whl"):
            # Add to tar, stripping the leading 'tmp_wheels' path
            tar.add(whl, arcname=whl.name)
        for whl in temp_wheel_dir.glob("*.tar.gz"):
            tar.add(whl, arcname=whl.name)

    # 4. Clean up the temporary wheels
    print("Cleaning up temporary files...")
    for whl in temp_wheel_dir.glob("*.whl"):
        whl.unlink()
    for whl in temp_wheel_dir.glob("*.tar.gz"):
        whl.unlink()
    temp_wheel_dir.rmdir()

    print(f"\nSuccess! Created {archive_path} containing all offline dependencies.")


if __name__ == "__main__":
    main()
