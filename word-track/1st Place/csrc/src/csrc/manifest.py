import json
from pathlib import Path


def load_manifest(manifest_path: Path) -> list[dict]:
    """JSONL manifest を読み込む。"""
    entries = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


def write_manifest(entries: list[dict], path: Path) -> Path:
    """JSONL manifest を書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path
