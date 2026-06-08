import json
from pathlib import Path
import os

def _resolve_audio_path(audio_path: str, default_audio_dir: Path, project_root: Path) -> Path:
    """Resolve audio file paths across competition and external dataset layouts."""
    rel_path = Path(audio_path)
    if rel_path.is_absolute() and rel_path.exists():
        return rel_path

    candidates = [
        default_audio_dir / rel_path.name,  # legacy behavior: data/audio/<basename>
        default_audio_dir / rel_path,       # supports nested relative paths
        project_root / "data" / rel_path,  # external sets: data/audio_*/*.flac
        project_root / rel_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_data_from_json(cfg, inference=False, pretraining=False, jsonl_paths=None):
    """Loads JSON data but strictly filters out entries with missing audio files."""
    if jsonl_paths is None:
        if inference:
            jsonl_paths = [cfg.data.test_jsonl]
        else:
            if cfg.data.train_jsonl_talkbank is not None:
                jsonl_paths = [cfg.data.train_jsonl, cfg.data.train_jsonl_talkbank]
            else:
                jsonl_paths = [cfg.data.train_jsonl]
    
    # Use absolute path to guarantee we are checking the same places the Dataset looks.
    audio_dir = Path(cfg.data.audio_folder).resolve()
    project_root = Path(__file__).resolve().parents[2]
    
    all_data = []
    missing_count = 0
    
    for path in jsonl_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())

                if entry.get("utterance_id") == "U_b8a4e8220e65219b":
                    continue  # Skip this known bad entry
                
                full_audio_path = _resolve_audio_path(
                    audio_path=entry.get("audio_path", ""),
                    default_audio_dir=audio_dir,
                    project_root=project_root,
                )
                
                # Check if it ACTUALLY exists
                if full_audio_path.exists():
                    all_data.append(entry)
                else:
                    missing_count += 1

    print(f"Loaded {len(all_data)} valid utterances.")
    if missing_count > 0:
        print(f"Skipped {missing_count} missing audio files.")
    
    # Enforce the expected training-set size, but allow smaller inference/test inputs.
    if not pretraining and not inference:
        assert len(all_data) == 153066, f"Expected 153066 utterances, but got {len(all_data)}"
    return all_data
