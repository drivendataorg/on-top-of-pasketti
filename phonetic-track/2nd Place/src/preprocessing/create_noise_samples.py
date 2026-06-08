import io
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.get_json import _resolve_audio_path, get_data_from_json


def _preprocess_worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    sys.stdout = io.StringIO()


def _add_noise_one(args: tuple):
    clean_path, save_path, noise_path, snr_db, noise_start_frac, sample_rate, out_name = args
    try:
        clean, sr = librosa.load(str(clean_path), sr=sample_rate, mono=True)
        noise_raw, _ = librosa.load(str(noise_path), sr=sample_rate, mono=True)

        if len(clean) == 0:
            raise ValueError(f"Empty clean waveform: {clean_path}")
        if len(noise_raw) == 0:
            raise ValueError(f"Empty noise waveform: {noise_path}")

        if len(noise_raw) < len(clean):
            reps = int(np.ceil(len(clean) / len(noise_raw)))
            noise_raw = np.tile(noise_raw, reps)

        max_start = max(0, len(noise_raw) - len(clean))
        start = int(noise_start_frac * max_start)
        noise_seg = noise_raw[start : start + len(clean)]

        clean_rms = np.sqrt(np.mean(clean**2) + 1e-9)
        noise_rms = np.sqrt(np.mean(noise_seg**2) + 1e-9)
        target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
        scaled_noise = noise_seg * (target_noise_rms / (noise_rms + 1e-9))

        mixed = clean + scaled_noise
        peak = float(np.max(np.abs(mixed)))
        if peak > 1.0:
            mixed = mixed / peak

        out_path = save_path / out_name
        sf.write(str(out_path), mixed, sr)
        return out_name, sr, None
    except Exception as exc:
        return out_name, None, str(exc)


def _indexed_add_noise_one(args: tuple):
    local_idx, task = args
    return local_idx, _add_noise_one(task)


def _collect_noise_files(noise_dir: str) -> list[Path]:
    root = Path(noise_dir)
    exts = {".flac"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_noise(
    sample_rate: int,
    noise_dir: str,
    data_fraction: float,
    snr_min: float,
    snr_max: float = 20.0,
    create_noise_manifest: bool = True,
    seed: int = 42,
    num_workers: int | None = None,
    output_manifest_path: str | None = None,
    min_duration_sec: float = 0.0,
):
    save_path = Path("data/noise/audio")
    save_path.mkdir(parents=True, exist_ok=True)

    noise_files = _collect_noise_files(noise_dir)
    if not noise_files:
        raise FileNotFoundError(f"No noise files found under: {noise_dir}")

    cfg = type("Config", (), {})()
    cfg.data = type("DataConfig", (), {})()
    cfg.data.audio_folder = "data/audio"
    cfg.data.train_jsonl = "data/train_phon_transcripts.jsonl"
    cfg.data.train_jsonl_talkbank = "data/train_phon_transcripts_talkbank.jsonl"

    data = get_data_from_json(cfg, inference=False, pretraining=False)
    if min_duration_sec > 0:
        data = [entry for entry in data if entry.get("audio_duration_sec", 0.0) >= min_duration_sec]

    if not data:
        raise ValueError("No samples available after filtering.")

    rng = np.random.default_rng(seed)

    data_count = len(data)
    n_noisy = int(data_count * data_fraction)
    if data_fraction > 0 and n_noisy == 0:
        n_noisy = 1
    n_noisy = min(n_noisy, data_count)

    all_indices = np.arange(data_count)
    noisy_indices = set(rng.choice(all_indices, size=n_noisy, replace=False).tolist()) if n_noisy > 0 else set()

    subset_items = [(i, data[i]) for i in sorted(noisy_indices)]
    snr_values = rng.uniform(snr_min, snr_max, size=n_noisy) if n_noisy > 0 else []
    noise_choices = [noise_files[i] for i in rng.integers(0, len(noise_files), size=n_noisy)] if n_noisy > 0 else []
    noise_starts = rng.uniform(0.0, 1.0, size=n_noisy) if n_noisy > 0 else []

    if num_workers is None:
        num_workers = (os.cpu_count() or 1) - 2
    num_workers = max(1, int(num_workers))

    project_root = Path(__file__).resolve().parents[2]
    audio_dir = Path(cfg.data.audio_folder).resolve()

    logger.info(
        "Adding noise to {} / {} files using {} workers (SNR {:.1f}-{:.1f} dB, seed {})",
        n_noisy,
        data_count,
        num_workers,
        snr_min,
        snr_max,
        seed,
    )

    tasks = [
        (
            _resolve_audio_path(entry.get("audio_path", ""), audio_dir, project_root),
            save_path,
            Path(noise_path),
            float(snr),
            float(start),
            sample_rate,
            f"{entry['utterance_id']}.flac",
        )
        for (_, entry), noise_path, snr, start in zip(subset_items, noise_choices, snr_values, noise_starts)
    ]

    processed: dict[int, str] = {}
    if num_workers == 1:
        for (orig_idx, _), task in tqdm(zip(subset_items, tasks), total=n_noisy):
            out_name, _, err = _add_noise_one(task)
            if err is None:
                processed[orig_idx] = out_name
            else:
                logger.error("Error processing {}: {}", task[0], err)
    else:
        indexed_tasks = list(enumerate(tasks))
        with mp.get_context("spawn").Pool(
            processes=num_workers,
            initializer=_preprocess_worker_init,
        ) as pool:
            for local_idx, (out_name, _, err) in tqdm(
                pool.imap_unordered(_indexed_add_noise_one, indexed_tasks),
                total=n_noisy,
                leave=True,
                position=0,
            ):
                orig_idx = subset_items[local_idx][0]
                if err is None:
                    processed[orig_idx] = out_name
                else:
                    logger.error("Error processing {}: {}", out_name, err)

    output_manifest = Path(output_manifest_path) if output_manifest_path else Path("data/train_phon_transcripts_noise.jsonl")

    full_rows: list[dict] = []
    noisy_rows: list[dict] = []
    for i, item in enumerate(data):
        row = dict(item)
        if i in processed:
            row["audio_path"] = str(Path("noise/audio") / processed[i])
            row["noise_added"] = True
            noisy_rows.append(row)
        else:
            row["noise_added"] = False
        full_rows.append(row)

    _write_jsonl(output_manifest, full_rows)
    logger.info("Created full noisy manifest at {} ({} rows)", output_manifest, len(full_rows))

    if create_noise_manifest:
        noise_only_manifest = output_manifest.with_name(f"{output_manifest.stem}_only{output_manifest.suffix}")
        _write_jsonl(noise_only_manifest, noisy_rows)
        logger.info("Created noise-only manifest at {} ({} rows)", noise_only_manifest, len(noisy_rows))

    return output_manifest


if __name__ == "__main__":
    sample_rate = 16000
    noise_dir = "data/audio"
    data_fraction = 0.05
    snr_min = 1.0
    snr_max = 3.0
    num_workers = 16

    add_noise(
        sample_rate=sample_rate,
        noise_dir=noise_dir,
        data_fraction=data_fraction,
        snr_min=snr_min,
        snr_max=snr_max,
        create_noise_manifest=True,
        seed=42,
        num_workers=num_workers,
        output_manifest_path="data/train_phon_transcripts_noise.jsonl",
        min_duration_sec=0.0,
    )
