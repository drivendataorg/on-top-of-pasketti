#!/usr/bin/env python3
"""
audio_cleaner.py
────────────────────────────────────────────────────────────────────────────────
DeepFilterNet audio cleaning pipeline.

Stages
  1) suppress_extreme_spikes()  -> reduces impulsive events (pen drops, thumps)
  2) denoise_deepfilternet()    -> aggressively isolates main speaker, removes noise/reverb
  3) normalize_loudness()       -> aligns loudness for downstream ASR

Run
  uv run audio_cleaner.py
      Processes ./audio -> ./audio_normalized (batch mode)

  uv run audio_cleaner.py input.wav output.wav
      Processes a single file
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
import sys
import torchaudio
from types import ModuleType

# --- PATCH FOR DEEPFILTERNET ---
# Trick DeepFilterNet into finding the AudioMetaData class in the new Torchaudio
if "torchaudio.backend.common" not in sys.modules:
    mock_backend = ModuleType("torchaudio.backend")
    mock_common = ModuleType("torchaudio.backend.common")
    mock_common.AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
    
    mock_backend.common = mock_common
    sys.modules["torchaudio.backend"] = mock_backend
    sys.modules["torchaudio.backend.common"] = mock_common
# -------------------------------

from df.enhance import enhance, init_df

# ----------------------------- Config -----------------------------
data_dir = Path(__file__).resolve().parent.parent.parent / "data"
INPUT_DIR = data_dir / "audio"
OUTPUT_DIR = data_dir / "audio_normalized"
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3", ".m4a"}

# Manifest files that define the relevant utterance set
MANIFEST_FILES = [
    data_dir / "train_phon_transcripts.jsonl",
    data_dir / "train_phon_transcripts_additional.jsonl",
]

# Parallel workers for batch processing.
# Each worker loads its own copy of the DeepFilterNet model, so balance
# CPU count against available RAM. Override with the AUDIO_CLEANER_WORKERS
# environment variable if needed.
_DEFAULT_WORKERS = os.cpu_count() - 1
NUM_WORKERS = int(os.getenv("AUDIO_CLEANER_WORKERS", _DEFAULT_WORKERS))

# Sample rates
PIPELINE_SAMPLE_RATE = 48000 # DeepFilterNet native rate
TARGET_SAMPLE_RATE = 16000   # Final output rate for downstream ASR

# Stage toggles
RUN_SPIKE_SUPPRESSOR = True
RUN_DENOISER = True
RUN_NORMALIZE = True

# Stage 1: spike suppressor
SPIKE_CEILING_PERCENTILE = 99.8   # lower = more aggressive
SPIKE_GAIN_FLOOR = 0.12           # never attenuate below this gain
SPIKE_ATTACK_MS = 4.0
SPIKE_RELEASE_MS = 70.0

# Stage 3: loudness
TARGET_DBFS = -20.0
HEADROOM_DB = 1.0

LOG_LEVEL = logging.INFO

# Global DeepFilterNet variables
DF_MODEL = None
DF_STATE = None
DF_MODEL_NAME = os.getenv("DF_MODEL_NAME", "DeepFilterNet2") # this is beacuse we are going to need to acess oflline maybe rn am doing online
DF_MODEL_BASE_DIR = os.getenv("DF_MODEL_BASE_DIR")
DF2_LOCAL_DIR = Path(__file__).resolve().parent / "DeepFilterNet2"

# ----------------------------- Utilities -----------------------------

def iter_audio_files(root: Path, extensions: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def load_manifest_paths(
    manifest_files: list[Path] = MANIFEST_FILES,
    audio_dir: Path = INPUT_DIR,
) -> list[Path]:
    """
    Return the list of audio paths that are both referenced in the manifest
    files *and* actually present on disk.

    Deduplication is done by utterance_id so that files appearing in both
    manifests are only counted once.
    """
    seen_ids: set[str] = set()
    wanted: list[Path] = []

    for manifest in manifest_files:
        if not manifest.exists():
            logging.getLogger("audio_cleaner").warning(
                "Manifest not found, skipping: %s", manifest
            )
            continue
        with manifest.open("r") as fh:
            for line in fh:
                item = json.loads(line)
                uid = item["utterance_id"]
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                # audio_path is relative (e.g. "audio/U_xxx.flac"); resolve against data_dir
                rel = Path(item["audio_path"])
                abs_path = audio_dir.parent / rel  # data_dir / audio/U_xxx.flac
                if abs_path.exists():
                    wanted.append(abs_path)

    return wanted


def _smooth_gain_envelope(gain: np.ndarray, attack_samples: int, release_samples: int) -> np.ndarray:
    """One-pole smoothing with separate attack/release behavior."""
    out = np.ones_like(gain, dtype=np.float32)
    attack_coeff = np.exp(-1.0 / max(attack_samples, 1))
    release_coeff = np.exp(-1.0 / max(release_samples, 1))

    prev = 1.0
    for i in range(len(gain)):
        target = float(gain[i])
        coeff = attack_coeff if target < prev else release_coeff
        prev = coeff * prev + (1.0 - coeff) * target
        out[i] = prev
    return out

def _init_deepfilter():
    """Initialize the DeepFilterNet model globally to avoid reloading per file."""
    global DF_MODEL, DF_STATE
    if DF_MODEL is None:
        model_base_dir = DF_MODEL_BASE_DIR
        if not model_base_dir and DF2_LOCAL_DIR.is_dir():
            model_base_dir = str(DF2_LOCAL_DIR)

        # Prefer DF2 by default and allow overriding with env vars:
        #   DF_MODEL_NAME (e.g. DeepFilterNet2), DF_MODEL_BASE_DIR (/path/to/model)
        DF_MODEL, DF_STATE, _ = init_df(
            model_base_dir=model_base_dir,
            default_model=DF_MODEL_NAME,
        )

# ----------------------------- Stage 1 -----------------------------

def suppress_extreme_spikes(
    waveform: np.ndarray,
    sample_rate: int,
    ceiling_percentile: float = SPIKE_CEILING_PERCENTILE,
    gain_floor: float = SPIKE_GAIN_FLOOR,
    attack_ms: float = SPIKE_ATTACK_MS,
    release_ms: float = SPIKE_RELEASE_MS,
) -> np.ndarray:
    """
    Transient suppressor for impulsive events.
    Uses amplitude percentile to estimate a robust ceiling and applies a smooth
    dynamic gain envelope where the signal exceeds that ceiling.
    """
    if waveform.size == 0:
        return waveform

    if waveform.ndim == 2:
        mono = np.mean(waveform, axis=1)
    else:
        mono = waveform

    abs_mono = np.abs(mono)
    ceiling = float(np.percentile(abs_mono, ceiling_percentile))
    ceiling = max(ceiling, 1e-6)

    raw_gain = np.where(abs_mono > ceiling, ceiling / np.maximum(abs_mono, 1e-9), 1.0)
    raw_gain = np.maximum(raw_gain, gain_floor).astype(np.float32)

    attack_samples = int((attack_ms / 1000.0) * sample_rate)
    release_samples = int((release_ms / 1000.0) * sample_rate)
    smooth_gain = _smooth_gain_envelope(raw_gain, attack_samples, release_samples)

    out = waveform.astype(np.float32, copy=True)
    if out.ndim == 2:
        out *= smooth_gain[:, np.newaxis]
    else:
        out *= smooth_gain
    return out

# ----------------------------- Stage 2 -----------------------------

def denoise_deepfilternet(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    DeepFilterNet voice isolation. Removes background noise and secondary voices.
    """
    if waveform.size == 0:
        return waveform

    _init_deepfilter()

    # DeepFilterNet expects shape (channels, time) as float32
    if waveform.ndim == 1:
        data = waveform[np.newaxis, :]  # (1, time)
        squeeze = True
    else:
        data = waveform.T  # (channels, time)
        squeeze = False

    # Mix to mono once, then expand back to original channel count later if needed
    mono = np.mean(data, axis=0, keepdims=True, dtype=np.float32)
    audio_tensor = torch.from_numpy(mono).float()

    # Resample if necessary (should be bypassed since we feed 48kHz now)
    df_sr = DF_STATE.sr()
    if sample_rate != df_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=df_sr
        )

    # Clear frame state before each file to avoid cross-file leakage.
    if hasattr(DF_STATE, "reset"):
        DF_STATE.reset()

    enhanced_mono = enhance(DF_MODEL, DF_STATE, audio_tensor)

    # Resample back to pipeline sample rate (should be bypassed)
    if sample_rate != df_sr:
        enhanced_mono = torchaudio.functional.resample(
            enhanced_mono, orig_freq=df_sr, new_freq=sample_rate
        )

    # Keep exact output length
    target_len = data.shape[1]
    current_len = enhanced_mono.shape[1]
    if current_len > target_len:
        enhanced_mono = enhanced_mono[:, :target_len]
    elif current_len < target_len:
        pad = target_len - current_len
        enhanced_mono = torch.nn.functional.pad(enhanced_mono, (0, pad))

    if data.shape[0] > 1:
        enhanced_tensor = enhanced_mono.repeat(data.shape[0], 1)
    else:
        enhanced_tensor = enhanced_mono

    enhanced_np = enhanced_tensor.cpu().numpy()

    if squeeze:
        return enhanced_np[0]
    else:
        return enhanced_np.T  # back to (time, channels)

# ----------------------------- Stage 3 -----------------------------

def normalize_loudness(
    waveform: np.ndarray,
    target_dbfs: float = TARGET_DBFS,
    headroom_db: float = HEADROOM_DB,
) -> np.ndarray:
    """
    RMS normalization with peak safety headroom.
    """
    if waveform.size == 0:
        return waveform

    x = waveform.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    if rms < 1e-9:
        return x

    current_dbfs = 20.0 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10.0 ** (gain_db / 20.0)

    peak = np.max(np.abs(x))
    max_gain = (10.0 ** (-headroom_db / 20.0)) / max(float(peak), 1e-9)
    gain_linear = min(gain_linear, max_gain)
    return (x * gain_linear).astype(np.float32)

# ----------------------------- Public API -----------------------------

def preprocess_audio_array(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    run_spike_suppressor: bool = RUN_SPIKE_SUPPRESSOR,
    run_denoiser: bool = RUN_DENOISER,
    run_normalize: bool = RUN_NORMALIZE,
    target_dbfs: float = TARGET_DBFS,
) -> np.ndarray:
    """
    Full preprocessing pipeline on a waveform array.
    """
    audio = waveform.astype(np.float32, copy=False)

    # 1. Catch the spikes before doing ANY volume boosting
    if run_spike_suppressor:
        audio = suppress_extreme_spikes(audio, sample_rate)

    # 2. Make it loud so the Neural Network can hear it
    if run_normalize:
        # Boost it to the target level first
        audio = normalize_loudness(audio, target_dbfs=target_dbfs)

    # 3. Apply DeepFilterNet to the now-audible speech
    if run_denoiser:
        audio = denoise_deepfilternet(audio, sample_rate)

        # 4. Final Polish: DeepFilterNet removing noise lowers the overall volume.
        # Run normalization ONE MORE TIME to ensure the clean speech hits exactly -20 dBFS.
        if run_normalize:
            audio = normalize_loudness(audio, target_dbfs=target_dbfs)

    return audio

def preprocess_audio_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    **kwargs,
) -> np.ndarray:
    """
    Load audio, force to mono & 48kHz for processing, run pipeline, then export at 16kHz.
    """
    input_path = Path(input_path)
    waveform, sample_rate = sf.read(str(input_path), always_2d=True, dtype="float32")

    # Force Mono mixdown BEFORE pipeline to save processing overhead
    if waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1, keepdims=True)

    # Force to 48kHz for the pipeline
    if sample_rate != PIPELINE_SAMPLE_RATE:
        wav_tensor = torch.from_numpy(waveform.T)
        wav_tensor = torchaudio.functional.resample(
            wav_tensor, orig_freq=sample_rate, new_freq=PIPELINE_SAMPLE_RATE
        )
        waveform = wav_tensor.T.numpy()
        sample_rate = PIPELINE_SAMPLE_RATE

    # Proceed with the pipeline at 48kHz
    cleaned = preprocess_audio_array(waveform, sample_rate, **kwargs)

    # Force to 16kHz at the very end
    if sample_rate != TARGET_SAMPLE_RATE:
        wav_tensor = torch.from_numpy(cleaned.T)
        wav_tensor = torchaudio.functional.resample(
            wav_tensor, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        cleaned = wav_tensor.T.numpy()
        sample_rate = TARGET_SAMPLE_RATE

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), cleaned, sample_rate)

    return cleaned

# ----------------------------- Parallel worker helpers -----------------------------
# These must be module-level functions so ProcessPoolExecutor can pickle them.

def _worker_init() -> None:
    """Called once per worker process at startup — loads the DeepFilterNet model."""
    _init_deepfilter()


def _process_one(args: tuple[Path, Path]) -> tuple[bool, str]:
    """Process a single (src, dst) pair.  Returns (ok, error_message)."""
    src, dst = args
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        preprocess_audio_file(src, dst)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ----------------------------- Batch runner -----------------------------

def run_batch(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    num_workers: int = NUM_WORKERS,
) -> tuple[int, int]:
    logger = logging.getLogger("audio_cleaner")

    # Use only files referenced in the manifests that exist on disk
    files = load_manifest_paths(audio_dir=input_dir)
    if not files:
        logger.warning(
            "No matching audio files found (checked manifests against %s)", input_dir
        )
        return 0, 0

    logger.info(
        "Processing %d files with %d worker(s) -> %s",
        len(files),
        num_workers,
        output_dir,
    )

    task_args = [
        (src, output_dir / src.relative_to(input_dir))
        for src in files
    ]

    success = 0
    failed = 0

    if num_workers <= 1:
        # Single-process fallback (easier to debug / profile)
        _worker_init()
        for args in tqdm(task_args, desc="Cleaning"):
            ok, err = _process_one(args)
            if ok:
                success += 1
            else:
                failed += 1
                logger.error("FAIL %s | %s", args[0].name, err)
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
        ) as pool:
            futures = {pool.submit(_process_one, a): a[0] for a in task_args}
            with tqdm(total=len(futures), desc="Cleaning") as pbar:
                for fut in as_completed(futures):
                    src_path = futures[fut]
                    try:
                        ok, err = fut.result()
                    except Exception as exc:
                        ok, err = False, str(exc)
                    if ok:
                        success += 1
                    else:
                        failed += 1
                        logger.error("FAIL %s | %s", src_path.name, err)
                    pbar.update(1)

    return success, failed

def audio_cleaner() -> int:
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("audio_cleaner")

    import sys

    # Single-file mode: audio_cleaner.py input.wav [output.wav]
    if len(sys.argv) >= 2:
        input_file = Path(sys.argv[1])
        output_file = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_file.with_name(f"{input_file.stem}_cleaned.wav")
        logger.info("Single-file mode: %s -> %s", input_file, output_file)
        preprocess_audio_file(input_file, output_file)
        logger.info("Done.")
        return 0

    # Default: batch mode
    if not INPUT_DIR.exists():
        logger.error("Input directory does not exist: %s", INPUT_DIR)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Batch mode: %s -> %s", INPUT_DIR, OUTPUT_DIR)
    logger.info(
        "Stages: spike=%s denoise(deepfilternet)=%s normalize=%s target=%.1f dBFS",
        RUN_SPIKE_SUPPRESSOR,
        RUN_DENOISER,
        RUN_NORMALIZE,
        TARGET_DBFS,
    )
    logger.info("Workers: %d (set AUDIO_CLEANER_WORKERS env var to override)", NUM_WORKERS)
    logger.info(
        "Manifests: %s",
        ", ".join(str(m) for m in MANIFEST_FILES),
    )

    success, failed = run_batch(INPUT_DIR, OUTPUT_DIR)
    logger.info("Done | success=%d | failed=%d", success, failed)
    return 0 if success > 0 else 1

if __name__ == "__main__":
    raise SystemExit(audio_cleaner())