#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from dataclasses import dataclass
import numpy as np
import pandas as pd
import soundfile as sf

# --- Default configuration ---
METRICS_CSV_PATH = "metrics_results.csv"
RAW_AUDIO_DIR = Path("./audio")
DEFAULT_CLEANED_AUDIO_DIR = Path("./audio_normalized")
DEFAULT_REJECTED_AUDIO_DIR = Path("./audio_rejected")
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".aif", ".aiff"}

EPS = 1e-12
SPEECH_BAND_HZ = (300, 4000)
NOISE_LOW_BAND_HZ = (0, 120)
NOISE_HIGH_BAND_HZ = (4500, 8000)
CHILD_F0_HZ = (180, 500)

ISOLATION_SCORE_WEIGHTS = {
    "speech_to_noise_db": 0.50,
    "speech_band_ratio": 0.30,
    "child_pitch_ratio": 0.20,
}

DELTA_COLUMNS_NEEDED = {"delta_speech_to_noise_db", "delta_child_isolation_score"}

def _resolve_metrics_path(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path)
    env_path = os.getenv("METRICS_CSV_PATH")
    if env_path:
        return Path(env_path)
    return Path(METRICS_CSV_PATH)


def _resolve_file_column(df: pd.DataFrame) -> str:
    for col in ("relative_path", "filename"):
        if col in df.columns:
            return col
    raise KeyError("CSV must contain either 'relative_path' or 'filename'.")


def build_audio_pair_index(raw_dir: Path, cleaned_dir: Path, valid_ext: set[str]) -> pd.DataFrame:
    raw_map: dict[str, Path] = {}
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_ext:
            raw_map[path.relative_to(raw_dir).as_posix()] = path

    cleaned_map: dict[str, Path] = {}
    for path in cleaned_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_ext:
            cleaned_map[path.relative_to(cleaned_dir).as_posix()] = path

    rows = []
    all_keys = sorted(set(raw_map.keys()) | set(cleaned_map.keys()))
    for key in all_keys:
        raw_path = raw_map.get(key)
        cleaned_path = cleaned_map.get(key)
        rows.append(
            {
                "relative_path": key,
                "raw_path": str(raw_path) if raw_path is not None else None,
                "cleaned_path": str(cleaned_path) if cleaned_path is not None else None,
                "raw_exists": raw_path is not None,
                "cleaned_exists": cleaned_path is not None,
            }
        )
    return pd.DataFrame(rows)


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1)
    return mono.astype(np.float32), int(sr)


def maybe_resample_linear(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return signal
    if len(signal) <= 1:
        return signal.astype(np.float32)
    duration_sec = len(signal) / float(source_sr)
    target_len = int(round(duration_sec * target_sr))
    if target_len <= 1:
        return signal.astype(np.float32)
    t_src = np.linspace(0.0, duration_sec, num=len(signal), endpoint=False)
    t_tgt = np.linspace(0.0, duration_sec, num=target_len, endpoint=False)
    return np.interp(t_tgt, t_src, signal).astype(np.float32)


def rms_dbfs(signal: np.ndarray, eps: float = EPS) -> float:
    rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))
    return float(20.0 * np.log10(max(rms, eps)))


def peak_dbfs(signal: np.ndarray, eps: float = EPS) -> float:
    peak = float(np.max(np.abs(signal)))
    return float(20.0 * np.log10(max(peak, eps)))


def average_power_spectrum(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=np.float32)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size), mode="constant")

    remainder = (x.size - n_fft) % hop
    pad_end = (hop - remainder) % hop
    if pad_end > 0:
        x = np.pad(x, (0, pad_end), mode="constant")

    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (x.size - n_fft) // hop

    power_accum = np.zeros((n_fft // 2 + 1,), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + n_fft] * window
        spec = np.fft.rfft(frame)
        power_accum += np.abs(spec) ** 2

    power_mean = power_accum / max(n_frames, 1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return freqs.astype(np.float32), power_mean.astype(np.float64)


def _band_mask(freqs: np.ndarray, band: tuple[int, int]) -> np.ndarray:
    lo, hi = band
    return (freqs >= lo) & (freqs < hi)


def spectral_band_ratios(freqs: np.ndarray, power: np.ndarray) -> dict[str, float]:
    low = (freqs >= 0) & (freqs < 300)
    mid = (freqs >= 300) & (freqs < 3000)
    high = freqs >= 3000

    total = float(power.sum()) + EPS
    return {
        "low_ratio": float(power[low].sum() / total),
        "mid_ratio": float(power[mid].sum() / total),
        "high_ratio": float(power[high].sum() / total),
    }


def spectral_centroid(freqs: np.ndarray, power: np.ndarray) -> float:
    denom = float(power.sum()) + EPS
    return float(np.sum(freqs * power) / denom)


def speech_noise_proxy_metrics(freqs: np.ndarray, power: np.ndarray) -> dict[str, float]:
    speech_mask = _band_mask(freqs, SPEECH_BAND_HZ)
    noise_mask = _band_mask(freqs, NOISE_LOW_BAND_HZ) | _band_mask(freqs, NOISE_HIGH_BAND_HZ)

    speech_energy = float(power[speech_mask].sum())
    noise_energy = float(power[noise_mask].sum())
    total_energy = float(power.sum()) + EPS

    return {
        "speech_band_ratio": speech_energy / total_energy,
        "noise_band_ratio": noise_energy / total_energy,
        "speech_to_noise_db": float(10.0 * np.log10((speech_energy + EPS) / (noise_energy + EPS))),
    }


def pitch_proxy_metrics(signal: np.ndarray, sr: int, *, compute_pitch: bool) -> dict[str, float]:
    if not compute_pitch or len(signal) < 2048:
        return {"child_pitch_ratio": np.nan, "voiced_ratio": np.nan, "median_f0_hz": np.nan}

    try:
        import librosa
    except Exception:
        return {"child_pitch_ratio": np.nan, "voiced_ratio": np.nan, "median_f0_hz": np.nan}

    fmin, fmax = CHILD_F0_HZ
    pyin_fmax = min(float(sr) / 2.0 - 1.0, fmax * 1.3)

    try:
        f0, voiced_flag, _ = librosa.pyin(
            signal.astype(np.float64),
            sr=sr,
            fmin=fmin,
            fmax=pyin_fmax,
            frame_length=2048,
            hop_length=256,
        )
    except Exception:
        return {"child_pitch_ratio": np.nan, "voiced_ratio": np.nan, "median_f0_hz": np.nan}

    valid = np.isfinite(f0)
    if voiced_flag is not None:
        voiced_ratio = float(np.mean(voiced_flag.astype(np.float32)))
    else:
        voiced_ratio = float(np.mean(valid.astype(np.float32)))

    if valid.any():
        voiced_f0 = f0[valid]
        child_pitch_ratio = float(np.mean((voiced_f0 >= fmin) & (voiced_f0 <= fmax)))
        median_f0_hz = float(np.median(voiced_f0))
    else:
        child_pitch_ratio = np.nan
        median_f0_hz = np.nan

    return {
        "child_pitch_ratio": child_pitch_ratio,
        "voiced_ratio": voiced_ratio,
        "median_f0_hz": median_f0_hz,
    }


def child_voice_isolation_score(metrics: dict[str, float]) -> float:
    snr_scaled = np.clip((metrics["speech_to_noise_db"] + 20.0) / 40.0, 0.0, 1.0)
    speech_ratio_scaled = np.clip((metrics["speech_band_ratio"] - 0.20) / 0.60, 0.0, 1.0)

    child_ratio = metrics.get("child_pitch_ratio", np.nan)
    if np.isnan(child_ratio):
        child_ratio_scaled = 0.50
    else:
        child_ratio_scaled = float(np.clip(child_ratio, 0.0, 1.0))

    score = (
        ISOLATION_SCORE_WEIGHTS["speech_to_noise_db"] * snr_scaled
        + ISOLATION_SCORE_WEIGHTS["speech_band_ratio"] * speech_ratio_scaled
        + ISOLATION_SCORE_WEIGHTS["child_pitch_ratio"] * child_ratio_scaled
    )
    return float(score * 100.0)


def compute_audio_metrics(signal: np.ndarray, sr: int, *, compute_pitch: bool) -> dict[str, float]:
    freqs, power = average_power_spectrum(signal, sr=sr, n_fft=2048, hop=512)
    bands = spectral_band_ratios(freqs, power)
    isolation = speech_noise_proxy_metrics(freqs, power)
    pitch = pitch_proxy_metrics(signal, sr, compute_pitch=compute_pitch)

    metrics: dict[str, float] = {
        "sample_rate": float(sr),
        "duration_sec": float(len(signal) / sr),
        "rms_dbfs": rms_dbfs(signal),
        "peak_dbfs": peak_dbfs(signal),
        "spectral_centroid_hz": spectral_centroid(freqs, power),
        "low_ratio": bands["low_ratio"],
        "mid_ratio": bands["mid_ratio"],
        "high_ratio": bands["high_ratio"],
        "speech_band_ratio": isolation["speech_band_ratio"],
        "noise_band_ratio": isolation["noise_band_ratio"],
        "speech_to_noise_db": isolation["speech_to_noise_db"],
        "child_pitch_ratio": pitch["child_pitch_ratio"],
        "voiced_ratio": pitch["voiced_ratio"],
        "median_f0_hz": pitch["median_f0_hz"],
    }
    metrics["child_isolation_score"] = child_voice_isolation_score(metrics)
    return metrics


def generate_metrics_csv(
    metrics_path: Path,
    raw_dir: Path,
    cleaned_dir: Path,
    *,
    compute_pitch: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw audio directory not found: {raw_dir}")
    if not cleaned_dir.exists():
        raise FileNotFoundError(f"Cleaned audio directory not found: {cleaned_dir}")

    pairs = build_audio_pair_index(raw_dir, cleaned_dir, SUPPORTED_EXTENSIONS)
    matched = pairs[pairs["raw_exists"] & pairs["cleaned_exists"]].copy()
    if matched.empty:
        raise RuntimeError("No matched raw/cleaned audio pairs were found.")

    if limit is not None and limit > 0 and len(matched) > limit:
        matched = matched.sample(limit, random_state=42).sort_values("relative_path")

    rows: list[dict[str, float | str]] = []
    for row in matched.itertuples(index=False):
        raw_path = Path(row.raw_path)
        cleaned_path = Path(row.cleaned_path)

        raw_sig, raw_sr = load_audio_mono(raw_path)
        cleaned_sig, cleaned_sr = load_audio_mono(cleaned_path)

        if cleaned_sr != raw_sr:
            cleaned_sig = maybe_resample_linear(cleaned_sig, source_sr=cleaned_sr, target_sr=raw_sr)
            cleaned_sr = raw_sr

        n = min(len(raw_sig), len(cleaned_sig))
        raw_sig = raw_sig[:n]
        cleaned_sig = cleaned_sig[:n]

        raw_metrics = compute_audio_metrics(raw_sig, raw_sr, compute_pitch=compute_pitch)
        cleaned_metrics = compute_audio_metrics(cleaned_sig, cleaned_sr, compute_pitch=compute_pitch)

        output_row: dict[str, float | str] = {
            "relative_path": row.relative_path,
            "raw_path": str(raw_path),
            "cleaned_path": str(cleaned_path),
        }

        metric_keys = sorted(set(raw_metrics.keys()) | set(cleaned_metrics.keys()))
        for key in metric_keys:
            raw_val = float(raw_metrics.get(key, np.nan))
            cleaned_val = float(cleaned_metrics.get(key, np.nan))
            output_row[f"raw_{key}"] = raw_val
            output_row[f"normalized_{key}"] = cleaned_val
            output_row[f"delta_{key}"] = cleaned_val - raw_val

        rows.append(output_row)

    out_df = pd.DataFrame(rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(metrics_path, index=False)
    return out_df


def _validate_metrics_columns(df: pd.DataFrame) -> None:
    missing = sorted(DELTA_COLUMNS_NEEDED - set(df.columns))
    if missing:
        raise KeyError(f"Metrics CSV is missing required columns: {missing}")

@dataclass
class FilterAudioConfig():
    metrics_csv: Optional[str] = None
    raw_dir: str = str(RAW_AUDIO_DIR)
    cleaned_dir: str = str(DEFAULT_CLEANED_AUDIO_DIR)
    rejected_dir: str = str(DEFAULT_REJECTED_AUDIO_DIR)
    recompute_metrics: bool = False
    build_metrics_only: bool = False
    skip_pitch: bool = False
    limit: Optional[int] = None

def filter_audio(args = FilterAudioConfig()):
    cleaned_audio_dir = Path(args.cleaned_dir)
    rejected_audio_dir = Path(args.rejected_dir)

    # 1. Load (or create) metrics CSV
    metrics_path = _resolve_metrics_path(args.metrics_csv)
    needs_build = args.recompute_metrics or not metrics_path.exists()
    if needs_build:
        print(f"Generating metrics CSV at: {metrics_path}")
        df = generate_metrics_csv(
            metrics_path=metrics_path,
            raw_dir=Path(args.raw_dir),
            cleaned_dir=cleaned_audio_dir,
            compute_pitch=not args.skip_pitch,
            limit=args.limit,
        )
        print(f"Saved metrics rows: {len(df)}")

    df = pd.read_csv(metrics_path)
    _validate_metrics_columns(df)
    file_col = _resolve_file_column(df)

    if args.build_metrics_only:
        print("Build-only mode: metrics CSV is ready. Exiting without filtering/moving files.")
        return
    
    # 2. Define the quality gate rules
    # Rule 1: Speech-to-Noise didn't drop by more than 2dB
    good_snr = df["delta_speech_to_noise_db"] >= -2.0
    
    # Rule 2: Child Isolation Score didn't drop below 0
    good_isolation = df["delta_child_isolation_score"] >= 0.0
    
    # 3. Separate the good from the bad
    rejected_files_df = df[~(good_snr & good_isolation)]
    
    print(f"Metrics CSV: {metrics_path}")
    print(f"Path column: {file_col}")
    print(f"Total files evaluated: {len(df)}")
    print(f"Files passed: {len(df) - len(rejected_files_df)}")
    print(f"Files rejected: {len(rejected_files_df)}")
    
    # 4. Move the rejected physical files out of the clean folder
    if not rejected_files_df.empty:
        rejected_audio_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        for relative_file in rejected_files_df[file_col]:
            source_path = cleaned_audio_dir / str(relative_file)
            dest_path = rejected_audio_dir / str(relative_file)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if source_path.exists():
                shutil.move(str(source_path), str(dest_path))
                moved_count += 1
                
        print(f"Successfully moved {moved_count} bad files to {rejected_audio_dir}")

def main():
    parser = argparse.ArgumentParser(description="Move rejected cleaned audio files based on metric deltas.")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Path to metrics CSV. Falls back to $METRICS_CSV_PATH then METRICS_CSV_PATH constant.",
    )
    parser.add_argument("--raw-dir", default=str(RAW_AUDIO_DIR), help="Directory of original raw audio.")
    parser.add_argument(
        "--cleaned-dir",
        default=str(DEFAULT_CLEANED_AUDIO_DIR),
        help="Directory of cleaned/normalized audio to evaluate and filter.",
    )
    parser.add_argument(
        "--rejected-dir",
        default=str(DEFAULT_REJECTED_AUDIO_DIR),
        help="Directory where rejected cleaned files are moved.",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Recompute metrics CSV even if it already exists.",
    )
    parser.add_argument(
        "--build-metrics-only",
        action="store_true",
        help="Create/validate metrics CSV and exit without moving any audio files.",
    )
    parser.add_argument(
        "--skip-pitch",
        action="store_true",
        help="Skip pitch proxy metric to speed up CSV generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of matched files to evaluate when creating metrics.",
    )
    args = parser.parse_args()

    filter_audio(args)

    

if __name__ == "__main__":
    main()
