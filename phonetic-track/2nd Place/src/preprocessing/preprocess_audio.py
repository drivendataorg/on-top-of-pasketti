"""
Offline audio preprocessing script.

Reads all JSONL metadata files used by the training pipeline, applies the
same audio processing steps that would happen in __getitem__ (mono conversion,
resampling, amplitude normalisation), and saves each waveform as a float32
NumPy array under data/audio_processed/<utterance_id>.npy.

Once this has been run once, Wav2Vec2Dataset can be initialised with
  cache_dir = "data/audio_processed"
and will load everything into RAM at startup instead of hitting disk on every
__getitem__ call.

Usage (from repo root):
    uv run python -m src.preprocessing.preprocess_audio
    uv run python -m src.preprocessing.preprocess_audio --workers 16 --sample-rate 16000
    uv run python -m src.preprocessing.preprocess_audio --overwrite   # reprocess existing files
    uv run python -m src.preprocessing.preprocess_audio --batch-size 128  # larger batches = less IPC overhead
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly


# ---------------------------------------------------------------------------
# Batch worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _process_batch(
    items: list[tuple[str, str, str]],
    target_sr: int,
) -> list[tuple[str, bool, str]]:
    """
    Load, mono-convert, resample, amplitude-normalise and save a batch of clips.

    Batching amortises per-task ProcessPoolExecutor overhead over many files.
    Uses soundfile + scipy/numpy to avoid torch framework overhead per file.
    Returns list of (utterance_id, success, message).
    """
    results = []
    for audio_path, utterance_id, out_path in items:
        try:
            data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            # data: [frames, channels]

            # Convert to mono
            if data.shape[1] > 1:
                data = data.mean(axis=1)
            else:
                data = data[:, 0]

            # Resample if needed
            if sr != target_sr:
                g = gcd(sr, target_sr)
                data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)

            # Amplitude normalise to [-1, 1]
            max_val = np.abs(data).max()
            data = data / (max_val + 1e-8)

            np.save(out_path, data)
            results.append((utterance_id, True, ""))
        except Exception as exc:
            results.append((utterance_id, False, str(exc)))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_items(
    jsonl_paths: list[Path],
    audio_dir: Path,
    out_dir: Path,
    overwrite: bool,
) -> list[tuple[str, str, str]]:
    """
    Return list of (audio_path, utterance_id, out_path) tuples for items that
    still need to be processed.
    """
    seen: set[str] = set()
    work: list[tuple[str, str, str]] = []
    skipped_missing = 0
    skipped_done = 0

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                uid = entry["utterance_id"]

                if uid in seen:
                    continue
                seen.add(uid)

                audio_filename = entry.get("audio_path", "").split("/")[-1]
                full_audio_path = audio_dir / audio_filename

                if not full_audio_path.exists():
                    skipped_missing += 1
                    continue

                out_path = out_dir / f"{uid}.npy"

                if not overwrite and out_path.exists():
                    skipped_done += 1
                    continue

                work.append((str(full_audio_path), uid, str(out_path)))

    print(f"  Unique utterances found : {len(seen)}")
    print(f"  Missing audio files     : {skipped_missing}")
    print(f"  Already processed       : {skipped_done}")
    print(f"  To process now          : {len(work)}")
    return work


def _print_progress(done: int, total: int, t0: float, errors: int) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    bar_width = 30
    filled = int(bar_width * done / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"\r  [{bar}] {done}/{total}  {rate:.0f}/s  ETA {eta:.0f}s  errors={errors}",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess all training audio to float32 .npy files."
    )
    parser.add_argument(
        "--audio-dir",
        default="data/audio",
        help="Source audio folder (default: data/audio)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/audio_processed",
        help="Output folder for .npy files (default: data/audio_processed)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count()-1,
        help="Number of parallel worker processes (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Files per worker task (higher = less IPC overhead; default: 64)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process files that already exist in the output folder.",
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        default=[
            "data/train_phon_transcripts.jsonl",
            "data/train_phon_transcripts_talkbank.jsonl",
        ],
        help="JSONL metadata files to source utterances from.",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_paths = [Path(p) for p in args.jsonl if Path(p).exists()]
    missing_jsonl = [p for p in args.jsonl if not Path(p).exists()]
    if missing_jsonl:
        print(f"[WARNING] JSONL files not found, skipping: {missing_jsonl}")
    if not jsonl_paths:
        print("[ERROR] No JSONL files found. Exiting.")
        sys.exit(1)

    print(f"\n=== Audio Preprocessor ===")
    print(f"  JSONL sources : {[str(p) for p in jsonl_paths]}")
    print(f"  Audio dir     : {audio_dir}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Sample rate   : {args.sample_rate} Hz")
    print(f"  Workers       : {args.workers}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Overwrite     : {args.overwrite}\n")

    print("Scanning metadata...")
    work = _collect_items(jsonl_paths, audio_dir, out_dir, args.overwrite)

    if not work:
        print("\nNothing to do — all files already processed.")
        return

    bs = args.batch_size
    batches = [work[i : i + bs] for i in range(0, len(work), bs)]
    print(f"\nProcessing {len(work)} files with {args.workers} workers ({len(batches)} batches of ~{bs})...\n")
    t0 = time.time()
    done = 0
    errors: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process_batch, batch, args.sample_rate): batch
            for batch in batches
        }

        for future in as_completed(futures):
            for uid, success, msg in future.result():
                done += 1
                if not success:
                    errors.append((uid, msg))
            _print_progress(done, len(work), t0, len(errors))

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.1f}s  ({done / elapsed:.0f} files/s)")
    print(f"  Succeeded : {done - len(errors)}")
    print(f"  Failed    : {len(errors)}")

    if errors:
        print("\nFailed utterances:")
        for uid, msg in errors[:20]:
            print(f"  {uid}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more.")

    # Write a small manifest so the dataset loader knows the cache is complete
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "sample_rate": args.sample_rate,
        "total_files": done - len(errors),
        "audio_dir": str(audio_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
