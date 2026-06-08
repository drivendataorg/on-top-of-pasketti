"""
Test augmentations on real audio samples and save the results for listening.

Outputs are saved to exploration/augmentation_samples/ with this structure:
    <utterance_id>/
        original.wav
        time_stretch_<X.XX>x.wav
        pitch_shift_<X.X>st.wav
        band_stop.wav
        combined.wav

Run:
    uv run python exploration/test_augmentations.py
"""

import json
import random
from pathlib import Path

import torch
import torchaudio

# Import your new bridge class!
from src.preprocessing.augmenations import WaveformAugmentor

# ── Config ──────────────────────────────────────────────────────────────────
AUDIO_FOLDER = Path("data/audio")
JSONL_PATH = Path("data/train_phon_transcripts.jsonl")
OUTPUT_DIR = Path("exploration/augmentation_samples")
SAMPLE_RATE = 16_000
N_SAMPLES = 5  # number of utterances to test

# Augmentation settings to demo (p=1.0 so every sample is augmented)
TIME_STRETCH_RATES = [0.85, 0.90, 0.95]    # < 1.0 lowers pitch/formants together
PITCH_SEMITONES = [-4.0, -3.0, -2.0]       # independent pitch shifts


def load_and_prepare(audio_path: Path) -> torch.Tensor:
    """Load audio, convert to mono 16 kHz, normalise to [-1, 1]."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.squeeze(0)  # [T]


def save_wav(waveform: torch.Tensor, path: Path) -> None:
    """Save 1-D waveform tensor as 16-bit WAV, preventing clipping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Re-normalize to avoid clipping distortion from the DSP
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / (peak + 1e-8)
        
    torchaudio.save(
        str(path),
        waveform.unsqueeze(0),  # Restore 2D shape for saving
        SAMPLE_RATE,
    )

def main():
    # ── Pick random samples ─────────────────────────────────────────────
    with open(JSONL_PATH) as f:
        all_items = [json.loads(line) for line in f]

    random.seed(42)
    samples = random.sample(all_items, min(N_SAMPLES, len(all_items)))

    print(f"Testing audiomentations on {len(samples)} samples")
    print(f"Output dir: {OUTPUT_DIR.resolve()}\n")

    for item in samples:
        uid = item["utterance_id"]
        audio_file = AUDIO_FOLDER / f"{uid}.flac"
        if not audio_file.exists():
            print(f"  [skip] {uid} — file not found")
            continue

        out_dir = OUTPUT_DIR / uid
        waveform = load_and_prepare(audio_file)
        dur_ms = waveform.shape[0] / SAMPLE_RATE * 1000

        print(f"── {uid}  ({dur_ms:.0f} ms, age={item.get('age_bucket', '?')}) ──")
        print(f"   phonetic: {item.get('phonetic_text', '')}")

        # 1. Original
        save_wav(waveform, out_dir / "original.wav")
        print(f"   saved original.wav")

        # 2. Time Stretch (Replaces VTLN)
        for rate in TIME_STRETCH_RATES:
            cfg = {
                "time_stretch": {
                    "enabled": True, "min_rate": rate, "max_rate": rate, "p": 1.0
                }
            }
            aug = WaveformAugmentor.from_config(cfg, sample_rate=SAMPLE_RATE)
            augmented = aug(waveform.clone())
            fname = f"time_stretch_{rate:.2f}x.wav"
            save_wav(augmented, out_dir / fname)
            print(f"   saved {fname}")

        # 3. Pitch shift at each semitone value
        for st in PITCH_SEMITONES:
            cfg = {
                "pitch_shift": {
                    "enabled": True, "min_semitones": st, "max_semitones": st, "p": 1.0
                }
            }
            aug = WaveformAugmentor.from_config(cfg, sample_rate=SAMPLE_RATE)
            augmented = aug(waveform.clone())
            fname = f"pitch_shift_{st:+.1f}st.wav"
            save_wav(augmented, out_dir / fname)
            print(f"   saved {fname}")

        # 4. Band Stop Filter isolated test
        cfg = {
            "band_stop_filter": {
                "enabled": True, 
                "min_center_freq": 1000.0, "max_center_freq": 1000.0,  # Fixed at 1kHz for testing
                "min_bandwidth_fraction": 1.0, "max_bandwidth_fraction": 1.0, 
                "p": 1.0
            }
        }
        aug = WaveformAugmentor.from_config(cfg, sample_rate=SAMPLE_RATE)
        augmented = aug(waveform.clone())
        save_wav(augmented, out_dir / "band_stop_filter.wav")
        print(f"   saved band_stop_filter.wav")

        # 5. Combined: Time Stretch + Pitch Shift + Band Stop
        mid_rate = TIME_STRETCH_RATES[len(TIME_STRETCH_RATES) // 2]
        mid_st = PITCH_SEMITONES[len(PITCH_SEMITONES) // 2]
        
        cfg_combined = {
            "time_stretch": {"enabled": True, "min_rate": mid_rate, "max_rate": mid_rate, "p": 1.0},
            "pitch_shift": {"enabled": True, "min_semitones": mid_st, "max_semitones": mid_st, "p": 1.0},
            "band_stop_filter": {
                "enabled": True, 
                "min_center_freq": 500.0, "max_center_freq": 2000.0,
                "min_bandwidth_fraction": 0.5, "max_bandwidth_fraction": 1.5,
                "p": 1.0
            }
        }
        aug_combined = WaveformAugmentor.from_config(cfg_combined, sample_rate=SAMPLE_RATE)
        combined_wav = aug_combined(waveform.clone())
        
        combined_fname = f"combined_ts{mid_rate:.2f}_ps{mid_st:+.1f}st_bsf.wav"
        save_wav(combined_wav, out_dir / combined_fname)
        print(f"   saved {combined_fname}")

        print()

    print(f"Done! Listen to the files in:\n  {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()