"""Convert competition training data to Qwen3-ASR fine-tuning JSONL format.

The competition provides training data as JSONL with fields like:
    {"utterance_id": "U_xxx", "audio_path": "audio/U_xxx.flac", "orthographic_text": "..."}

This script converts them into the format expected by train.py:
    {"audio": "/abs/path/to/audio.flac", "text": "language English<asr_text>transcription"}

Usage:
    python src/prepare_train_data.py \
        --data_dir ./data \
        --transcript_file ./data/train_word_transcripts.jsonl \
        --output_file ./data/train.jsonl
"""

import argparse
import json
from pathlib import Path

from loguru import logger


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Qwen3-ASR training data from competition format")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root data directory (audio_path in JSONL is relative to this)")
    p.add_argument("--transcript_file", type=str, required=True,
                   help="JSONL file with utterance_id, audio_path, and orthographic_text fields")
    p.add_argument("--output_file", type=str, default="./train.jsonl", help="Output training JSONL")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    with open(args.transcript_file, "r") as f:
        items = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(items)} transcription entries from {args.transcript_file}")

    samples = []
    skipped = 0
    for item in items:
        text = item.get("orthographic_text", "").strip()
        if not text:
            skipped += 1
            continue

        audio_rel = item.get("audio_path", "")
        audio_path = data_dir / audio_rel

        if not audio_path.exists():
            uid = item.get("utterance_id", "")
            for ext in [".flac", ".wav", ".mp3", ".ogg"]:
                candidate = data_dir / "audio" / f"{uid}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break

        if not audio_path.exists():
            logger.warning(f"Audio not found: {audio_path}, skipping {item.get('utterance_id')}")
            skipped += 1
            continue

        samples.append({
            "audio": str(audio_path),
            "text": f"language English<asr_text>{text}",
        })

    logger.info(f"Built {len(samples)} training samples (skipped {skipped})")

    if not samples:
        logger.error("No valid samples found. Check --data_dir and audio paths.")
        return

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info(f"Wrote {len(samples)} training samples to {args.output_file}")


if __name__ == "__main__":
    main()
