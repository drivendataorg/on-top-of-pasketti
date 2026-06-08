"""Benchmark ASR models on training data with a random held-out split."""
import argparse
import json
import random

import dirtygit
import jiwer
from loguru import logger
from whisper_normalizer.english import EnglishTextNormalizer

from src.config import setup_logging
from src.data.utils import load_jsonl
from src.paths import RAW_AUDIO_DIR, TRAIN_TRANSCRIPTS

normalize = EnglishTextNormalizer()


def benchmark(model, metric="wer", split_ratio=0.1, seed=42, max_samples=None, output=None, githash=None):
    metric_fn = getattr(jiwer, metric)
    transcripts = load_jsonl(TRAIN_TRANSCRIPTS)

    random.seed(seed)
    random.shuffle(transcripts)

    n_eval = int(len(transcripts) * split_ratio)
    eval_set = transcripts[:n_eval]
    if max_samples:
        eval_set = eval_set[:max_samples]

    results = []
    refs_raw, hyps_raw = [], []
    refs_norm, hyps_norm = [], []
    for entry in eval_set:
        audio_path = RAW_AUDIO_DIR.parent / entry["audio_path"]
        ref_raw = entry["orthographic_text"]
        hyp_raw = model.inference_file(audio_path)
        ref_n = normalize(ref_raw)
        hyp_n = normalize(hyp_raw)
        sample_wer = metric_fn(ref_raw, hyp_raw)
        results.append({
            "utterance_id": entry["utterance_id"],
            "reference": ref_raw,
            "hypothesis": hyp_raw,
            "reference_norm": ref_n,
            "hypothesis_norm": hyp_n,
            "wer": round(sample_wer, 4),
        })
        refs_raw.append(ref_raw)
        hyps_raw.append(hyp_raw)
        refs_norm.append(ref_n)
        hyps_norm.append(hyp_n)

    score_raw = metric_fn(refs_raw, hyps_raw)
    score_norm = metric_fn(refs_norm, hyps_norm)
    logger.info(f"{metric.upper()}: {score_raw:.4f} (raw), {score_norm:.4f} (norm) — {len(eval_set)} samples, seed={seed}")

    if output:
        payload = {
            "githash": githash,
            "metric": metric,
            "seed": seed,
            "num_samples": len(results),
            f"{metric}_raw": score_raw,
            f"{metric}_norm": score_norm,
            "results": results,
        }
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved {len(results)} results to {output}")

    return score_raw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="wer", choices=["wer", "cer"])
    parser.add_argument("--split_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    return args


def main(args):
    logger.info(f"githash={args.githash}")
    logger.info("Import and pass a model to benchmark(). See README.")


if __name__ == "__main__":
    githash = dirtygit.check()
    args = parse_args()
    args.githash = githash
    setup_logging()
    main(args)
