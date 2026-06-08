"""Prepare pure (non-augmented) dataset from competition + TalkBank data.

Merges data/train.jsonl and data/talkbank_train.jsonl,
shuffles with seed=42, splits into 97/3 train/eval, and writes to data/pure/.

Usage:
    python src/prepare_pure.py
"""

import json
import os
import random

SEED = 42
EVAL_RATIO = 0.03

SRC_FILES = [
    "data/train.jsonl",
    "data/talkbank_train.jsonl",
]
OUT_DIR = "data/pure"


def main():
    all_samples = []
    for path in SRC_FILES:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(line)
        print(f"Loaded {path}: running total = {len(all_samples)}")

    random.seed(SEED)
    random.shuffle(all_samples)

    n_eval = int(len(all_samples) * EVAL_RATIO)
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    print(f"Total: {len(all_samples)}  Train: {len(train_samples)}  Eval: {len(eval_samples)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, samples in [("train.jsonl", train_samples), ("eval.jsonl", eval_samples)]:
        out_path = os.path.join(OUT_DIR, name)
        with open(out_path, "w") as f:
            for s in samples:
                f.write(s + "\n")
        print(f"Wrote {out_path} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
