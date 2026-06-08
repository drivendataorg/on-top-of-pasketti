"""Competition submission entry point for the Word track using Qwen3-ASR.

Reads utterance metadata from data/, runs Qwen3-ASR inference via vLLM,
and writes submission JSONL to submission/.

Usage:
    python src/run_inference.py
"""

import json
import os
from itertools import islice
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

from loguru import logger
import torch
from qwen_asr import Qwen3ASRModel

# ── Configuration ──────────────────────────────────────────────────────
SRC_ROOT = Path(__file__).parent.resolve()
MODEL_PATH = str(SRC_ROOT.parent / "model")
DATA_DIR =  Path("data")

USE_VLLM = True
BATCH_SIZE = 64
GPU_MEM_UTIL = 0.85
MAX_NEW_TOKENS = 256


def batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def main():
    logger.info("Torch version: {}", torch.__version__)
    logger.info("CUDA available: {}", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: {}", torch.cuda.get_device_name(0))

    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Backend: {'vLLM' if USE_VLLM else 'transformers'}")

    if USE_VLLM:
        logger.info(f"Loading Qwen3-ASR (vLLM) from: {MODEL_PATH}")
        model = Qwen3ASRModel.LLM(
            model=MODEL_PATH,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_inference_batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    else:
        logger.info(f"Loading Qwen3-ASR (transformers) from: {MODEL_PATH}")
        model = Qwen3ASRModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # ── Load data ──────────────────────────────────────────────────────
    data_dir = DATA_DIR
    manifest_path = data_dir / "utterance_metadata.jsonl"

    with manifest_path.open("r") as f:
        items = [json.loads(line) for line in f]

    items.sort(key=lambda x: x["audio_duration_sec"], reverse=True)
    logger.info(f"Processing {len(items)} utterances")

    # ── Inference ──────────────────────────────────────────────────────
    predictions = {}
    processed = 0
    log_step = max(1, len(items) // 20)

    for batch in batched(items, BATCH_SIZE):
        audio_paths = [str(data_dir / item["audio_path"]) for item in batch]
        results = model.transcribe(audio=audio_paths, language="English")

        for item, result in zip(batch, results):
            predictions[item["utterance_id"]] = result.text
            
        processed += len(batch)
        if processed % log_step < BATCH_SIZE:
            logger.info(f"Progress: {processed}/{len(items)}")

    logger.success(f"Transcription complete: {len(predictions)} utterances")

    # ── Write submission ───────────────────────────────────────────────
    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    with submission_format_path.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["orthographic_text"] = predictions.get(item["utterance_id"], "")
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    main()
