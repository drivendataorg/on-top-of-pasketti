#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
MODEL_PATH="Qwen3-ASR/Qwen3-ASR-1.7B"
TRAIN_FILE="data/pure/train.jsonl"
EVAL_FILE="data/pure/eval.jsonl"
OUTPUT_DIR="checkpoints"

# ── Training ───────────────────────────────────────────────────────────
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python -u src/run_train.py \
    --model_path "${MODEL_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --eval_file "${EVAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 3 \
    --eval_batch_size 1 \
    --grad_acc 12 \
    --lr 1e-5 \
    --epochs 2 \
    --warmup_ratio 0.02 \
    --save_steps 2000 \
    --log_steps 50 \
    --save_total_limit 5 \
    --num_workers 8 \
    --gradient_checkpointing 1 \
    --precache_workers 16 \
    --max_length 4096
