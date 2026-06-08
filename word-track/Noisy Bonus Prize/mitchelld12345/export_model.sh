#!/bin/bash
set -eou pipefail

export PYTHONPATH=.

CHECKPOINT="${1:?Usage: export_model.sh <checkpoint.ckpt> [output_dir]}"
OUTPUT_DIR="${2:-exported_model}"

echo "=== Exporting Qwen3-ASR checkpoint to safetensors ==="
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_DIR"

micromamba run -n child_asr python src/models/export_qwen.py \
    "$CHECKPOINT" \
    --output "$OUTPUT_DIR"

echo ""
echo "Done. Model saved to $OUTPUT_DIR"
echo "Run inference with:"
echo "  bash run_inference.sh $OUTPUT_DIR data/raw/audio runtime/data-smoke/word/submission_format.jsonl"
