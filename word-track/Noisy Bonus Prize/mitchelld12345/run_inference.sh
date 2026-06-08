#!/bin/bash
set -eou pipefail

export PYTHONPATH=.
ENV="micromamba run -n child_asr"

MODEL_DIR="${1:-submission/qwen_model}"
AUDIO_DIR="${2:-data/raw/audio}"
INPUT_JSONL="${3:-data/raw/submission_format_aqPHQ8m.jsonl}"
OUTPUT_JSONL="${4:-predictions.jsonl}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: model directory '$MODEL_DIR' not found"
    echo "Usage: bash run_inference.sh [MODEL_DIR] [AUDIO_DIR] [INPUT_JSONL] [OUTPUT_JSONL]"
    echo ""
    echo "  MODEL_DIR    Path to Qwen3-ASR model directory (default: submission/qwen_model)"
    echo "  AUDIO_DIR    Path to directory containing .flac audio files (default: data/raw/audio)"
    echo "  INPUT_JSONL  Path to JSONL with utterance_id fields (default: data/raw/submission_format_aqPHQ8m.jsonl)"
    echo "  OUTPUT_JSONL Path for output predictions (default: predictions.jsonl)"
    exit 1
fi

echo "=== Running inference ==="
echo "Model:  $MODEL_DIR"
echo "Audio:  $AUDIO_DIR"
echo "Input:  $INPUT_JSONL"
echo "Output: $OUTPUT_JSONL"

$ENV python src/eval/run_inference.py \
    --model_dir "$MODEL_DIR" \
    --audio_dir "$AUDIO_DIR" \
    --input "$INPUT_JSONL" \
    --output "$OUTPUT_JSONL"

echo ""
echo "=== Done ==="
echo "Predictions saved to $OUTPUT_JSONL"
