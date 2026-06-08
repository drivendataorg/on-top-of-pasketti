#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SUBMISSION_SRC="$SCRIPT_DIR/src"
CONFIGS_DIR="$SCRIPT_DIR/configs"
CONFIG_FILE="${1:-$CONFIGS_DIR/config.yaml}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/submission}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# 指定された config を src/config.yaml として配置
cp "$CONFIG_FILE" "$SUBMISSION_SRC/config.yaml"

# csrc パッケージをコピー（中身を上書き）
CSRC_PKG="$SCRIPT_DIR/../csrc/src/csrc"
rm -rf "$SUBMISSION_SRC/lib/csrc"
cp -r "$CSRC_PKG" "$SUBMISSION_SRC/lib/csrc"

# qwen_asr パッケージをコピー（src/ 直下に配置）
QWEN_ASR_PKG="$SCRIPT_DIR/../csrc_qwen/src/csrc_qwen/qwen_asr"
rm -rf "$SUBMISSION_SRC/qwen_asr"
cp -r "$QWEN_ASR_PKG" "$SUBMISSION_SRC/qwen_asr"

rm -f "$OUTPUT_DIR/submission.zip"

PACK_FILES=(main.py config.yaml lib/ qwen_asr/ model/)
if [ -f "$SUBMISSION_SRC/vllm_wheels.zip" ]; then
    PACK_FILES+=(vllm_wheels.zip)
fi

(
    cd "$SUBMISSION_SRC" \
    && zip -r -0 "$OUTPUT_DIR/submission.zip" "${PACK_FILES[@]}"
)

echo "Created: $OUTPUT_DIR/submission.zip"
