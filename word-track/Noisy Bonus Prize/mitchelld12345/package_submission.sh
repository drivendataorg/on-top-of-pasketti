#!/bin/bash
set -eou pipefail

export PYTHONPATH=.

MODEL="${1:?Usage: package_submission.sh <model_type> <checkpoint>}"
CHECKPOINT="${2:?Usage: package_submission.sh <model_type> <checkpoint>}"
TRACK="word"

SUBMISSION_DIR="submission"
RUNTIME_DIR="runtime"
OUTPUT="${RUNTIME_DIR}/submission/submission.zip"
MAIN_FILE="${SUBMISSION_DIR}/main_${MODEL}_${TRACK}.py"

[ -f "$MAIN_FILE" ] || { echo "Error: $MAIN_FILE not found"; exit 1; }
[ "$CHECKPOINT" = "none" ] || [ -e "$CHECKPOINT" ] || { echo "Error: checkpoint $CHECKPOINT not found"; exit 1; }

echo "=== Packaging ${MODEL} ${TRACK} submission ==="

micromamba run -n child_asr python submission/package.py "$MODEL" "$TRACK" "$CHECKPOINT"

echo "Copying ${MAIN_FILE} -> submission/main.py"
cp "$MAIN_FILE" "${SUBMISSION_DIR}/main.py"

echo "Packaging submission.zip..."
mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"

cd "$SUBMISSION_DIR"
mapfile -t FILES < .zip_manifest
zip -r -0 "../$OUTPUT" "${FILES[@]}"
cd ..

ls -lh "$OUTPUT"
echo "Done: $OUTPUT"
echo ""
echo "Test locally with:"
echo "  cd runtime && just track=word run"
