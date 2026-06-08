#!/usr/bin/env bash
# ==============================================================================
# Pack a DrivenData submission.zip for the Pasketti Phonetic track.
#
# Modes:
#   single   bash scripts/pack_submission.sh single  <model_dir>
#   ensemble bash scripts/pack_submission.sh ensemble <models_file> [model_dir1 model_dir2 ...]
#
# The script:
#   1. Stages a temporary directory.
#   2. Copies submit.py -> main.py (the runtime entry point inside the Docker
#      container; see the DrivenData Phonetic runtime for the contract).
#   3. Copies all training source files (src/, models/, ...) verbatim — the
#      submission re-uses the exact same model code as training.
#   4. Tarballs the bundled compatibility shims (_compat/{gezi,melt,lele,husky})
#      as ``pikachu_utils.tar.gz`` so submit.py's existing extraction logic
#      can put them on sys.path. NO modification to submit.py is required.
#   5. Optionally copies tree_reranker/ artifacts for tree-based ensemble
#      inference (reranker_meta.json, tree_cb_fold*/, ...).
#   6. Copies model weight directories into model/, model_1/, ... .
#   7. Writes ensemble_meta.json (ensemble mode only).
#   8. zip -r submission.zip ./*  with .nemo files stored uncompressed.
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/src"
TREE_RERANKER_DIR="${TREE_RERANKER_DIR:-$SRC/tree_reranker}"

usage() {
  cat <<'EOF'
Usage:
  pack_submission.sh single  <model_dir> [output_zip]
  pack_submission.sh ensemble <models_file> [output_zip]

  models_file is a text file with one model directory path per line. Lines
  starting with '#' are skipped. Order = inference order (matters for
  ensemble logprob accumulation).
EOF
  exit 1
}

[ $# -lt 2 ] && usage

MODE="$1"; shift
STAGE=$(mktemp -d)
trap "rm -rf $STAGE" EXIT
echo "==> Staging in $STAGE"

# ---- 1. Copy training source verbatim ----
mkdir -p "$STAGE/src"
cp -r "$SRC"/*.py "$STAGE/src/"
cp -r "$SRC/models" "$STAGE/src/"
[ -d "$SRC/metric" ] && cp -r "$SRC/metric" "$STAGE/src/" || true

# ---- 1b. Copy tree reranker artifacts when available ----
if [ -d "$TREE_RERANKER_DIR" ]; then
  cp -r "$TREE_RERANKER_DIR" "$STAGE/src/"
  echo "  bundled tree_reranker from $TREE_RERANKER_DIR"
fi

# ---- 2. submit.py -> main.py (Docker entry point) ----
cp "$SRC/submit.py" "$STAGE/main.py"

# ---- 3. Bundle the compatibility shim as pikachu_utils.tar.gz ----
#         (drop into the same staging dir; submit.py auto-extracts it.)
( cd "$SRC/_compat" && tar -czf "$STAGE/pikachu_utils.tar.gz" gezi melt lele husky )
echo "  bundled $(du -h "$STAGE/pikachu_utils.tar.gz" | cut -f1) of compatibility shims"

# ---- 4. Copy model directories ----
copy_model() {
  local idx="$1"; local src_dir="$2"
  local dst
  if [ "$idx" = "0" ]; then dst="$STAGE/model"; else dst="$STAGE/model_${idx}"; fi
  mkdir -p "$dst"
  for f in model.pt flags.json model_meta.json nemo_model_slim.nemo \
           preprocessor_config.json config.json; do
    [ -f "$src_dir/$f" ] && cp "$src_dir/$f" "$dst/" || true
  done
  echo "  model/${idx}: $(basename "$src_dir")"
}

if [ "$MODE" = "single" ]; then
  MODEL_DIR="$1"; shift
  OUT_ZIP="${1:-$REPO/submission.zip}"
  copy_model 0 "$MODEL_DIR"
elif [ "$MODE" = "ensemble" ]; then
  MODELS_FILE="$1"; shift
  OUT_ZIP="${1:-$REPO/submission.zip}"
  if [ ! -f "$MODELS_FILE" ]; then
    echo "ERROR: models_file not found: $MODELS_FILE"; exit 1
  fi
  if [ ! -d "$TREE_RERANKER_DIR" ]; then
    echo "WARNING: tree_reranker dir not found at $TREE_RERANKER_DIR"
    echo "         final zip will only work for non-tree-reranker ensemble modes"
  fi
  IDX=0
  python3 "$REPO/scripts/_resolve_models.py" "$MODELS_FILE" | while IFS= read -r mdir; do
    copy_model "$IDX" "$mdir"
    IDX=$((IDX+1))
  done

  # ensemble_meta.json — submit.py reads this to enable multi-model mode.
  python3 - "$STAGE" "$MODELS_FILE" <<'PY'
import json, os, sys
stage, models_file = sys.argv[1], sys.argv[2]
n = 0
for m in sorted(os.listdir(stage)):
    if m == 'model' or m.startswith('model_'):
        n += 1
meta = {'ensemble': True, 'n_models': n,
        'inference_order': [m for m in sorted(os.listdir(stage))
                            if m == 'model' or m.startswith('model_')]}
with open(os.path.join(stage, 'ensemble_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print(f'==> ensemble_meta.json: n_models={n}')
PY
else
  usage
fi

# ---- 5. Zip — store .nemo uncompressed (already a tar.gz internally) ----
mkdir -p "$(dirname "$OUT_ZIP")"
rm -f "$OUT_ZIP"
( cd "$STAGE" && zip -r "$OUT_ZIP" . -x '*.nemo' \
  && find . -name '*.nemo' -print0 | xargs -0 zip -0 "$OUT_ZIP" 2>/dev/null || true )

echo "==> Wrote $OUT_ZIP ($(du -sh "$OUT_ZIP" | cut -f1))"
unzip -l "$OUT_ZIP" | tail -10
