#!/usr/bin/env bash
# ============================================================================
# Sync core project source from the upstream pikachu development repo into
# this self-contained solution repo. This is a one-shot helper used while
# preparing the open-source release; end users should NOT need to run it
# (the files it copies are already committed here).
#
# Usage:
#   bash scripts/sync_core_from_pikachu.sh /path/to/pikachu
# ============================================================================

set -euo pipefail

PIKACHU="${1:-/home/gezi/pikachu}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -d "$PIKACHU/projects/drivendata/pasketti/src" ]]; then
  echo "Error: $PIKACHU does not look like the pikachu repo." >&2
  exit 1
fi

PA_SRC="$PIKACHU/projects/drivendata/pasketti/src"
PH_SRC="$PIKACHU/projects/drivendata/pasketti-phonetic/src"
DST="$HERE/src"

echo "Syncing from $PIKACHU into $DST ..."

# --- shared training/inference code (from pasketti/src) ---
for f in dataset.py preprocess.py eval.py ctc_decode.py util.py \
         submit.py submit_nbest_helper.py reranker_features.py \
         train_sampling.py nemo_trim_vocab_data.py \
         config_base.py __init__.py; do
  cp -v "$PA_SRC/$f" "$DST/$f"
done

# --- model wrappers ---
mkdir -p "$DST/models"
for f in __init__.py base.py nemo.py wav2vec2.py whisper.py squeezeformer.py moonshine.py; do
  if [[ -f "$PA_SRC/models/$f" ]]; then
    cp -v "$PA_SRC/models/$f" "$DST/models/$f"
  fi
done

# --- phonetic-track entry point + ensemble ---
cp -v "$PH_SRC/config.py"          "$DST/config.py"
cp -v "$PH_SRC/ensemble.py"        "$DST/ensemble.py"
cp -v "$PH_SRC/ensemble_feats.py"  "$DST/ensemble_feats.py"
cp -v "$PH_SRC/models.txt"         "$DST/models.txt"

# --- flag files (only the ones referenced by the final ensemble chain) ---
mkdir -p "$DST/flags"
for f in base v8 v9 v11 v12-dual-ipa v13 v13-ema v13-dual v13-dual-bpe \
         v14-ema v14-ema5-shuffle v14-ema5-shuffle-more v14-ema5-shuffle-more-dual \
         v15 v15-cnoise v15-dual-bpe v16 v17; do
  if [[ -f "$PH_SRC/flags/$f" ]]; then
    cp -v "$PH_SRC/flags/$f" "$DST/flags/$f"
  fi
done

echo "Done."
