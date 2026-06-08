#!/usr/bin/env bash
# Reproduce the second-stage CatBoost tree reranker from the 11 offline fold0
# model eval artifacts listed in models.txt.
#
# Important: this script trains the reranker from *offline fold-0 eval artifacts*.
# The released Hugging Face weights are online/full-data inference checkpoints plus
# a pre-trained tree_reranker/ directory for make pack; they are not sufficient to
# retrain the tree reranker from scratch.
#
# Prerequisite:
#   bash reproduce_offline_fold0.sh
#
# Required per model under ../working/offline/9/<model_name>/0/:
#   eval.csv
# Used when available:
#   ctc_logprobs.pt, dual_head_preds.pt, aux_meta_preds.pt, pred_score columns in eval.csv
# model.pt is not required for the default tree-reranker training path.
#
# Usage:
#   bash reproduce_tree_reranker.sh
#   DRY_RUN=1 bash reproduce_tree_reranker.sh
#
# Env:
#   GPU=0
#   PYTHON=python
#   RUN_ROOT=../working/offline/9  # override offline artifact root if needed
#   EXTRA_ARGS=""      # appended to ensemble.py command
#   COPY_TO_RELEASE=1  # copy artifacts to src/tree_reranker for pack_submission.sh
set -euo pipefail

cd "$(dirname "$0")"

resolve_run_root() {
  if [[ -n "${RUN_ROOT:-}" ]]; then
    echo "$RUN_ROOT"
    return
  fi
  local first_model
  first_model="$(grep -v '^#' models.txt | sed '/^$/d' | head -1 | xargs || true)"
  local candidates=(
    "../working/offline/9"
    "../../pasketti-phonetic/working/offline/9"
    "../../pasketti/working/offline/9"
  )
  local root
  for root in "${candidates[@]}"; do
    if [[ -n "$first_model" && -s "$root/$first_model/0/eval.csv" ]]; then
      echo "$root"
      return
    fi
  done
  echo "../working/offline/9"
}

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
COPY_TO_RELEASE="${COPY_TO_RELEASE:-1}"
RUN_ROOT="$(resolve_run_root)"
TREE_MNS="ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407"
TREE_DIR="${TREE_DIR:-$RUN_ROOT/$TREE_MNS/0}"
RELEASE_TREE_DIR="${RELEASE_TREE_DIR:-tree_reranker}"
echo "Using offline artifact root: $RUN_ROOT"

missing=0
n_models=0
n_logprobs=0
while IFS= read -r raw; do
  line="${raw%%#*}"
  mn="$(echo "$line" | xargs || true)"
  [[ -n "$mn" ]] || continue
  n_models=$((n_models + 1))
  model_dir="$RUN_ROOT/$mn/0"
  if [[ ! -s "$model_dir/eval.csv" ]]; then
    echo "ERROR: missing required $model_dir/eval.csv" >&2
    missing=1
  fi
  if [[ -s "$model_dir/ctc_logprobs.pt" ]]; then
    n_logprobs=$((n_logprobs + 1))
  else
    echo "WARN: missing optional score artifact $model_dir/ctc_logprobs.pt" >&2
  fi
  for f in dual_head_preds.pt aux_meta_preds.pt flags.json model.pt; do
    if [[ ! -s "$model_dir/$f" ]]; then
      echo "WARN: missing optional $model_dir/$f" >&2
    fi
  done
done < models.txt

if [[ "$missing" == "1" ]]; then
  echo "Run first: bash reproduce_offline_fold0.sh" >&2
  if [[ "$DRY_RUN" != "1" ]]; then
    exit 1
  fi
fi
if [[ "$n_logprobs" == "0" ]]; then
  echo "ERROR: no ctc_logprobs.pt found. Tree reranker needs at least one score model." >&2
  if [[ "$DRY_RUN" != "1" ]]; then
    exit 1
  fi
fi
echo "Found $n_models model eval dirs; $n_logprobs have ctc_logprobs.pt."

cmd="PYTHONPATH=_compat:\$PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU $PYTHON ensemble.py --ensemble_working_dir=$RUN_ROOT --feat_nemo_group --feat_tdt_group --feat_wavlm_group --mns=.0407"
if [[ -n "$EXTRA_ARGS" ]]; then
  cmd="$cmd $EXTRA_ARGS"
fi

echo "+ $cmd"
if [[ "$DRY_RUN" != "1" ]]; then
  eval "$cmd"
fi

if [[ "$COPY_TO_RELEASE" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ copy $TREE_DIR -> $RELEASE_TREE_DIR"
  else
    mkdir -p "$RELEASE_TREE_DIR"
    for f in reranker_meta.json reranker_features.txt reranker_experiment.json reranker_feature_importance.csv metrics.csv eval.csv; do
      [[ -f "$TREE_DIR/$f" ]] && cp "$TREE_DIR/$f" "$RELEASE_TREE_DIR/"
    done
    for d in "$TREE_DIR"/tree_*_fold*; do
      [[ -d "$d" ]] || continue
      rm -rf "$RELEASE_TREE_DIR/$(basename "$d")"
      cp -r "$d" "$RELEASE_TREE_DIR/"
    done
    echo "Copied tree reranker artifacts to $RELEASE_TREE_DIR"
  fi
fi
