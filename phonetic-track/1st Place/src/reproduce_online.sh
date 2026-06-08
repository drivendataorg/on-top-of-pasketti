#!/usr/bin/env bash
# Reproduce the 11 official online/full-data phonetic models from src/models.txt
# in this standalone solution repo.
#
# Usage:
#   bash reproduce_online.sh        # train all missing models
#   FORCE=1 bash reproduce_online.sh # retrain even if output dirs exist
#   DRY_RUN=1 bash reproduce_online.sh
#
# Env:
#   GPU=0
#   PYTHON=python
#   ROOT=/path/to/childrens-phonetic-asr
#   EXT_ROOT=/path/to/childrens-ext-asr
#   EXTRA_ARGS="--bs=1 --eval_bs=1 --num_workers=0"  # appended to every run
set -euo pipefail

cd "$(dirname "$0")"

resolve_data_dir() {
  local name="$1"
  local explicit="${2:-}"
  if [[ -n "$explicit" ]]; then
    echo "$explicit"
    return
  fi
  local candidates=(
    "../input/$name"
    "../../input/$name"
    "../../pasketti-phonetic/input/$name"
    "../../pasketti/input/$name"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -e "$path" ]]; then
      echo "$path"
      return
    fi
  done
  echo "../../input/$name"
}

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
ROOT="$(resolve_data_dir childrens-phonetic-asr "${ROOT:-}")"
EXT_ROOT="$(resolve_data_dir childrens-ext-asr "${EXT_ROOT:-}")"
EXPORT_ARGS="${EXPORT_ARGS:---eval_ext_full --eval_ext_weight=1 --save_logprobs --save_dual_head_preds --save_pred_score}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_ROOT="../working/online/9"

MODELS=(
  "v17.backbone-wavlm-large.ep3.5.leval|--flagfile=flags/v17 --backbone=wavlm-large --exit_epoch=3.5"
  "v16.backbone-wavlm-large.dual_bpe.mix4.eval|--flagfile=flags/v16 --backbone=wavlm-large --dual_bpe --mix4 --exit_epoch=5"
  "v16.backbone-wavlm-large.dual_bpe.mix4.mix_csss.ep4.5.eval|--flagfile=flags/v16 --backbone=wavlm-large --dual_bpe --mix4 --mix_csss --exit_epoch=4.5"
  "v16.backbone-wavlm-large.dual_bpe.eval|--flagfile=flags/v16 --backbone=wavlm-large --dual_bpe --exit_epoch=7.5"
  "v16.dual_bpe.tdt_only.eval|--flagfile=flags/v16 --dual_bpe --tdt_only --exit_epoch=10.5"
  "v16.dual_bpe.mix_csss.tdt_only.eval|--flagfile=flags/v16 --dual_bpe --mix_csss --tdt_only --exit_epoch=4"
  "v16.dual_bpe.mix2.mix_csss.tdt_only.eval|--flagfile=flags/v16 --dual_bpe --mix2 --mix_csss --tdt_only --exit_epoch=7 --gradc"
  "v16.dual_bpe.wo_scale-2.eval|--flagfile=flags/v16 --dual_bpe --wo_scale=2 --exit_epoch=13"
  "v16.aux_loss.dual_bpe.eval|--flagfile=flags/v16 --dual_bpe --aux_loss --exit_epoch=11 --gradc"
  "v16.dual_bpe.mix4.eval|--flagfile=flags/v16 --dual_bpe --mix4 --exit_epoch=11.5 --gradc"
  "v16.dual_bpe.mix2.eval|--flagfile=flags/v16 --dual_bpe --mix2 --exit_epoch=18 --gradc"
)

run_cmd() {
  local cmd="$1"
  echo "+ $cmd"
  if [[ "$DRY_RUN" != "1" ]]; then
    eval "$cmd"
  fi
}

for rec in "${MODELS[@]}"; do
  IFS='|' read -r mn args <<< "$rec"
  out_dir="$RUN_ROOT/$mn/0"
  if [[ "$FORCE" != "1" && -s "$out_dir/model.pt" ]]; then
    echo "[skip] $mn has checkpoint: $out_dir/model.pt"
    continue
  fi

  cmd="PYTHONPATH=_compat:\$PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py $args --mn=$mn --online --wandb=0 --root=$ROOT --ext_root=$EXT_ROOT --eval_ext_root=$EXT_ROOT $EXPORT_ARGS"
  if [[ -n "$EXTRA_ARGS" ]]; then
    cmd="$cmd $EXTRA_ARGS"
  fi
  echo "[online] $mn"
  run_cmd "$cmd"
done
