#!/usr/bin/env bash
# A: E_geo direct 50% pixel geometry; B: frozen decoder + E_full alignment/C.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
export GPUS="${GPUS:-4}"
export OUTPUT="${OUTPUT:-exp_f/staged_geo_direct50_full_align_gpu4}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_staged_geo_direct50"
export RUN_EVAL="${RUN_EVAL:-0}"
export EPOCHS_A="${EPOCHS_A:-4}"
export EPOCHS_B="${EPOCHS_B:-8}"
export EPOCHS_C="${EPOCHS_C:-0}"
export PRETRAINED="${PRETRAINED:-ckpt/model.pt}"

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --point-weight 0.0 --decomposition-weight 1.0 --pair-weight 0.10 \
  --event-normal-weight 1.0 --depth-event-normal-weight 0.5 \
  --update-weight 0.001 "model.depth_update_scale=0.50" \
  "model.support_dilation_kernel=5" "model.event_decay_tau=0.003" "$@"
