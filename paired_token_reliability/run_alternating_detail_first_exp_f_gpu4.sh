#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
export GPUS="${GPUS:-4}"
export OUTPUT="${OUTPUT:-exp_f/alternating_geo_detail_first_dual_c_gpu4}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_alternating_detail_first"
export RUN_EVAL="${RUN_EVAL:-0}"
export EPOCHS_A=1
export EPOCHS_B="${EPOCHS_B:-6}"
export EPOCHS_C=0
export PRETRAINED="${PRETRAINED:-ckpt/model.pt}"

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --point-weight 0.0 --pair-weight 0.0 --decomposition-weight 0.0 \
  --event-normal-weight 1.0 --depth-event-normal-weight 0.5 \
  --update-weight 0.0 --no-budget \
  "model.c_delay_steps=1000" "model.c_transition_steps=1000" \
  "model.pixel_refine_log_limit=0.30" "model.event_decay_tau=0.0015" "$@"
