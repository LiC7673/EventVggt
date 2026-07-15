#!/usr/bin/env bash
# Final version: HDR token base plus RGB-free event/coarse-geometry pixel refiner.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
export GPUS="${GPUS:-4}"
export OUTPUT="${OUTPUT:-exp/linear_voxel_dual_alignment_hdr_final_pixel_refiner_gpu4}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_dual_alignment_hdr_final_pixel_refiner"
export RUN_EVAL=0 EPOCHS_A="${EPOCHS_A:-12}" EPOCHS_B=0 EPOCHS_C=0
export PRETRAINED="${PRETRAINED:-ckpt/model.pt}"

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --pair-mode anchor --point-weight 1.0 --decomposition-weight 0.0 \
  --no-pair-consistency --event-normal-weight 1.0 \
  --depth-event-normal-weight 0.5 --update-weight 0.001 \
  "model.hdr_token_bottleneck=256" "model.hdr_warmup_steps=1000" \
  "model.pixel_refiner_hidden=64" "model.pixel_refine_log_limit=0.20" \
  "model.event_count_cmax=3.0" \
  "model.event_decay_tau=0.0015" "model.depth_update_scale=0.50" \
  "model.support_dilation_kernel=3" "$@"
