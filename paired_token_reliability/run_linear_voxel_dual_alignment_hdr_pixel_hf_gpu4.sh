#!/usr/bin/env bash
# V11: full-resolution, edge-balanced, multi-scale event normal derivatives.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
export GPUS="${GPUS:-4}"
export OUTPUT="${OUTPUT:-exp/linear_voxel_dual_alignment_hdr_pixel_hf_v11_gpu4}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_dual_alignment_hdr_pixel_hf"
export RUN_EVAL=0 EPOCHS_A="${EPOCHS_A:-12}" EPOCHS_B=0 EPOCHS_C=0
export PRETRAINED="${PRETRAINED:-ckpt/model.pt}"

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --pair-mode anchor --point-weight 1.0 --decomposition-weight 0.0 \
  --no-pair-consistency --event-normal-weight 1.0 \
  --depth-event-normal-weight 0.5 --update-weight 0.001 \
  "model.hdr_token_bottleneck=256" "model.hdr_warmup_steps=1000" \
  "model.normal_refine_iterations=1" "model.depth_update_scale=0.50" \
  "model.support_dilation_kernel=3" "$@"
