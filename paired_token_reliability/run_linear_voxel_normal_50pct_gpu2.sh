#!/usr/bin/env bash
set -Eeuo pipefail
export GPUS=2 OUTPUT="${OUTPUT:-exp/linear_voxel_normal_50pct}"
bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --decomposition-weight 1.0 --pair-weight 0.10 \
  "model.depth_update_scale=0.50" "$@"
