#!/usr/bin/env bash
set -Eeuo pipefail
export GPUS="${GPUS:-0}" OUTPUT="${OUTPUT:-exp/ablation_full_three_strategies}"
bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
 --point-weight 0 --decomposition-weight 1.0 --pair-weight 0.10 \
 "model.depth_update_scale=1.0" "$@"
