#!/usr/bin/env bash
set -Eeuo pipefail
export GPUS="${GPUS:-2}" OUTPUT="${OUTPUT:-exp/ablation_no_contribution_c1}"
bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
 --point-weight 0 --decomposition-weight 0 --no-pair-consistency --geometry-rank-weight 0 \
 "model.force_full_contribution=true" "model.depth_update_scale=1.0" "$@"
