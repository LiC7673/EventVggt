#!/usr/bin/env bash
set -Eeuo pipefail
export GPUS="${GPUS:-1}" OUTPUT="${OUTPUT:-exp/ablation_no_multildr}"
bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
 --point-weight 0 --decomposition-weight 1.0 --no-pair-consistency \
 "model.depth_update_scale=1.0" "$@"
