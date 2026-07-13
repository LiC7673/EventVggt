#!/usr/bin/env bash
set -Eeuo pipefail
export GPUS="${GPUS:-0}" OUTPUT="${OUTPUT:-exp/ablation_no_event_normal}"
bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
 --point-weight 0 --decomposition-weight 1.0 --pair-weight 0.10 \
 --event-normal-weight 0 --depth-event-normal-weight 0 \
 "model.depth_update_scale=1.0" "$@"
