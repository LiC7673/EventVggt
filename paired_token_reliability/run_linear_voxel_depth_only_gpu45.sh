#!/usr/bin/env bash
set -Eeuo pipefail
export GPUS=4,5 OUTPUT="${OUTPUT:-exp/linear_voxel_depth_only_metric_100pct_gpu45}"
bash paired_token_reliability/run_linear_voxel_depth_metric_optimized.sh "$@"
