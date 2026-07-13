#!/usr/bin/env bash
# Metric-only pipeline: no normal/EN/DN loss; depth and point decoder tuning,
# with a ±100% pixel-depth correction range. Runs only on physical GPU 5.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
export GPUS=5
export OUTPUT="${OUTPUT:-exp/linear_voxel_metric_only_100pct_gpu5}"
export DEPTH_UPDATE_SCALE=1.00
bash paired_token_reliability/run_linear_voxel_depth_metric_optimized.sh "$@"
