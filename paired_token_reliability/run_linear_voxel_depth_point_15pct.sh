#!/usr/bin/env bash
# Independent 15% depth-update version; depth and point-map objectives only.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
export EXP_NAME="${EXP_NAME:-linear_voxel_depth_point_15pct}"
export OUTPUT="${OUTPUT:-exp/${EXP_NAME}}"
export DEPTH_UPDATE_SCALE=0.15
bash paired_token_reliability/run_linear_voxel_depth_metric_optimized.sh "$@"
