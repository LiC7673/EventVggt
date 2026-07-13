#!/usr/bin/env bash
# One-click parallel launch:
#   GPU 4 -> full fusion 100%
#   GPU 5 -> metric-only 100%
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
mkdir -p exp/launcher_logs
OUTPUT="${FUSION_OUTPUT:-exp/linear_voxel_fusion_100pct_gpu4_fixed}" \
bash paired_token_reliability/run_linear_voxel_fusion_100pct_gpu4.sh "$@" \
  > exp/launcher_logs/fusion_100pct_gpu4.log 2>&1 & P4=$!
OUTPUT="${METRIC_OUTPUT:-exp/linear_voxel_metric_only_100pct_gpu5}" \
bash paired_token_reliability/run_linear_voxel_metric_only_100pct_gpu5.sh "$@" \
  > exp/launcher_logs/metric_only_100pct_gpu5.log 2>&1 & P5=$!
trap 'kill "$P4" "$P5" 2>/dev/null || true' INT TERM
status=0
wait "$P4" || status=1
wait "$P5" || status=1
exit "$status"
