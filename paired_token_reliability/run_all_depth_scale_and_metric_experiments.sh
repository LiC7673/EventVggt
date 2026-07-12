#!/usr/bin/env bash
# Launch 15/30/50% normal-pipeline ablations on GPUs 0/1/2 and the depth-only
# metric preset with DDP on GPUs 4/5. All receive the same CLI overrides.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
mkdir -p exp/launcher_logs
bash paired_token_reliability/run_linear_voxel_normal_15pct_gpu0.sh "$@" \
  > exp/launcher_logs/normal_15pct_gpu0.log 2>&1 & P0=$!
bash paired_token_reliability/run_linear_voxel_normal_30pct_gpu1.sh "$@" \
  > exp/launcher_logs/normal_30pct_gpu1.log 2>&1 & P1=$!
bash paired_token_reliability/run_linear_voxel_normal_50pct_gpu2.sh "$@" \
  > exp/launcher_logs/normal_50pct_gpu2.log 2>&1 & P2=$!
bash paired_token_reliability/run_linear_voxel_depth_only_gpu45.sh "$@" \
  > exp/launcher_logs/depth_only_gpu45.log 2>&1 & P45=$!
trap 'kill "$P0" "$P1" "$P2" "$P45" 2>/dev/null || true' INT TERM
status=0
for pid in "$P0" "$P1" "$P2" "$P45"; do wait "$pid" || status=1; done
exit "$status"
