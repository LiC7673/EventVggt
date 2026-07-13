#!/usr/bin/env bash
# 50% event-geometry freedom experiment.  The value is a ceiling, not a
# forced update: both predicted per-view DC and supervised relative residual
# may range to +/-0.50 while learned losses determine the actual correction.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export OUTPUT="${OUTPUT:-exp/linear_voxel_conditioned_soft_dc_50pct_gpu4}"

bash paired_token_reliability/run_linear_voxel_conditioned_soft_dc_gpu4.sh \
  "model.depth_update_scale=0.50" \
  "model.event_dc_limit=0.50" \
  "model.event_residual_target_limit=0.50" \
  "$@"
