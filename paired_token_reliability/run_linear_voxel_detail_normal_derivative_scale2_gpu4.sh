#!/usr/bin/env bash
# Scale-modified experiment (no visualization changes):
#   - event relative-depth update range: 1.0 -> 2.0
#   - update magnitude/TV regularization: 0.01 -> 0.001
# Keeps RGB metric-scale calibration, direct detail residual supervision, and
# event-normal derivative-only supervision unchanged.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export OUTPUT="${OUTPUT:-exp/linear_voxel_detail_normal_derivative_scale2_gpu4}"

bash paired_token_reliability/run_linear_voxel_detail_normal_derivative_gpu4.sh \
  --update-weight 0.001 \
  "model.depth_update_scale=2.0" \
  "$@"
