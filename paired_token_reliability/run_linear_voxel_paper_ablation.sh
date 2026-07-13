#!/usr/bin/env bash
# Paper ablation entry: one corrected fusion model, followed by matched
# counterfactual evaluation. The experiment is named by method components,
# not by the internal safety bound.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export GPUS="${GPUS:-0}"
export OUTPUT="${OUTPUT:-exp/linear_voxel_paper_ablation}"
export RUN_EVAL=1

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --point-weight 0.0 \
  --decomposition-weight 1.0 \
  --pair-weight 0.10 \
  --geometry-rank-weight 0.10 \
  --normal-weight 0.75 \
  --event-normal-weight 0.50 \
  --depth-event-normal-weight 0.50 \
  "model.depth_update_scale=${DEPTH_UPDATE_SCALE:-1.0}" \
  "$@"
