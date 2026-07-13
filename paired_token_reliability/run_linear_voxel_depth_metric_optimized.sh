#!/usr/bin/env bash
# Depth-metric-oriented preset for the linear-time voxel model.
# Target validation/test metrics: AbsRel ↓, RMSE-log ↓, delta1 ↑,
# and depth-derived normal mean angle ↓. This is a separate paper experiment.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export EXP_NAME="${EXP_NAME:-linear_voxel_depth_metric_optimized}"
export OUTPUT="${OUTPUT:-exp/${EXP_NAME}}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_depth_point"

# Pure metric preset: allow up to ±100% relative correction. tanh still keeps
# the update finite, while the held-out depth metric is the selection target.
export DEPTH_UPDATE_SCALE="${DEPTH_UPDATE_SCALE:-1.00}"

# Retain more useful structure than tau=3 ms; linear temporal splatting still
# preserves ordering, while the larger tau is less destructive for sparse data.
export EVENT_DECAY_TAU="${EVENT_DECAY_TAU:-0.005}"

# More capacity is allocated only to the full-resolution event/pixel branch.
export PIXEL_HIDDEN="${PIXEL_HIDDEN:-64}"
export EPOCHS_A="${EPOCHS_A:-2}"
export EPOCHS_B="${EPOCHS_B:-14}"
export EPOCHS_C="${EPOCHS_C:-1}"

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --lr "${LR:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-5}" \
  --point-weight 0.25 \
  --normal-weight 0.0 \
  --event-normal-weight 0.0 \
  --depth-event-normal-weight 0.0 \
  --depth-gradient-weight 0.50 \
  --depth-curvature-weight 0.20 \
  --patch-grid-weight 0.0 \
  --update-weight 0.01 \
  --pair-weight 0.02 \
  --decomposition-weight 0.05 \
  --geometry-rank-weight 0.0 \
  --clip-grad 2.0 \
  "model.signed_pixel_hidden=${PIXEL_HIDDEN}" \
  "model.depth_update_scale=${DEPTH_UPDATE_SCALE}" \
  "model.event_decay_tau=${EVENT_DECAY_TAU}" \
  "model.support_dilation_kernel=${SUPPORT_KERNEL:-7}" \
  "$@"
