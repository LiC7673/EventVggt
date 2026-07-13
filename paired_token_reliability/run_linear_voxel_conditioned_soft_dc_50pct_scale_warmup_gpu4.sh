#!/usr/bin/env bash
# 50% Soft-DC with a true 1000-step scale-only warmup.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export GPUS="${GPUS:-4}"
export OUTPUT="${OUTPUT:-exp/linear_voxel_conditioned_soft_dc_50pct_scale_warmup_gpu4}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_conditioned_soft_dc_scale_warmup"
export RUN_EVAL=0

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --point-weight 0.0 --decomposition-weight 1.0 --pair-weight 0.10 \
  --update-weight 0.01 \
  "model.depth_update_scale=0.50" \
  "model.depth_log_scale_limit=2.0" \
  "model.event_dc_limit=0.50" \
  "model.event_residual_target_limit=0.50" \
  "model.scale_warmup_steps=1000" \
  "model.event_min_pixel_mass=0.10" \
  "model.support_dilation_kernel=3" \
  "$@"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m paired_token_reliability.evaluate_linear_voxel_conditioned_soft_dc \
  --checkpoint "${OUTPUT}/checkpoint-best.pth" \
  --output-dir "${OUTPUT}/test_all_exposures" \
  --initial-scene-idx "${TEST_INITIAL_SCENE_IDX:-12}" --active-scene-count 3 \
  --test-frame-count 5 --window-stride 1 --num-views 1 --visualize-every 1 \
  --event-resize-method voxel_linear_time --event-resize-bins 5 \
  --num-workers "${NUM_WORKERS:-2}" 2>&1 | tee "${OUTPUT}/logs/evaluate_all_exposures.log"
