#!/usr/bin/env bash
# Old route's absolute event normal + strict normal-to-depth bottleneck.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export GPUS="${GPUS:-4}"
export OUTPUT="${OUTPUT:-exp/linear_voxel_absolute_normal_bottleneck_gpu4}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_absolute_normal_bottleneck"
export RUN_EVAL=0

RESUME="${RESUME:-auto}"
RESUME_ARGS=()
if [[ "${RESUME}" == "auto" ]]; then
  LATEST=""
  for CANDIDATE in "${OUTPUT}"/checkpoint-*-last.pth; do
    [[ -f "${CANDIDATE}" ]] || continue
    if [[ -z "${LATEST}" || "${CANDIDATE}" -nt "${LATEST}" ]]; then LATEST="${CANDIDATE}"; fi
  done
  if [[ -n "${LATEST}" ]]; then
    RESUME_ARGS=(--resume "${LATEST}"); export APPEND_TRAIN_LOG=1
    echo "[resume] auto-selected ${LATEST}"
  else
    echo "[resume] no absolute-normal-bottleneck checkpoint found; starting fresh"
  fi
elif [[ "${RESUME}" != "none" && "${RESUME}" != "off" && -n "${RESUME}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}"); export APPEND_TRAIN_LOG=1
  echo "[resume] using explicit checkpoint ${RESUME}"
fi

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --point-weight 0.0 --decomposition-weight 1.0 --pair-weight 0.10 \
  --event-normal-weight 1.0 --depth-event-normal-weight 0.5 \
  --update-weight 0.001 \
  "model.depth_update_scale=0.50" \
  "model.depth_log_scale_limit=2.0" \
  "model.event_residual_target_limit=0.50" \
  "model.scale_warmup_steps=1000" \
  "model.normal_bottleneck_warmup_steps=2000" \
  "model.event_min_pixel_mass=0.10" \
  "model.support_dilation_kernel=5" \
  "model.normal_refine_iterations=3" \
  "model.normal_refine_step_limit=0.05" \
  "${RESUME_ARGS[@]}" "$@"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m paired_token_reliability.evaluate_linear_voxel_absolute_normal_bottleneck \
  --checkpoint "${OUTPUT}/checkpoint-best.pth" \
  --output-dir "${OUTPUT}/test_all_exposures" \
  --initial-scene-idx "${TEST_INITIAL_SCENE_IDX:-12}" --active-scene-count 3 \
  --test-frame-count 5 --window-stride 1 --num-views 1 --visualize-every 1 \
  --event-resize-method voxel_linear_time --event-resize-bins 5 \
  --num-workers "${NUM_WORKERS:-2}" 2>&1 | tee "${OUTPUT}/logs/evaluate_all_exposures.log"
