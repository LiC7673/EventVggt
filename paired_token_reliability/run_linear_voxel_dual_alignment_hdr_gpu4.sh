#!/usr/bin/env bash
# E_full->E_geo token alignment + (LDR token, aligned event token)->HDR token alignment.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export GPUS="${GPUS:-4}"
# v10: C_source selects events; token/normal/point use independent gates.
export OUTPUT="${OUTPUT:-exp/linear_voxel_dual_alignment_hdr_event_conditioned_adapter_v10_gpu4}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_dual_alignment_hdr"
export RUN_EVAL=0
export EPOCHS_A="${EPOCHS_A:-12}"
export EPOCHS_B=0
export EPOCHS_C=0

# Reuse the event geometry learned by the stable fusion route when available.
# This is initialization, not resume: the new dual-alignment modules are new
# parameters and will be initialized by this route.
if [[ -z "${PRETRAINED:-}" ]]; then
  FIXED_INIT="exp/linear_voxel_fusion_100pct_gpu4_fixed/checkpoint-adapter-last.pth"
  if [[ -f "${FIXED_INIT}" ]]; then
    export PRETRAINED="${FIXED_INIT}"
  else
    export PRETRAINED="ckpt/model.pt"
  fi
fi
echo "[init] pretrained=${PRETRAINED}"

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
    echo "[resume] no dual-alignment checkpoint found; starting fresh"
  fi
elif [[ "${RESUME}" != "none" && "${RESUME}" != "off" && -n "${RESUME}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}"); export APPEND_TRAIN_LOG=1
  echo "[resume] using explicit checkpoint ${RESUME}"
fi

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --pair-mode anchor \
  --point-weight 1.0 --decomposition-weight 0.0 --no-pair-consistency \
  --event-normal-weight 1.0 --depth-event-normal-weight 0.5 \
  --update-weight 0.001 \
  "model.depth_log_scale_limit=2.0" \
  "model.alignment_confidence_tau=0.10" \
  "model.hdr_token_bottleneck=256" \
  "model.hdr_warmup_steps=1000" \
  "model.normal_refine_iterations=1" \
  "model.normal_refine_step_limit=0.05" \
  "model.depth_update_scale=0.50" \
  "model.support_dilation_kernel=5" \
  "${RESUME_ARGS[@]}" "$@"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m paired_token_reliability.evaluate_linear_voxel_dual_alignment_hdr \
  --checkpoint "${OUTPUT}/checkpoint-adapter-best.pth" \
  --output-dir "${OUTPUT}/test_all_exposures" \
  --initial-scene-idx "${TEST_INITIAL_SCENE_IDX:-12}" --active-scene-count 3 \
  --test-frame-count 5 --window-stride 1 --num-views 1 --visualize-every 1 \
  --event-resize-method voxel_linear_time --event-resize-bins 5 \
  --num-workers "${NUM_WORKERS:-2}" 2>&1 | tee "${OUTPUT}/logs/evaluate_all_exposures.log"
