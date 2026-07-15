#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

SOURCE="${SOURCE:-full}"
if [[ "${SOURCE}" != "full" && "${SOURCE}" != "geo" ]]; then
  echo "SOURCE must be full or geo, got ${SOURCE}" >&2; exit 2
fi
export EVENT_SOURCE_ABLATION="${SOURCE}"
export GPUS="${GPUS:-2}"
EXPERIMENT="${EXPERIMENT:-no_contribution_${SOURCE}_event_gpu${GPUS}}"
export OUTPUT="${OUTPUT:-exp/${EXPERIMENT}}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_no_contribution_source_ablation"
export RUN_EVAL=0 EPOCHS_A="${EPOCHS_A:-12}" EPOCHS_B=0 EPOCHS_C=0

if [[ -z "${PRETRAINED:-}" ]]; then
  CANDIDATE="exp/linear_voxel_fusion_100pct_gpu4_fixed/checkpoint-adapter-last.pth"
  if [[ -f "${CANDIDATE}" ]]; then export PRETRAINED="${CANDIDATE}"; else export PRETRAINED="ckpt/model.pt"; fi
fi

RESUME="${RESUME:-auto}"; RESUME_ARGS=()
if [[ "${RESUME}" == "auto" ]]; then
  LATEST=""
  for CANDIDATE in "${OUTPUT}"/checkpoint-*-last.pth; do
    [[ -f "${CANDIDATE}" ]] || continue
    if [[ -z "${LATEST}" || "${CANDIDATE}" -nt "${LATEST}" ]]; then LATEST="${CANDIDATE}"; fi
  done
  if [[ -n "${LATEST}" ]]; then RESUME_ARGS=(--resume "${LATEST}"); export APPEND_TRAIN_LOG=1; fi
elif [[ "${RESUME}" != "none" && "${RESUME}" != "off" && -n "${RESUME}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}"); export APPEND_TRAIN_LOG=1
fi

echo "[no-C source ablation] gpu=${GPUS} source=E_${SOURCE} output=${OUTPUT}"
bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --pair-mode anchor --point-weight 1.0 --decomposition-weight 0.0 \
  --no-pair-consistency --no-budget --event-normal-weight 1.0 \
  --depth-event-normal-weight 0.5 --update-weight 0.001 \
  "model.event_source_ablation=${SOURCE}" \
  "model.depth_log_scale_limit=2.0" \
  "model.hdr_warmup_steps=1000" \
  "model.normal_refine_iterations=3" \
  "model.normal_refine_step_limit=0.05" \
  "model.depth_update_scale=0.50" \
  "model.point_update_scale=0.10" \
  "${RESUME_ARGS[@]}" "$@"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m paired_token_reliability.evaluate_linear_voxel_no_contribution_source_ablation \
  --checkpoint "${OUTPUT}/checkpoint-adapter-best.pth" \
  --output-dir "${OUTPUT}/test_all_exposures" \
  --initial-scene-idx "${TEST_INITIAL_SCENE_IDX:-12}" --active-scene-count 3 \
  --test-frame-count 5 --window-stride 1 --num-views 1 --visualize-every 1 \
  --event-resize-method voxel_linear_time --event-resize-bins 5 \
  --num-workers "${NUM_WORKERS:-2}" 2>&1 | tee "${OUTPUT}/logs/evaluate_all_exposures.log"
