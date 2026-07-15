#!/usr/bin/env bash
# Comparison: first learn a stable E_geo geometry teacher, then align E_full.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export GPUS="${GPUS:-4}"
export OUTPUT="${OUTPUT:-exp/linear_voxel_dual_alignment_hdr_staged_geo_v10_gpu4}"
export TRAIN_MODULE="paired_token_reliability.train_linear_voxel_dual_alignment_hdr_staged_geo"
export RUN_EVAL=0
export EPOCHS_A="${EPOCHS_A:-3}"
export EPOCHS_B="${EPOCHS_B:-9}"
export EPOCHS_C=0
export PRETRAINED="${PRETRAINED:-ckpt/model.pt}"

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
  fi
elif [[ "${RESUME}" != "none" && "${RESUME}" != "off" && -n "${RESUME}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}"); export APPEND_TRAIN_LOG=1
fi

bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --pair-mode anchor --point-weight 1.0 --decomposition-weight 0.0 \
  --no-pair-consistency --event-normal-weight 1.0 \
  --depth-event-normal-weight 0.5 --update-weight 0.001 \
  "model.depth_log_scale_limit=2.0" \
  "model.alignment_confidence_tau=0.10" \
  "model.hdr_token_bottleneck=256" \
  "model.hdr_warmup_steps=1000" \
  "model.normal_refine_iterations=1" \
  "model.normal_refine_step_limit=0.05" \
  "model.depth_update_scale=0.50" \
  "model.point_update_scale=0.10" \
  "model.support_dilation_kernel=5" \
  "${RESUME_ARGS[@]}" "$@"
