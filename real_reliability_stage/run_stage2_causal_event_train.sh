#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPUS="${GPUS:-6,7}"
EXP_NAME="${EXP_NAME:-stage2_causal_event_reliability_train12_test4}"
CAUSAL_SUPPORT_THRESHOLD="${CAUSAL_SUPPORT_THRESHOLD:-0.01}"
CAUSAL_DILATE_KERNEL="${CAUSAL_DILATE_KERNEL:-5}"
CAUSAL_BLUR_KERNEL="${CAUSAL_BLUR_KERNEL:-3}"

echo "[causal train] zero event is constrained to produce zero residual"
echo "[causal train] support threshold=${CAUSAL_SUPPORT_THRESHOLD}, dilate=${CAUSAL_DILATE_KERNEL}, blur=${CAUSAL_BLUR_KERNEL}"

GPUS="${GPUS}" \
EXP_NAME="${EXP_NAME}" \
bash real_reliability_stage/run_stage2_vggt_12train_4test.sh \
  +model.causal_output_gate=true \
  +model.causal_support_threshold="${CAUSAL_SUPPORT_THRESHOLD}" \
  +model.causal_support_dilate_kernel="${CAUSAL_DILATE_KERNEL}" \
  +model.causal_support_blur_kernel="${CAUSAL_BLUR_KERNEL}" \
  "$@"

