#!/usr/bin/env bash
# GPU1: full method; GPU2: no event-source attribution; GPU6: no missing-residual supervision.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
mkdir -p exp/launcher_logs

GPUS=1 EXPERIMENT=attribution_residual_full_gpu1 \
ABLATE_ATTRIBUTION=false ABLATE_RESIDUAL=false RESUME="${RESUME:-auto}" \
bash paired_token_reliability/run_linear_voxel_attribution_residual_single.sh \
  > exp/launcher_logs/attribution_residual_full_gpu1.log 2>&1 &
PID_FULL=$!

GPUS=2 EXPERIMENT=attribution_residual_no_attribution_gpu2 \
ABLATE_ATTRIBUTION=true ABLATE_RESIDUAL=false RESUME="${RESUME:-auto}" \
bash paired_token_reliability/run_linear_voxel_attribution_residual_single.sh \
  > exp/launcher_logs/attribution_residual_no_attribution_gpu2.log 2>&1 &
PID_NO_C=$!

GPUS=6 EXPERIMENT=attribution_residual_no_missing_residual_gpu6 \
ABLATE_ATTRIBUTION=false ABLATE_RESIDUAL=true RESUME="${RESUME:-auto}" \
bash paired_token_reliability/run_linear_voxel_attribution_residual_single.sh \
  > exp/launcher_logs/attribution_residual_no_missing_residual_gpu6.log 2>&1 &
PID_NO_RES=$!

echo "[launched] full gpu1 pid=${PID_FULL}"
echo "[launched] no-attribution gpu2 pid=${PID_NO_C}"
echo "[launched] no-missing-residual gpu6 pid=${PID_NO_RES}"

STATUS=0
wait "${PID_FULL}" || STATUS=1
wait "${PID_NO_C}" || STATUS=1
wait "${PID_NO_RES}" || STATUS=1
exit "${STATUS}"
