#!/usr/bin/env bash
# GPU1: oracle geometry events, C=1. GPU2: raw full events, C=1.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
mkdir -p exp/launcher_logs

GPUS=1 SOURCE=geo EXPERIMENT=no_contribution_geo_event_gpu1 RESUME="${RESUME:-auto}" \
bash paired_token_reliability/run_no_contribution_event_source_single.sh \
  > exp/launcher_logs/no_contribution_geo_event_gpu1.log 2>&1 &
PID_GEO=$!

GPUS=2 SOURCE=full EXPERIMENT=no_contribution_full_event_gpu2 RESUME="${RESUME:-auto}" \
bash paired_token_reliability/run_no_contribution_event_source_single.sh \
  > exp/launcher_logs/no_contribution_full_event_gpu2.log 2>&1 &
PID_FULL=$!

echo "[launched] E_geo C=1 gpu1 pid=${PID_GEO}"
echo "[launched] E_full C=1 gpu2 pid=${PID_FULL}"
STATUS=0
wait "${PID_GEO}" || STATUS=1
wait "${PID_FULL}" || STATUS=1
exit "${STATUS}"
