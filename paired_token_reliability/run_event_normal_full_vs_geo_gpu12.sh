#!/usr/bin/env bash
# GPU1: direct noisy E_full normal learning. GPU2: clean E_geo normal learning.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

EPOCHS="${EPOCHS:-3}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
mkdir -p exp/event_normal_direct_full_noisy_gpu1 exp/event_normal_direct_geo_clean_gpu2

run_source() {
  local source="$1" gpu="$2" output="$3"
  EVENT_NORMAL_SOURCE="${source}" GPUS="${gpu}" \
  OUTPUT="${output}" PRETRAINED="${PRETRAINED}" \
  TRAIN_MODULE="paired_token_reliability.train_event_normal_full_vs_geo" \
  EPOCHS_A="${EPOCHS}" EPOCHS_B=0 EPOCHS_C=0 RUN_EVAL=0 \
  bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
    --pair-mode anchor --point-weight 0.0 --normal-weight 0.0 \
    --event-normal-weight 0.0 --depth-event-normal-weight 0.0 \
    --decomposition-weight 0.0 --no-pair-consistency --no-budget \
    "model.hdr_warmup_steps=1000" "model.normal_refine_iterations=1"
}

run_source full 1 exp/event_normal_direct_full_noisy_gpu1 > exp/event_normal_direct_full_noisy_gpu1.launch.log 2>&1 &
PID_FULL=$!
run_source geo 2 exp/event_normal_direct_geo_clean_gpu2 > exp/event_normal_direct_geo_clean_gpu2.launch.log 2>&1 &
PID_GEO=$!

echo "[launched] E_full pid=${PID_FULL} gpu=1; E_geo pid=${PID_GEO} gpu=2"
wait "${PID_FULL}"
wait "${PID_GEO}"
echo "[done] full and geo event-normal source ablations completed"
