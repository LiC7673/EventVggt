#!/usr/bin/env bash
# Parallel four-scene/all-exposure evaluation of the three latest ablations.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ABLATION_ROOT="${ABLATION_ROOT:-exp_f/latest_three_strategy_ablation_3epoch_v2_rgb_routes}"
LOG_ROOT="${ABLATION_ROOT}/test_launcher_logs_gpu456"
mkdir -p "${LOG_ROOT}"
export ABLATION_ROOT

echo "[GPU 4] noisy_event_only"
CUDA_VISIBLE_DEVICES=4 \
  bash paired_token_reliability/run_test_ablation_noisy_event_only_gpu0.sh \
  > "${LOG_ROOT}/noisy_event_only_gpu4.log" 2>&1 &
pid_noisy=$!

echo "[GPU 5] multi_ldr_only"
CUDA_VISIBLE_DEVICES=5 \
  bash paired_token_reliability/run_test_ablation_multi_ldr_only_gpu1.sh \
  > "${LOG_ROOT}/multi_ldr_only_gpu5.log" 2>&1 &
pid_multi=$!

echo "[GPU 6] without_refiner_normal"
CUDA_VISIBLE_DEVICES=6 \
  bash paired_token_reliability/run_test_ablation_without_refiner_normal_gpu2.sh \
  > "${LOG_ROOT}/without_refiner_normal_gpu6.log" 2>&1 &
pid_without=$!

echo "PIDs: noisy=${pid_noisy} multi=${pid_multi} without=${pid_without}"

status=0
wait "${pid_noisy}" || { echo "noisy_event_only failed; inspect ${LOG_ROOT}/noisy_event_only_gpu4.log" >&2; status=1; }
wait "${pid_multi}" || { echo "multi_ldr_only failed; inspect ${LOG_ROOT}/multi_ldr_only_gpu5.log" >&2; status=1; }
wait "${pid_without}" || { echo "without_refiner_normal failed; inspect ${LOG_ROOT}/without_refiner_normal_gpu6.log" >&2; status=1; }

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

echo "All three four-scene/all-EV ablation evaluations completed."
echo "Results:"
echo "  ${ABLATION_ROOT}/noisy_event_only/test_four_scenes_all_ev"
echo "  ${ABLATION_ROOT}/multi_ldr_only/test_four_scenes_all_ev"
echo "  ${ABLATION_ROOT}/without_refiner_normal/test_four_scenes_all_ev"
echo "Logs: ${LOG_ROOT}"
