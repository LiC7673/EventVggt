#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="${ABLATION_ROOT:-exp_f/latest_three_strategy_ablation_3epoch}/test_launcher_logs"
mkdir -p "${LOG_ROOT}"

CUDA_VISIBLE_DEVICES=0 bash paired_token_reliability/run_test_ablation_noisy_event_only_gpu0.sh \
 > "${LOG_ROOT}/noisy_event_only.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 bash paired_token_reliability/run_test_ablation_multi_ldr_only_gpu1.sh \
 > "${LOG_ROOT}/multi_ldr_only.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=2 bash paired_token_reliability/run_test_ablation_without_refiner_normal_gpu2.sh \
 > "${LOG_ROOT}/without_refiner_normal.log" 2>&1 & p2=$!

echo "test PIDs: noisy=${p0} multi=${p1} no-normal=${p2}"
wait "${p0}"; wait "${p1}"; wait "${p2}"
echo "All three ablation evaluations completed."
