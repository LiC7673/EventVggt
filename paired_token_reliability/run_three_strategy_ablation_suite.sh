#!/usr/bin/env bash
# Queue four ablations on the only available GPUs: 0, 1, 2.
# GPU 0 runs two jobs sequentially; GPUs 1 and 2 each run one job.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
mkdir -p exp/ablation_launcher_logs

worker_gpu0() {
  GPUS=0 bash paired_token_reliability/run_ablation_full_gpu0.sh "$@" \
    >exp/ablation_launcher_logs/full_gpu0.log 2>&1
  GPUS=0 bash paired_token_reliability/run_ablation_no_event_normal_gpu0.sh "$@" \
    >exp/ablation_launcher_logs/no_event_normal_gpu0_after_full.log 2>&1
}

worker_gpu0 "$@" & P0=$!
GPUS=1 bash paired_token_reliability/run_ablation_no_multildr_gpu1.sh "$@" \
  >exp/ablation_launcher_logs/no_multildr_gpu1.log 2>&1 & P1=$!
GPUS=2 bash paired_token_reliability/run_ablation_no_contribution_gpu2.sh "$@" \
  >exp/ablation_launcher_logs/no_contribution_gpu2.log 2>&1 & P2=$!

trap 'kill "$P0" "$P1" "$P2" 2>/dev/null || true' INT TERM
status=0
for pid in "$P0" "$P1" "$P2"; do wait "$pid" || status=1; done
exit "$status"
