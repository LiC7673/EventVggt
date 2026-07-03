#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash paper_main_ablation/run_train_gpus_03567.sh "$@"
GPU="${EVAL_GPU:-7}" bash paper_main_ablation/run_eval_heldout.sh
