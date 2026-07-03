#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_GROUPS="${GPU_GROUPS:-0,3 5,6 7}" \
bash paper_main_ablation/run_train_gpus_234567.sh "$@"
