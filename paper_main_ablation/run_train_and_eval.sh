#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash paper_main_ablation/run_train_gpus_234567.sh "$@"
bash paper_main_ablation/run_ldr_scene_eval_gpus_234567.sh
