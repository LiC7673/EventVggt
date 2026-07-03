#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[compat] running the paper-facing four-scene, four-LDR evaluation"
bash paper_main_ablation/run_ldr_scene_eval_gpus_234567.sh "$@"
