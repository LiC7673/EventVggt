#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/paper_module_ablation_extrel_normal}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
EVAL_GPUS="${EVAL_GPUS:-2 3 4 5}"

echo "[train] A5 Full with frozen external ReliabilityUNet on GPUs ${TRAIN_GPUS}"
VARIANTS=a5_full \
GPU_GROUPS="${TRAIN_GPUS}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
SKIP_EXISTING=false \
bash paper_main_ablation/run_train_gpus_234567.sh "$@"

echo "[eval] A5 Full on held-out scenes at LDR 1/2/5/10"
EVAL_VARIANTS="a5_full" \
GPU_POOL="${EVAL_GPUS}" \
CHECKPOINT_ROOT="${OUTPUT_ROOT}" \
bash paper_main_ablation/run_ldr_scene_eval_gpus_234567.sh

echo "[done] ${OUTPUT_ROOT}/test_4scenes_ldr_1_2_5_10"
