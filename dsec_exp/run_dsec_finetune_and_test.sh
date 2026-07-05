#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
APPROACH="${APPROACH:-full_img_reliability}"
GPUS="${GPUS:-4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PORT="${PORT:-29671}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-dsec_${APPROACH}_${RUN_ID}}"
OUT="${ROOT_DIR}/dsec_exp/results/${EXP_NAME}"

mkdir -p "${OUT}"
python -m paper_main_ablation.inspect_dsec_vggt --root "${DSEC_ROOT}" --output "${OUT}/layout_report.json"
python -m dsec_exp.check_dsec_loader --root "${DSEC_ROOT}" --split train --output "${OUT}/loader_check"

CUDA_VISIBLE_DEVICES="${GPUS}" accelerate launch \
  --multi_gpu --num_processes "${NUM_PROCESSES}" --main_process_port "${PORT}" \
  --mixed_precision bf16 --dynamo_backend no \
  -m dsec_exp.finetune_dsec \
  approach="${APPROACH}" data.root="${DSEC_ROOT}" \
  exp_name="${EXP_NAME}" save_dir="${ROOT_DIR}/dsec_exp/results" \
  "$@"

CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -m dsec_exp.evaluate_dsec \
  --checkpoint "${OUT}/checkpoint-last.pth" \
  --root "${DSEC_ROOT}" --approach "${APPROACH}" \
  --output-dir "${OUT}/heldout_test"

echo "DSEC train+test complete: ${OUT}"
