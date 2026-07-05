#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/checkpoint.pth}"
DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
APPROACH="${APPROACH:-auto}"
GPU="${GPU:-4}"
OUT="${OUT:-${ROOT_DIR}/dsec_exp/results/zero_shot_$(date +%Y%m%d_%H%M%S)}"

python -m paper_main_ablation.inspect_dsec_vggt --root "${DSEC_ROOT}" --output "${OUT}/layout_report.json"
CUDA_VISIBLE_DEVICES="${GPU}" python -m dsec_exp.evaluate_dsec \
  --checkpoint "${CHECKPOINT}" --root "${DSEC_ROOT}" \
  --approach "${APPROACH}" --output-dir "${OUT}"
