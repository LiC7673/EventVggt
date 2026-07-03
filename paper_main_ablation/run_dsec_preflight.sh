#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
OUT="${OUT:-${ROOT_DIR}/abl_event_exp/dsec_preflight/layout_report.json}"
STRICT="${STRICT:-false}"

ARGS=()
if [[ "${STRICT}" == "true" ]]; then
  ARGS+=(--strict)
fi

python -m paper_main_ablation.inspect_dsec_vggt \
  --root "${DSEC_ROOT}" \
  --output "${OUT}" \
  "${ARGS[@]}"

echo "[done] ${OUT}"
