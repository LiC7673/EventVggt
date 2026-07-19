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
AUTO_PREPARE_ALIGN="${AUTO_PREPARE_ALIGN:-1}"

mkdir -p "${OUT}"

alignment_missing=()
for split in val test; do
  [[ -d "${DSEC_ROOT}/${split}" ]] || continue
  while IFS= read -r -d '' scene; do
    aligned_dir="${scene}/images/event_aligned"
    marker="${aligned_dir}/vggt_alignment.json"
    first_png="$(find "${aligned_dir}" -maxdepth 1 -type f -iname '*.png' -print -quit 2>/dev/null || true)"
    if [[ ! -f "${marker}" || -z "${first_png}" ]]; then
      alignment_missing+=("${scene}")
    fi
  done < <(find "${DSEC_ROOT}/${split}" -mindepth 1 -maxdepth 1 -type d -print0)
done

if (( ${#alignment_missing[@]} > 0 )); then
  echo "[alignment] missing event-aligned RGB in ${#alignment_missing[@]} scene(s):"
  printf '  - %s\n' "${alignment_missing[@]}"
  if [[ "${AUTO_PREPARE_ALIGN}" == "1" ]]; then
    bash "${ROOT_DIR}/dsec_exp/download_prepare_event_aligned_rgb.sh" "${DSEC_ROOT}"
  else
    echo "Run dsec_exp/download_prepare_event_aligned_rgb.sh first, or set AUTO_PREPARE_ALIGN=1." >&2
    exit 3
  fi
fi

alignment_failed=0
for scene in "${alignment_missing[@]}"; do
  aligned_dir="${scene}/images/event_aligned"
  marker="${aligned_dir}/vggt_alignment.json"
  first_png="$(find "${aligned_dir}" -maxdepth 1 -type f -iname '*.png' -print -quit 2>/dev/null || true)"
  if [[ ! -f "${marker}" || -z "${first_png}" ]]; then
    echo "[alignment failed] ${scene}" >&2
    alignment_failed=1
  fi
done
if (( alignment_failed != 0 )); then
  echo "Event-aligned RGB preparation did not produce all required outputs." >&2
  exit 3
fi

python -m paper_main_ablation.inspect_dsec_vggt --root "${DSEC_ROOT}" --output "${OUT}/layout_report.json"
python -m dsec_exp.check_dsec_loader --root "${DSEC_ROOT}" --split train --output "${OUT}/loader_check"

TRAIN_LOG="${OUT}/train.log"
echo "[train] complete output: ${TRAIN_LOG}"
if ! HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${GPUS}" accelerate launch \
  --multi_gpu --num_processes "${NUM_PROCESSES}" --main_process_port "${PORT}" \
  --mixed_precision bf16 --dynamo_backend no \
  -m dsec_exp.finetune_dsec \
  approach="${APPROACH}" data.root="${DSEC_ROOT}" \
  exp_name="${EXP_NAME}" save_dir="${ROOT_DIR}/dsec_exp/results" \
  "$@" 2>&1 | tee "${TRAIN_LOG}"; then
  echo "[error] DSEC training failed. Root traceback is in ${TRAIN_LOG}" >&2
  echo "[error] Inspect it with: grep -nE 'Traceback|Error|Exception|CUDA out of memory' '${TRAIN_LOG}'" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -m dsec_exp.evaluate_dsec \
  --checkpoint "${OUT}/checkpoint-last.pth" \
  --root "${DSEC_ROOT}" --approach "${APPROACH}" \
  --output-dir "${OUT}/heldout_test"

echo "DSEC train+test complete: ${OUT}"
