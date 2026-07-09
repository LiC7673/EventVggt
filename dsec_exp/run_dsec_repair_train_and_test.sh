#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
GPUS="${GPUS:-4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PORT="${PORT:-29691}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-dsec_repair_full_img_reliability_${RUN_ID}}"
OUT="${ROOT_DIR}/dsec_exp/results/${EXP_NAME}"
AUTO_PREPARE_ALIGN="${AUTO_PREPARE_ALIGN:-1}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"

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

python -m paper_main_ablation.inspect_dsec_vggt --root "${DSEC_ROOT}" --output "${OUT}/layout_report.json"
python -m dsec_exp.check_dsec_loader --root "${DSEC_ROOT}" --split train --output "${OUT}/loader_check"

TRAIN_LOG="${OUT}/train.log"
echo "[train] DSEC repair output: ${OUT}"
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${GPUS}" accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${PORT}" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m dsec_exp.finetune_dsec_repair \
  epochs="${EPOCHS}" \
  num_workers="${NUM_WORKERS}" \
  data.root="${DSEC_ROOT}" \
  exp_name="${EXP_NAME}" \
  save_dir="${ROOT_DIR}/dsec_exp/results" \
  "$@" 2>&1 | tee "${TRAIN_LOG}"

CHECKPOINT="${OUT}/checkpoint-last.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] checkpoint missing after DSEC repair training: ${CHECKPOINT}" >&2
  exit 4
fi

EVAL_GPU="${GPUS%%,*}"
echo "[eval] DSEC held-out test on GPU ${EVAL_GPU}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python -m dsec_exp.evaluate_dsec \
  --checkpoint "${CHECKPOINT}" \
  --root "${DSEC_ROOT}" \
  --approach full_img_reliability \
  --output-dir "${OUT}/heldout_test" \
  --visual-batches 2 \
  --visual-views 4

echo "DSEC repair train+test complete: ${OUT}"
