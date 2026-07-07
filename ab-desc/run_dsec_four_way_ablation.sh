#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
PRETRAINED="${PRETRAINED:-${ROOT_DIR}/ckpt/model.pt}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/ab-desc/results/${RUN_ID}}"
RGB_GPUS="${RGB_GPUS:-2,3}"
EVENT_GPUS="${EVENT_GPUS:-4,5}"
FULL_GPUS="${FULL_GPUS:-6,7}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_VIEWS="${NUM_VIEWS:-4}"
CLIP_STRIDE="${CLIP_STRIDE:-4}"
VISUAL_BATCHES="${VISUAL_BATCHES:-2}"
PORT_BASE="${PORT_BASE:-29740}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
EXTRA_ARGS=("$@")

mkdir -p "${RUN_ROOT}"

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "[error] pretrained checkpoint does not exist: ${PRETRAINED}" >&2
  exit 2
fi
if ! python -c 'import hdf5plugin, h5py' >/dev/null 2>&1; then
  echo "[error] hdf5plugin is unavailable in the active Python environment." >&2
  echo "        Run: python -m pip install hdf5plugin" >&2
  exit 2
fi
if [[ -n "${HDF5_PLUGIN_PATH:-}" && ! -d "${HDF5_PLUGIN_PATH}" ]]; then
  echo "[warning] unsetting invalid HDF5_PLUGIN_PATH=${HDF5_PLUGIN_PATH}"
  unset HDF5_PLUGIN_PATH
fi

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  python -m paper_main_ablation.inspect_dsec_vggt \
    --root "${DSEC_ROOT}" --output "${RUN_ROOT}/layout_report.json"
  python -m dsec_exp.check_dsec_loader \
    --root "${DSEC_ROOT}" --split train --num-views "${NUM_VIEWS}" \
    --output "${RUN_ROOT}/loader_check"
fi

evaluate_checkpoint() {
  local variant="$1"
  local checkpoint="$2"
  local gpu="$3"
  local approach="$4"
  local out_dir="${RUN_ROOT}/${variant}/heldout_test"
  mkdir -p "${out_dir}"
  echo "[eval] ${variant} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m dsec_exp.evaluate_dsec \
    --checkpoint "${checkpoint}" \
    --root "${DSEC_ROOT}" \
    --approach "${approach}" \
    --output-dir "${out_dir}" \
    --num-views "${NUM_VIEWS}" \
    --clip-stride "${CLIP_STRIDE}" \
    --num-workers "${NUM_WORKERS}" \
    --visual-batches "${VISUAL_BATCHES}" \
    > "${RUN_ROOT}/${variant}/eval.log" 2>&1
}

train_and_evaluate() {
  local variant="$1"
  local gpu_group="$2"
  local port="$3"
  local first_gpu="${gpu_group%%,*}"
  local out_dir="${RUN_ROOT}/${variant}"
  mkdir -p "${out_dir}"

  echo "[train] ${variant} on physical GPUs ${gpu_group}"
  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${gpu_group}" accelerate launch \
    --multi_gpu --num_processes 2 --num_machines 1 \
    --main_process_port "${port}" --mixed_precision bf16 --dynamo_backend no \
    "${ROOT_DIR}/ab-desc/finetune_dsec_ablation.py" \
    +ablation_variant="${variant}" \
    approach="${variant}" \
    pretrained="${PRETRAINED}" \
    save_dir="${RUN_ROOT}" \
    exp_name="${variant}" \
    epochs="${EPOCHS}" \
    num_workers="${NUM_WORKERS}" \
    data.root="${DSEC_ROOT}" \
    data.num_views="${NUM_VIEWS}" \
    data.train_clip_stride="${CLIP_STRIDE}" \
    data.test_clip_stride="${CLIP_STRIDE}" \
    "${EXTRA_ARGS[@]}" \
    > "${out_dir}/train.log" 2>&1

  evaluate_checkpoint \
    "${variant}" "${out_dir}/checkpoint-last.pth" "${first_gpu}" auto
}

echo "[ablation] output=${RUN_ROOT}"
echo "[ablation] 1/4 pretrained RGB zero-shot"
mkdir -p "${RUN_ROOT}/rgb_pretrained"
evaluate_checkpoint rgb_pretrained "${PRETRAINED}" "${RGB_GPUS%%,*}" rgb

echo "[ablation] launching three two-GPU fine-tuning jobs"
train_and_evaluate rgb_finetune "${RGB_GPUS}" "$((PORT_BASE + 1))" &
pid_rgb=$!
train_and_evaluate event_plain "${EVENT_GPUS}" "$((PORT_BASE + 2))" &
pid_event=$!
train_and_evaluate full_img_reliability "${FULL_GPUS}" "$((PORT_BASE + 3))" &
pid_full=$!

status=0
for job in "rgb_finetune:${pid_rgb}" "event_plain:${pid_event}" "full_img_reliability:${pid_full}"; do
  name="${job%%:*}"
  pid="${job##*:}"
  if wait "${pid}"; then
    echo "[done] ${name}"
  else
    echo "[fail] ${name}; inspect ${RUN_ROOT}/${name}/train.log" >&2
    status=1
  fi
done

if (( status != 0 )); then
  exit "${status}"
fi

python "${ROOT_DIR}/ab-desc/finetune_dsec_ablation.py" --collect "${RUN_ROOT}"
echo "[done] four-way DSEC ablation: ${RUN_ROOT}"
echo "       summary: ${RUN_ROOT}/summary_metrics.csv"
echo "       scenes:  ${RUN_ROOT}/per_scene_metrics.csv"
