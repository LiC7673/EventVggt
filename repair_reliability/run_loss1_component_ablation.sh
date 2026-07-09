#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
TEACHER="${TEACHER:-abl_event_exp/multildr_token_strategy/paired_token_full/checkpoint-last.pth}"
OUT_ROOT="${OUT_ROOT:-abl_event_exp/loss1_component_ablation}"
LABEL_DIR="${LABEL_DIR:-${OUT_ROOT}/labels}"
MODES="${MODES:-full event geometry token}"
STAGE1_GPU="${STAGE1_GPU:-6}"
STAGE2_GPUS="${STAGE2_GPUS:-6,7}"
STAGE2_PROCESSES="${STAGE2_PROCESSES:-2}"
BASE_PORT="${BASE_PORT:-29710}"
EPOCHS_STAGE1="${EPOCHS_STAGE1:-20}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${OUT_ROOT}/logs"

if [[ ! -f "${LABEL_DIR}/manifest.json" ]]; then
  if [[ ! -f "${TEACHER}" ]]; then
    echo "[error] paired-token teacher missing: ${TEACHER}" >&2
    exit 2
  fi
  echo "[export] component labels -> ${LABEL_DIR}"
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m paired_token_reliability.export_targets \
    --teacher "${TEACHER}" \
    --output "${LABEL_DIR}" \
    --ldr-ids ev_1 ev_2 ev_5 ev_10 \
    --val-scenes 2 \
    --token-cosine-floor 0.80 \
    --dilate-kernel 3 \
    2>&1 | tee "${OUT_ROOT}/logs/export_targets.log"
else
  echo "[export] reuse ${LABEL_DIR}/manifest.json"
  first_npz="$(find "${LABEL_DIR}/targets" -maxdepth 1 -type f -name '*.npz' -print -quit 2>/dev/null || true)"
  if [[ -n "${first_npz}" ]]; then
    if ! python - "${first_npz}" <<'PY'
import sys, numpy as np
data = np.load(sys.argv[1])
missing = [k for k in ("event_support", "geometry", "token_agreement") if k not in data]
raise SystemExit(1 if missing else 0)
PY
    then
      echo "[error] existing labels do not contain component maps. Remove ${LABEL_DIR} and rerun." >&2
      exit 3
    fi
  fi
fi

summary_csv="${OUT_ROOT}/component_runs.csv"
echo "mode,stage1_dir,stage2_dir,eval_dir" > "${summary_csv}"

mode_index=0
for mode in ${MODES}; do
  mode_index=$((mode_index + 1))
  reliability_dir="${OUT_ROOT}/reliability_${mode}"
  exp_name="loss1_${mode}_repair_stage2"
  stage2_dir="${OUT_ROOT}/${exp_name}"
  port=$((BASE_PORT + mode_index))
  echo
  echo "=============================="
  echo "[mode=${mode}] Stage1 ReliabilityNet"
  echo "=============================="
  if [[ ! -f "${reliability_dir}/checkpoint-best.pth" ]]; then
    CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m paired_token_reliability.train_reliability_component \
      --manifest "${LABEL_DIR}/manifest.json" \
      --output "${reliability_dir}" \
      --target-mode "${mode}" \
      --epochs "${EPOCHS_STAGE1}" \
      --batch-size 1 \
      --num-workers "${NUM_WORKERS}" \
      --amp \
      2>&1 | tee "${OUT_ROOT}/logs/stage1_${mode}.log"
  else
    echo "[mode=${mode}] reuse ${reliability_dir}/checkpoint-best.pth"
  fi

  echo
  echo "=============================="
  echo "[mode=${mode}] Stage2 repair VGGT"
  echo "=============================="
  CUDA_VISIBLE_DEVICES="${STAGE2_GPUS}" accelerate launch \
    --multi_gpu \
    --num_processes "${STAGE2_PROCESSES}" \
    --main_process_port "${port}" \
    --gpu_ids all \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m repair_reliability.finetune_stage2_repair \
    exp_name="${exp_name}" \
    epochs="${EPOCHS_STAGE2}" \
    num_workers="${NUM_WORKERS}" \
    data.root="${DATA_ROOT}" \
    data.num_views=4 \
    ++data.train_initial_scene_idx=0 \
    ++data.train_scene_count=12 \
    ++data.train_holdout_frame_count=0 \
    ++data.test_initial_scene_idx=12 \
    ++data.test_scene_count=4 \
    ++data.heldout_test_frame_count=120 \
    ++model.reliability_checkpoint="${reliability_dir}/checkpoint-best.pth" \
    ++model.reliability_gate_floor=0.20 \
    ++model.repair_reliability_threshold=0.45 \
    ++model.repair_reliability_temperature=0.18 \
    ++model.repair_reliability_top_fraction=0.80 \
    ++model.repair_event_support_dilate_kernel=5 \
    ++model.repair_event_support_floor=0.25 \
    ++model.repair_residual_gain=2.0 \
    ++model.repair_output_abs_limit=0.06 \
    ++loss.stage2_residual_target_weight=2.0 \
    ++loss.stage2_residual_gradient_weight=3.0 \
    ++loss.stage2_target_reliability_floor=0.10 \
    ++loss.stage2_target_abs_limit=0.06 \
    ++vis.save_every_steps=3000 \
    2>&1 | tee "${OUT_ROOT}/logs/stage2_${mode}.log"

  checkpoint="${stage2_dir}/checkpoint-last.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] missing Stage2 checkpoint for mode=${mode}: ${checkpoint}" >&2
    exit 4
  fi

  echo
  echo "=============================="
  echo "[mode=${mode}] held-out causal eval"
  echo "=============================="
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m repair_reliability.evaluate_stage2_repair \
    --checkpoint "${checkpoint}" \
    --reliability-checkpoint "${reliability_dir}/checkpoint-best.pth" \
    --output-dir "${stage2_dir}/heldout_eval" \
    --root "${DATA_ROOT}" \
    --initial-scene-idx 12 \
    --active-scene-count 4 \
    --test-frame-count 120 \
    --window-stride 4 \
    --num-views 4 \
    --event-resize-bins 10 \
    --amp bf16 \
    2>&1 | tee "${OUT_ROOT}/logs/eval_${mode}.log"

  echo "${mode},${reliability_dir},${stage2_dir},${stage2_dir}/heldout_eval" >> "${summary_csv}"
done

echo
echo "done. summary: ${summary_csv}"
