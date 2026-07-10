#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
TEACHER="${TEACHER:-abl_event_exp/multildr_token_strategy/paired_token_full/checkpoint-last.pth}"
OUT_ROOT="${OUT_ROOT:-abl_event_exp/loss1_factor_ablation_v2}"
LABEL_DIR="${LABEL_DIR:-${OUT_ROOT}/labels}"
# Paper Loss1 factorial ablation for R*=E*G*T:
#   full; remove one factor; retain only one factor.
ABLATIONS="${ABLATIONS:-full drop_event drop_geometry drop_token only_event only_geometry only_token}"
STAGE1_GPU="${STAGE1_GPU:-6}"
STAGE2_GPUS="${STAGE2_GPUS:-6,7}"
STAGE2_PROCESSES="${STAGE2_PROCESSES:-2}"
BASE_PORT="${BASE_PORT:-29710}"
EPOCHS_STAGE1="${EPOCHS_STAGE1:-20}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
MIN_START_ID="${MIN_START_ID:-2}"
TARGET_DILATE_KERNEL="${TARGET_DILATE_KERNEL:-3}"
LABEL_SCHEMA_VERSION="${LABEL_SCHEMA_VERSION:-2}"
TOKEN_SOURCE="${TOKEN_SOURCE:-raw}"
TOKEN_AGREEMENT_MODE="${TOKEN_AGREEMENT_MODE:-robust_quantile}"
TOKEN_QUANTILE_LOW="${TOKEN_QUANTILE_LOW:-0.05}"
TOKEN_QUANTILE_HIGH="${TOKEN_QUANTILE_HIGH:-0.95}"
STAGE2_TAG="${STAGE2_TAG:-flatguard_v3}"
NORMAL_WEIGHT="${NORMAL_WEIGHT:-0.05}"
FLAT_NORMAL_WEIGHT="${FLAT_NORMAL_WEIGHT:-0.50}"
POSE_TRANSLATION_SCALE="${POSE_TRANSLATION_SCALE:-0.01}"
POSE_QUATERNION_SCALE="${POSE_QUATERNION_SCALE:-0.01}"
CONTRIBUTION_DEPTH_GUARD_WEIGHT="${CONTRIBUTION_DEPTH_GUARD_WEIGHT:-1.0}"
CONTRIBUTION_NORMAL_GUARD_WEIGHT="${CONTRIBUTION_NORMAL_GUARD_WEIGHT:-1.0}"
POSE_WEIGHT="${POSE_WEIGHT:-1.0}"
DETAIL_NORMAL_WEIGHT="${DETAIL_NORMAL_WEIGHT:-0.0}"
DETAIL_HF_WEIGHT="${DETAIL_HF_WEIGHT:-0.0}"
DETAIL_GRAD_WEIGHT="${DETAIL_GRAD_WEIGHT:-0.0}"
GATE_FLOOR="${GATE_FLOOR:-0.05}"
EVENT_SUPPORT_FLOOR="${EVENT_SUPPORT_FLOOR:-0.05}"
STAGE2_MODULE="${STAGE2_MODULE:-repair_reliability.finetune_stage2_repair}"
EVAL_MODULE="${EVAL_MODULE:-repair_reliability.evaluate_stage2_repair}"

target_mode_for() {
  case "$1" in
    full)          echo "full" ;;
    drop_event)    echo "geometry_token" ;;
    drop_geometry) echo "event_token" ;;
    drop_token)    echo "event_geometry" ;;
    only_event)    echo "event" ;;
    only_geometry) echo "geometry" ;;
    only_token)    echo "token" ;;
    *) echo "[error] unknown ablation: $1" >&2; return 1 ;;
  esac
}

factors_for() {
  case "$1" in
    full)          echo "E*G*T" ;;
    drop_event)    echo "G*T" ;;
    drop_geometry) echo "E*T" ;;
    drop_token)    echo "E*G" ;;
    only_event)    echo "E" ;;
    only_geometry) echo "G" ;;
    only_token)    echo "T" ;;
  esac
}

validate_stage1_checkpoint() {
  python - "$1" "$2" "${TARGET_DILATE_KERNEL}" "${SEED}" <<'PY'
import sys, torch
path, expected_mode, expected_dilate, expected_seed = sys.argv[1:]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
args = checkpoint.get("args", {})
actual = (
    str(checkpoint.get("target_mode", args.get("target_mode", ""))),
    str(args.get("weight_mode", "")),
    int(args.get("target_dilate_kernel", -1)),
    int(args.get("seed", -1)),
)
expected = (expected_mode, "common_valid", int(expected_dilate), int(expected_seed))
if actual != expected:
    raise SystemExit(f"Stage1 checkpoint config mismatch: actual={actual}, expected={expected}")
PY
}

validate_stage2_binding() {
  python - "$1" "$2" "$3" <<'PY'
import os, sys, torch
checkpoint_path, expected_stage1, expected_output = sys.argv[1:]
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
cfg = checkpoint.get("cfg", {})
model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
actual_stage1 = model_cfg.get("reliability_checkpoint", "")
actual_output = cfg.get("output_dir", "") if isinstance(cfg, dict) else ""
same_stage1 = os.path.realpath(actual_stage1) == os.path.realpath(expected_stage1)
same_output = os.path.realpath(actual_output) == os.path.realpath(expected_output)
loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
expected_values = {
    "reliability_gate_floor": (model_cfg.get("reliability_gate_floor"), float(os.environ.get("GATE_FLOOR", "0.05"))),
    "repair_reliability_threshold": (model_cfg.get("repair_reliability_threshold"), 0.58),
    "repair_reliability_temperature": (model_cfg.get("repair_reliability_temperature"), 0.12),
    "repair_reliability_top_fraction": (model_cfg.get("repair_reliability_top_fraction"), 0.35),
    "repair_event_support_floor": (model_cfg.get("repair_event_support_floor"), float(os.environ.get("EVENT_SUPPORT_FLOOR", "0.05"))),
    "repair_residual_gain": (model_cfg.get("repair_residual_gain"), 1.6),
    "repair_pose_translation_scale": (model_cfg.get("repair_pose_translation_scale"), float(os.environ.get("POSE_TRANSLATION_SCALE", "0.01"))),
    "repair_pose_quaternion_scale": (model_cfg.get("repair_pose_quaternion_scale"), float(os.environ.get("POSE_QUATERNION_SCALE", "0.01"))),
    "repair_train_coarse_heads": (model_cfg.get("repair_train_coarse_heads"), 0.0),
    "normal_weight": (loss_cfg.get("normal_weight"), float(os.environ.get("NORMAL_WEIGHT", "0.05"))),
    "pose_weight": (loss_cfg.get("pose_weight"), float(os.environ.get("POSE_WEIGHT", "1.0"))),
    "detail_gt_normal_weight": (loss_cfg.get("detail_gt_normal_weight"), float(os.environ.get("DETAIL_NORMAL_WEIGHT", "0.0"))),
    "detail_gt_hf_weight": (loss_cfg.get("detail_gt_hf_weight"), float(os.environ.get("DETAIL_HF_WEIGHT", "0.0"))),
    "detail_gt_grad_weight": (loss_cfg.get("detail_gt_grad_weight"), float(os.environ.get("DETAIL_GRAD_WEIGHT", "0.0"))),
    "stage2_flat_normal_weight": (loss_cfg.get("stage2_flat_normal_weight"), float(os.environ.get("FLAT_NORMAL_WEIGHT", "0.50"))),
    "stage2_flat_residual_gradient_weight": (loss_cfg.get("stage2_flat_residual_gradient_weight"), 0.50),
    "stage2_contribution_depth_guard_weight": (loss_cfg.get("stage2_contribution_depth_guard_weight"), float(os.environ.get("CONTRIBUTION_DEPTH_GUARD_WEIGHT", "1.0"))),
    "stage2_contribution_normal_guard_weight": (loss_cfg.get("stage2_contribution_normal_guard_weight"), float(os.environ.get("CONTRIBUTION_NORMAL_GUARD_WEIGHT", "1.0"))),
}
mismatched_values = {
    key: {"actual": actual, "expected": expected}
    for key, (actual, expected) in expected_values.items()
    if actual is None or abs(float(actual) - expected) > 1.0e-8
}
if not same_stage1 or not same_output or mismatched_values:
    raise SystemExit(
        "Stage2 checkpoint binding mismatch:\n"
        f"  reliability actual={actual_stage1}\n"
        f"  reliability expected={os.path.realpath(expected_stage1)}\n"
        f"  output actual={actual_output}\n"
        f"  output expected={os.path.realpath(expected_output)}\n"
        f"  config mismatches={mismatched_values}"
    )
PY
}

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
    --token-source "${TOKEN_SOURCE}" \
    --token-agreement-mode "${TOKEN_AGREEMENT_MODE}" \
    --token-quantile-low "${TOKEN_QUANTILE_LOW}" \
    --token-quantile-high "${TOKEN_QUANTILE_HIGH}" \
    --dilate-kernel 3 \
    --min-start-id "${MIN_START_ID}" \
    2>&1 | tee "${OUT_ROOT}/logs/export_targets.log"
else
  echo "[export] reuse ${LABEL_DIR}/manifest.json"
  first_npz="$(find "${LABEL_DIR}/targets" -maxdepth 1 -type f -name '*.npz' -print -quit 2>/dev/null || true)"
  if [[ -n "${first_npz}" ]]; then
    if ! python - "${LABEL_DIR}/manifest.json" "${first_npz}" "${MIN_START_ID}" "${LABEL_SCHEMA_VERSION}" "${TOKEN_SOURCE}" "${TOKEN_AGREEMENT_MODE}" <<'PY'
import json, sys, numpy as np
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
data = np.load(sys.argv[2])
missing = [k for k in ("event_support", "geometry", "token_agreement") if k not in data]
wrong_start = int(manifest.get("min_start_id", -1)) != int(sys.argv[3])
wrong_schema = int(manifest.get("label_schema_version", -1)) != int(sys.argv[4])
wrong_token = manifest.get("token_source") != sys.argv[5] or manifest.get("token_agreement_mode") != sys.argv[6]
degenerate = float(manifest.get("token_agreement_mean_spatial_std", 0.0)) < 0.03
raise SystemExit(1 if missing or wrong_start or wrong_schema or wrong_token or degenerate else 0)
PY
    then
      echo "[error] existing labels are stale, incomplete, or use a different min_start_id." >&2
      echo "        Use a new LABEL_DIR (recommended) or remove ${LABEL_DIR} and rerun." >&2
      exit 3
    fi
  fi
fi

summary_csv="${OUT_ROOT}/component_runs.csv"
echo "ablation,target_mode,factors,stage1_dir,stage2_dir,checkpoint,eval_dir" > "${summary_csv}"
metrics_csv="${OUT_ROOT}/paper_metrics.csv"
echo "ablation,target_mode,factors,stage1_best_iou,AbsRel,delta1,RMSElog,normal_mean_deg,normal_11_25,ATE,event_reliability_mean,normal_gain_vs_zero,AbsRel_gain_vs_zero,full_vs_zero_depth_diff" > "${metrics_csv}"

mode_index=0
for ablation in ${ABLATIONS}; do
  mode_index=$((mode_index + 1))
  target_mode="$(target_mode_for "${ablation}")"
  factors="$(factors_for "${ablation}")"
  reliability_dir="${OUT_ROOT}/stage1_${ablation}"
  stage1_checkpoint="${reliability_dir}/checkpoint-best.pth"
  # Version the Stage2 output whenever its gate/loss semantics change so an
  # older geometry checkpoint cannot be silently reused with new Stage1 runs.
  exp_name="loss1_${ablation}_repair_stage2_${STAGE2_TAG}"
  stage2_dir="${OUT_ROOT}/${exp_name}"
  checkpoint="${stage2_dir}/checkpoint-last.pth"
  port=$((BASE_PORT + mode_index))
  echo
  echo "=============================="
  echo "[${ablation}] Stage1 ReliabilityNet target=${factors} mode=${target_mode}"
  echo "=============================="
  echo "[paths] Stage1=${stage1_checkpoint}"
  echo "[paths] Stage2=${checkpoint}"
  if [[ ! -f "${stage1_checkpoint}" ]]; then
    CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m paired_token_reliability.train_reliability_component \
      --manifest "${LABEL_DIR}/manifest.json" \
      --output "${reliability_dir}" \
      --target-mode "${target_mode}" \
      --target-dilate-kernel "${TARGET_DILATE_KERNEL}" \
      --weight-mode common_valid \
      --seed "${SEED}" \
      --epochs "${EPOCHS_STAGE1}" \
      --batch-size 1 \
      --num-workers "${NUM_WORKERS}" \
      --amp \
      2>&1 | tee "${OUT_ROOT}/logs/stage1_${ablation}.log"
  else
    echo "[${ablation}] validate and reuse ${stage1_checkpoint}"
  fi
  if ! validate_stage1_checkpoint "${stage1_checkpoint}" "${target_mode}"; then
    echo "[error] refusing stale/wrong Stage1 checkpoint for ${ablation}." >&2
    echo "        Use a new OUT_ROOT or remove only this ablation directory." >&2
    exit 5
  fi

  if [[ -f "${checkpoint}" ]]; then
    echo "[${ablation}] found existing Stage2 checkpoint; binding will be validated: ${checkpoint}"
  else
    echo
    echo "=============================="
    echo "[${ablation}] Stage2 repair VGGT"
    echo "=============================="
    CUDA_VISIBLE_DEVICES="${STAGE2_GPUS}" accelerate launch \
      --multi_gpu \
      --num_processes "${STAGE2_PROCESSES}" \
      --main_process_port "${port}" \
      --gpu_ids all \
      --mixed_precision bf16 \
      --dynamo_backend no \
      -m "${STAGE2_MODULE}" \
      exp_name="${exp_name}" \
      ++repair_save_dir="${OUT_ROOT}" \
      epochs="${EPOCHS_STAGE2}" \
      seed="${SEED}" \
      num_workers="${NUM_WORKERS}" \
      data.root="${DATA_ROOT}" \
      data.num_views=4 \
      ++data.train_initial_scene_idx=0 \
      ++data.train_scene_count=12 \
      ++data.train_holdout_frame_count=0 \
      ++data.train_min_start_id=2 \
      ++data.test_initial_scene_idx=12 \
      ++data.test_scene_count=4 \
      ++data.heldout_test_frame_count=120 \
      ++model.reliability_checkpoint="${stage1_checkpoint}" \
      ++model.reliability_gate_floor="${GATE_FLOOR}" \
      ++model.repair_reliability_threshold=0.58 \
      ++model.repair_reliability_temperature=0.12 \
      ++model.repair_reliability_top_fraction=0.35 \
      ++model.repair_event_support_dilate_kernel=5 \
      ++model.repair_event_support_floor="${EVENT_SUPPORT_FLOOR}" \
      ++model.repair_residual_gain=1.6 \
      ++model.repair_output_abs_limit=0.06 \
      ++model.repair_pose_translation_scale="${POSE_TRANSLATION_SCALE}" \
      ++model.repair_pose_quaternion_scale="${POSE_QUATERNION_SCALE}" \
      ++model.repair_train_coarse_heads=false \
      ++model.repair_refiner_residual_scale=0.05 \
      ++model.repair_event_delta_highpass_kernel=0 \
      ++model.repair_event_delta_patch_zero_mean=false \
      ++model.repair_event_delta_abs_limit=0.05 \
      ++loss.stage2_residual_target_weight=2.0 \
      ++loss.stage2_residual_gradient_weight=3.0 \
      ++loss.stage2_target_reliability_floor=0.10 \
      ++loss.stage2_target_abs_limit=0.06 \
      ++loss.stage2_target_highpass_kernel=0 \
      ++loss.stage2_event_top_fraction=0.50 \
      ++loss.normal_weight="${NORMAL_WEIGHT}" \
      ++loss.repair_pose_weight="${POSE_WEIGHT}" \
      ++loss.repair_detail_gt_normal_weight="${DETAIL_NORMAL_WEIGHT}" \
      ++loss.repair_detail_gt_hf_weight="${DETAIL_HF_WEIGHT}" \
      ++loss.repair_detail_gt_grad_weight="${DETAIL_GRAD_WEIGHT}" \
      ++loss.stage2_flat_normal_weight="${FLAT_NORMAL_WEIGHT}" \
      ++loss.stage2_flat_residual_gradient_weight=0.50 \
      ++loss.stage2_no_event_residual_weight=0.20 \
      ++loss.stage2_contribution_depth_guard_weight="${CONTRIBUTION_DEPTH_GUARD_WEIGHT}" \
      ++loss.stage2_contribution_normal_guard_weight="${CONTRIBUTION_NORMAL_GUARD_WEIGHT}" \
      ++vis.save_every_steps=3000 \
      2>&1 | tee "${OUT_ROOT}/logs/stage2_${ablation}.log"
  fi

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] missing Stage2 checkpoint for ablation=${ablation}: ${checkpoint}" >&2
    exit 4
  fi
  if ! validate_stage2_binding "${checkpoint}" "${stage1_checkpoint}" "${stage2_dir}"; then
    echo "[error] Stage2 checkpoint is bound to another Stage1/output path." >&2
    echo "        Use a new OUT_ROOT or remove only this Stage2 ablation directory." >&2
    exit 6
  fi
  echo "[${ablation}] validated Stage2 -> Stage1 binding"

  echo
  echo "=============================="
  echo "[${ablation}] held-out causal eval"
  echo "=============================="
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m "${EVAL_MODULE}" \
    --checkpoint "${checkpoint}" \
    --reliability-checkpoint "${stage1_checkpoint}" \
    --output-dir "${stage2_dir}/heldout_eval" \
    --root "${DATA_ROOT}" \
    --initial-scene-idx 12 \
    --active-scene-count 4 \
    --test-frame-count 120 \
    --window-stride 4 \
    --num-views 4 \
    --event-resize-bins 10 \
    --amp bf16 \
    2>&1 | tee "${OUT_ROOT}/logs/eval_${ablation}.log"

  echo "${ablation},${target_mode},${factors},${reliability_dir},${stage2_dir},${checkpoint},${stage2_dir}/heldout_eval" >> "${summary_csv}"
  python - \
    "${ablation}" "${target_mode}" "${factors}" \
    "${reliability_dir}/history.json" \
    "${stage2_dir}/heldout_eval/summary.json" \
    "${metrics_csv}" <<'PY'
import csv, json, sys

ablation, target_mode, factors, history_path, summary_path, output_path = sys.argv[1:]
history = json.load(open(history_path, "r", encoding="utf-8"))
summary = json.load(open(summary_path, "r", encoding="utf-8"))
best_iou = max((float(row["val"]["iou"]) for row in history), default=float("nan"))
full = summary["conditions"]["full_event"]
gain = summary["comparisons"]["full_event_net_gain"]
causal = summary.get("diagnostics", {}).get("causal_prediction_differences", {})
row = [
    ablation,
    target_mode,
    factors,
    best_iou,
    full.get("abs_rel"),
    full.get("delta1"),
    full.get("rmse_log"),
    full.get("normal_mean_deg"),
    full.get("normal_11_25"),
    full.get("ate"),
    full.get("event_reliability_mean"),
    gain.get("normal_mean_deg_reduction"),
    gain.get("abs_rel_reduction"),
    causal.get("full_vs_zero"),
]
with open(output_path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(row)
PY
done

echo
echo "done. summary: ${summary_csv}"
echo "paper metrics: ${metrics_csv}"
