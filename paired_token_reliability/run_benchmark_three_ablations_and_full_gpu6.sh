#!/usr/bin/env bash
# Hardware ablation: full method and three controlled variants, sequentially.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ABLATION_ROOT="${ABLATION_ROOT:-exp_f/latest_three_strategy_ablation_3epoch_v2_rgb_routes}"
FULL_CHECKPOINT="${FULL_CHECKPOINT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/hardware_ablation_full_and_three_variants}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
SCENE="${SCENE:-DH2_Socrates and Seneca_Car_Paint_Midnight}"
EXPOSURE="${EXPOSURE:-ev_2}"
WARMUP="${WARMUP:-10}"
REPEATS="${REPEATS:-50}"
AMP="${AMP:-none}"

mkdir -p "${OUTPUT_ROOT}/individual" "${OUTPUT_ROOT}/logs"

variants=(
  "full"
  "noisy_event_only"
  "multi_ldr_only"
  "without_refiner_normal"
)

for variant in "${variants[@]}"; do
  if [[ "${variant}" == "full" ]]; then
    checkpoint="${FULL_CHECKPOINT}"
  else
    checkpoint="${ABLATION_ROOT}/${variant}/checkpoint-adapter-last.pth"
  fi
  [[ -f "${checkpoint}" ]] || {
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
  }

  echo "[hardware ablation] variant=${variant} checkpoint=${checkpoint}"
  python -m paired_token_reliability.benchmark_cur_event_hardware \
    --variant "${variant}" \
    --checkpoint "${checkpoint}" \
    --output "${OUTPUT_ROOT}/individual/${variant}.json" \
    --root "${DATA_ROOT}" \
    --scene "${SCENE}" \
    --exposure "${EXPOSURE}" \
    --num-views 4 \
    --resolution 518 392 \
    --event-resize-bins 5 \
    --event-resize-method voxel_linear_time \
    --warmup "${WARMUP}" \
    --repeats "${REPEATS}" \
    --amp "${AMP}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${variant}.log"
done

python -m paired_token_reliability.collect_hardware_ablation_results \
  --input-dir "${OUTPUT_ROOT}/individual" \
  --output-prefix "${OUTPUT_ROOT}/hardware_ablation_summary"

echo "Hardware ablation complete: ${OUTPUT_ROOT}/hardware_ablation_summary.csv"
