#!/usr/bin/env bash
# Direct-add EventVGGT under nested raw-event E_geo -> E_full injection.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
EXP_DIR="${EXP_DIR:-exp_f/direct_full_event_no_strategy_12train_3epoch}"
CHECKPOINT="${CHECKPOINT:-${EXP_DIR}/checkpoint-last.pth}"
OUTPUT="${OUTPUT:-${EXP_DIR}/test_geo_to_full_10level_scale1}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing direct-full checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m paired_token_reliability.evaluate_direct_full_geo_to_full_10level \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" \
  --root "${DATA_ROOT}" \
  --levels "${LEVELS:-10}" \
  --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --geo-depth-scale "${GEO_DEPTH_SCALE:-1.0}" \
  --full-depth-scale "${FULL_DEPTH_SCALE:-1.0}" \
  --num-views 4 \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS:-0}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/test.log"

echo "[done] ${OUTPUT}/all_levels_metrics.csv"
echo "[done] ${OUTPUT}/all_levels_summary.json"
echo "[done] ${OUTPUT}/plots"
echo "[done] ${OUTPUT}/level_*/visualizations"
