#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
EXP_NAME="${EXP_NAME:-stage2_causal_event_reliability_train12_test4}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/abl_event_exp/${EXP_NAME}/checkpoint-last.pth}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/abl_event_exp/${EXP_NAME}/heldout_eval_scene12_15}"

GPU="${GPU}" \
EXP_NAME="${EXP_NAME}" \
CHECKPOINT="${CHECKPOINT}" \
OUT_DIR="${OUT_DIR}" \
bash real_reliability_stage/run_stage2_heldout_eval.sh "$@"

