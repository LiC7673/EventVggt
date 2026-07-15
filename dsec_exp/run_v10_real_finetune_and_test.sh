#!/usr/bin/env bash
# Adapt a synthetic/LDR-trained V10 to real single-stream DSEC, then test.
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"

DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
GPU="${GPU:-4}"
V10_CHECKPOINT="${V10_CHECKPOINT:-exp/linear_voxel_dual_alignment_hdr_event_conditioned_adapter_v10_gpu4/checkpoint-adapter-best.pth}"
OUT="${OUT:-dsec_exp/results/v10_real_single_event_gpu${GPU}}"
EPOCHS="${EPOCHS:-10}"

[[ -d "${DSEC_ROOT}/train" ]] || { echo "missing ${DSEC_ROOT}/train" >&2; exit 2; }
[[ -d "${DSEC_ROOT}/test" ]] || { echo "missing ${DSEC_ROOT}/test" >&2; exit 2; }
[[ -f "${V10_CHECKPOINT}" ]] || { echo "missing V10 checkpoint: ${V10_CHECKPOINT}" >&2; exit 2; }
mkdir -p "${OUT}/logs"

CUDA_VISIBLE_DEVICES="${GPU}" python -m dsec_exp.finetune_v10_real \
  --checkpoint "${V10_CHECKPOINT}" --root "${DSEC_ROOT}" \
  --output "${OUT}" --epochs "${EPOCHS}" --num-views "${NUM_VIEWS:-4}" \
  --num-workers "${NUM_WORKERS:-4}" --batch-size "${BATCH_SIZE:-1}" \
  --train-stride "${TRAIN_STRIDE:-4}" --test-stride "${TEST_STRIDE:-4}" \
  --lr "${LR:-1e-5}" 2>&1 | tee "${OUT}/logs/train.log"

CUDA_VISIBLE_DEVICES="${GPU}" python -m dsec_exp.evaluate_v10_real \
  --checkpoint "${OUT}/checkpoint-best.pth" --root "${DSEC_ROOT}" \
  --output-dir "${OUT}/heldout_test" --num-views "${NUM_VIEWS:-4}" \
  --clip-stride "${TEST_STRIDE:-4}" --num-workers "${NUM_WORKERS:-4}" \
  2>&1 | tee "${OUT}/logs/test.log"

echo "DSEC V10 real adaptation complete: ${OUT}"
