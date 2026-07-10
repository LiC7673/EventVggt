#!/usr/bin/env bash
set -Eeuo pipefail

# v5 adds an event-conditioned pose residual and contribution-aware depth /
# normal guards. It uses the non-saturated schema-v2 labels introduced in v4.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUT_ROOT="${OUT_ROOT:-abl_event_exp/loss1_factor_ablation_v5_poseguard}"
export LABEL_DIR="${LABEL_DIR:-${OUT_ROOT}/labels_schema2_raw_quantile}"
export LABEL_SCHEMA_VERSION=2
export TOKEN_SOURCE=raw
export TOKEN_AGREEMENT_MODE=robust_quantile
export STAGE2_TAG="poseguard_v5"
export NORMAL_WEIGHT="${NORMAL_WEIGHT:-0.30}"
export FLAT_NORMAL_WEIGHT="${FLAT_NORMAL_WEIGHT:-0.75}"
export POSE_TRANSLATION_SCALE="${POSE_TRANSLATION_SCALE:-0.01}"
export POSE_QUATERNION_SCALE="${POSE_QUATERNION_SCALE:-0.01}"
export CONTRIBUTION_DEPTH_GUARD_WEIGHT="${CONTRIBUTION_DEPTH_GUARD_WEIGHT:-1.0}"
export CONTRIBUTION_NORMAL_GUARD_WEIGHT="${CONTRIBUTION_NORMAL_GUARD_WEIGHT:-1.0}"
export POSE_WEIGHT="${POSE_WEIGHT:-1.0}"
# Do not supervise final normals with the legacy Detail-GT objective in the
# residual stage. Geometry detail is already used in the residual target
# weights; normal quality is protected by global/flat/contribution guards.
export DETAIL_NORMAL_WEIGHT="${DETAIL_NORMAL_WEIGHT:-0.0}"
export DETAIL_HF_WEIGHT="${DETAIL_HF_WEIGHT:-0.0}"
export DETAIL_GRAD_WEIGHT="${DETAIL_GRAD_WEIGHT:-0.0}"
export GATE_FLOOR="${GATE_FLOOR:-0.0}"
export EVENT_SUPPORT_FLOOR="${EVENT_SUPPORT_FLOOR:-0.0}"

exec bash "${ROOT_DIR}/repair_reliability/run_loss1_component_ablation.sh" "$@"
