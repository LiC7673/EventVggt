#!/usr/bin/env bash
set -Eeuo pipefail

# Loss1 v4: non-saturated pre-adapter token agreement plus explicit global
# and planar-region cosine-normal constraints.  It intentionally writes to a
# fresh root, so no v1/v2/v3 labels or checkpoints can be silently reused.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUT_ROOT="${OUT_ROOT:-abl_event_exp/loss1_factor_ablation_v4_tokencos}"
export LABEL_DIR="${LABEL_DIR:-${OUT_ROOT}/labels_schema2_raw_quantile}"
export LABEL_SCHEMA_VERSION=2
export TOKEN_SOURCE=raw
export TOKEN_AGREEMENT_MODE=robust_quantile
export TOKEN_QUANTILE_LOW="${TOKEN_QUANTILE_LOW:-0.05}"
export TOKEN_QUANTILE_HIGH="${TOKEN_QUANTILE_HIGH:-0.95}"
export STAGE2_TAG="flatguard_v4_tokencos"
export NORMAL_WEIGHT="${NORMAL_WEIGHT:-0.30}"
export FLAT_NORMAL_WEIGHT="${FLAT_NORMAL_WEIGHT:-0.75}"

exec bash "${ROOT_DIR}/repair_reliability/run_loss1_component_ablation.sh" "$@"
