#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/paired_token_reliability/run_full_event_cross_view_dn_three_ablation_gpu4.sh" "$@"
