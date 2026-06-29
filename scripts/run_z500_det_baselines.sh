#!/usr/bin/env bash
# z500 deterministic MSE baselines (OpenSTL).
#
# Usage:
#   bash scripts/run_z500_det_baselines.sh simvp              # full (~2.5 d)
#   bash scripts/run_z500_det_baselines.sh simvp fast         # fast (~10–14 h)
#   bash scripts/run_z500_det_baselines.sh both fast          # SimVP + TAU fast
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${2:-full}"

_run() {
  local BB="$1"
  echo "========== z500 deterministic MSE (${BB}), mode=${MODE}, seed=42 =========="
  python scripts/run_deterministic.py \
    --config configs/openstl_baseline_z500.yaml \
    --backbone "$BB" \
    --mode "$MODE"
}

case "${1:-simvp}" in
  simvp) _run simvp ;;
  tau)   _run tau ;;
  both)  _run simvp; _run tau ;;
  *)
    echo "Usage: bash scripts/run_z500_det_baselines.sh simvp|tau|both [full|fast]"
    exit 1
    ;;
esac
