#!/usr/bin/env bash
# Minimal ProbWrapper smoke test: 2 epochs, capped val/test batches.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export DATA_ROOT="${DATA_ROOT:-$REPO/../OpenSTL/data}"
export OUT_DIR="${OUT_DIR:-$REPO/results/examples_smoke}"
SEED="${SEED:-42}"

echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_DIR=$OUT_DIR"

pytest tests/ -q

python scripts/run_probabilistic.py \
  --backbone SimVP --loss crps --multi_frame \
  --epochs 2 --lr 2e-4 --pct_start 0.2 \
  --batch_size 4 --val_batch_size 4 --num_workers 0 \
  --limit_val_batches 20 --limit_test_batches 20 \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_DIR/simvp_smoke_s${SEED}" \
  --seed "$SEED"

echo "Smoke done. See: $OUT_DIR/simvp_smoke_s${SEED}/paper_eval_summary.txt"
