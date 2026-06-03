#!/usr/bin/env bash
# Re-run full test evaluation from an existing checkpoint (no training).
# Set INIT_CKPT to your best.pth path.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export DATA_ROOT="${DATA_ROOT:-$REPO/../OpenSTL/data}"
# Directory that already contains checkpoints/best.pth from a prior train run.
export RUN_DIR="${RUN_DIR:?Set RUN_DIR=/path/to/run_dir_with_checkpoints}"

python scripts/run_probabilistic.py \
  --backbone SimVP --loss crps --multi_frame \
  --eval_only \
  --num_workers 0 --val_batch_size 4 \
  --data_root "$DATA_ROOT" \
  --output_dir "$RUN_DIR" \
  --seed 42

echo "Eval done: $RUN_DIR/paper_eval_summary.txt"
