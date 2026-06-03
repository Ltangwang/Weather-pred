#!/usr/bin/env bash
# Full-dataset ProbWrapper runs for backbone generalization (TAU / PredRNN).
# Usage: bash scripts/run_backbone_ablation.sh [TAU|PredRNN|both] [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA_ROOT:-/root/autodl-tmp/OpenSTL/data}"
OUT="${OUTPUT_ROOT:-/root/autodl-fs/Weather-pred-results}"
WHICH="${1:-both}"
SEED="${2:-42}"

_run_tau() {
  local S1="${OUT}/tau_prob_crps_mf_s${SEED}"
  local S2="${OUT}/tau_prob_nll_ft_mf_s${SEED}"
  echo "========== TAU seed=${SEED} Stage-1 CRPS (full val/test) =========="
  python scripts/run_probabilistic.py \
    --backbone TAU --loss crps \
    --lr 2e-4 --pct_start 0.2 --grad_clip 1.0 --patience 10 --min_epochs 15 \
    --epochs 50 --multi_frame \
    --num_workers 0 --batch_size 16 --val_batch_size 4 \
    --data_root "$DATA" --output_dir "$S1" --seed "$SEED"
  echo "========== TAU seed=${SEED} Stage-2 NLL-FT =========="
  python scripts/run_finetune_nll.py \
    --backbone TAU \
    --init_from "${S1}/checkpoints/best.pth" \
    --multi_frame \
    --num_workers 0 --batch_size 16 --val_batch_size 4 \
    --data_root "$DATA" --output_dir "$S2" \
    --epochs 10 --lr 1e-4 --seed "$SEED"
}

_run_predrnn() {
  local S1="${OUT}/predrnn_prob_crps_mf_s${SEED}"
  local S2="${OUT}/predrnn_prob_nll_ft_mf_s${SEED}"
  echo "========== PredRNN seed=${SEED} Stage-1 CRPS (full val/test) =========="
  python scripts/run_probabilistic.py \
    --backbone PredRNN --loss crps \
    --lr 2e-4 --pct_start 0.2 --grad_clip 1.0 --patience 10 --min_epochs 15 \
    --epochs 50 --multi_frame \
    --num_workers 0 --batch_size 8 --val_batch_size 4 \
    --data_root "$DATA" --output_dir "$S1" --seed "$SEED"
  echo "========== PredRNN seed=${SEED} Stage-2 NLL-FT =========="
  python scripts/run_finetune_nll.py \
    --backbone PredRNN \
    --init_from "${S1}/checkpoints/best.pth" \
    --multi_frame \
    --num_workers 0 --batch_size 8 --val_batch_size 4 \
    --data_root "$DATA" --output_dir "$S2" \
    --epochs 10 --lr 1e-4 --seed "$SEED"
}

case "$WHICH" in
  TAU)     _run_tau ;;
  PredRNN) _run_predrnn ;;
  both)    _run_tau; _run_predrnn ;;
  *)       echo "Usage: $0 [TAU|PredRNN|both] [SEED]"; exit 1 ;;
esac
