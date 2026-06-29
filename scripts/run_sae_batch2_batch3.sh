#!/usr/bin/env bash
# SAE revision experiments — Batch 2 (β-NLL ablation) and Batch 3 (z500 generalization).
#
# Usage:
#   bash scripts/run_sae_batch2_batch3.sh batch2 [SEED]          # β-NLL from scratch (t2m)
#   bash scripts/run_sae_batch2_batch3.sh batch2_seeds             # β-NLL, seeds 42/43/44
#   bash scripts/run_sae_batch2_batch3.sh batch3 [SEED]            # z500 SimVP two-stage
#   bash scripts/run_sae_batch2_batch3.sh batch3_tau [SEED]        # z500 TAU two-stage
#   bash scripts/run_sae_batch2_batch3.sh batch3_all [SEED]        # z500 SimVP + TAU
#   bash scripts/run_sae_batch2_batch3.sh tables                   # rebuild LaTeX tables
#
# Prerequisites:
#   - t2m data under DATA_ROOT (see download_weatherbench_t2m_5625deg.sh)
#   - z500 data for batch3 (see download_weatherbench_z500_5625deg.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA_ROOT:-/root/autodl-tmp/OpenSTL/data}"
OUT="${OUTPUT_ROOT:-/root/autodl-fs/Weather-pred-results}"

cd "$ROOT"

COMMON_TRAIN=(
  --backbone SimVP
  --lr 2e-4 --pct_start 0.2
  --grad_clip 1.0 --patience 10 --min_epochs 15
  --epochs 50 --multi_frame
  --num_workers 0 --batch_size 16 --val_batch_size 4
  --data_root "$DATA"
  --variable t2m
)

_run_beta_nll_seed() {
  local SEED="$1"
  local DIR="${OUT}/simvp_prob_beta_nll_scratch_mf_s${SEED}"
  echo "========== Batch 2: β-NLL from scratch (t2m), seed=${SEED} =========="
  python scripts/run_probabilistic.py \
    "${COMMON_TRAIN[@]}" \
    --loss beta_nll --beta 0.5 \
    --output_dir "$DIR" \
    --seed "$SEED"
}

_run_z500_two_stage() {
  local BACKBONE="$1"
  local SEED="$2"
  local S1="${OUT}/${BACKBONE,,}_z500_prob_crps_mf_s${SEED}"
  local S2="${OUT}/${BACKBONE,,}_z500_prob_nll_ft_mf_s${SEED}"

  echo "========== Batch 3: ${BACKBONE} z500 Stage-1 CRPS, seed=${SEED} =========="
  python scripts/run_probabilistic.py \
    --backbone "$BACKBONE" --loss crps \
    --lr 2e-4 --pct_start 0.2 \
    --grad_clip 1.0 --patience 10 --min_epochs 15 \
    --epochs 50 --multi_frame \
    --num_workers 0 --batch_size 16 --val_batch_size 4 \
    --data_root "$DATA" \
    --variable z500 \
    --output_dir "$S1" \
    --seed "$SEED"

  echo "========== Batch 3: ${BACKBONE} z500 Stage-2 NLL-FT, seed=${SEED} =========="
  python scripts/run_finetune_nll.py \
    --backbone "$BACKBONE" \
    --init_from "${S1}/checkpoints/best.pth" \
    --multi_frame \
    --num_workers 0 --batch_size 16 --val_batch_size 4 \
    --data_root "$DATA" \
    --variable z500 \
    --output_dir "$S2" \
    --epochs 10 --lr 1e-4 --seed "$SEED"
}

_rebuild_tables() {
  python scripts/build_paper_tables.py \
    --results_root "$OUT" \
    --output_dir "${OUT}/paper_tables"
}

case "${1:-}" in
  batch2)
    _run_beta_nll_seed "${2:-42}"
    ;;
  batch2_seeds)
    for s in 42 43 44; do _run_beta_nll_seed "$s"; done
    ;;
  batch3)
    _run_z500_two_stage SimVP "${2:-42}"
    ;;
  batch3_tau)
    _run_z500_two_stage TAU "${2:-42}"
    ;;
  batch3_all)
    SEED="${2:-42}"
    _run_z500_two_stage SimVP "$SEED"
    _run_z500_two_stage TAU "$SEED"
    ;;
  tables)
    _rebuild_tables
    ;;
  *)
    echo "Usage:"
    echo "  bash scripts/run_sae_batch2_batch3.sh batch2 [SEED]"
    echo "  bash scripts/run_sae_batch2_batch3.sh batch2_seeds"
    echo "  bash scripts/run_sae_batch2_batch3.sh batch3 [SEED]"
    echo "  bash scripts/run_sae_batch2_batch3.sh batch3_tau [SEED]"
    echo "  bash scripts/run_sae_batch2_batch3.sh batch3_all [SEED]"
    echo "  bash scripts/run_sae_batch2_batch3.sh tables"
    exit 1
    ;;
esac
