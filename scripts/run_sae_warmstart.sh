#!/usr/bin/env bash
# Warm-start ablation — deterministic SimVP MSE checkpoint -> ProbWrapper.
#
# Variant (a): warm-start backbone + end-to-end CRPS (backbone trainable)
# Variant (b): freeze backbone, CRPS on prob_head only (retrofit, skip Stage 1)
#
# Usage:
#   export DET_CKPT=/path/to/simvp_det_best.ckpt
#   bash scripts/run_sae_warmstart.sh a 42
#   bash scripts/run_sae_warmstart.sh b 42
#   bash scripts/run_sae_warmstart.sh both 42
#
# DET_CKPT: OpenSTL Lightning checkpoint (``state_dict``) or ProbWrapper ``best.pth``.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA_ROOT:-/root/autodl-tmp/OpenSTL/data}"
OUT="${OUTPUT_ROOT:-/root/autodl-fs/Weather-pred-results}"
DET_CKPT="${DET_CKPT:?Set DET_CKPT to your SimVP deterministic MSE checkpoint}"

VARIANT="${1:?usage: run_sae_warmstart.sh a|b|both [SEED]}"
SEED="${2:-42}"

cd "$ROOT"

COMMON=(
  --backbone SimVP --loss crps --multi_frame
  --lr 2e-4 --pct_start 0.2
  --grad_clip 1.0 --patience 10 --min_epochs 15
  --epochs 50
  --num_workers 0 --batch_size 16 --val_batch_size 4
  --data_root "$DATA" --variable t2m
  --init_from "$DET_CKPT"
  --seed "$SEED"
)

_run_a() {
  echo "========== Warm-start (a): det init + end-to-end CRPS, seed=${SEED} =========="
  python scripts/run_probabilistic.py \
    "${COMMON[@]}" \
    --output_dir "${OUT}/simvp_prob_crps_warmstart_e2e_mf_s${SEED}"
}

_run_b() {
  echo "========== Warm-start (b): det init + frozen backbone CRPS retrofit, seed=${SEED} =========="
  python scripts/run_probabilistic.py \
    "${COMMON[@]}" \
    --freeze_backbone \
    --output_dir "${OUT}/simvp_prob_crps_warmstart_frozen_mf_s${SEED}"
}

case "$VARIANT" in
  a)    _run_a ;;
  b)    _run_b ;;
  both) _run_a; _run_b ;;
  *)
    echo "Usage: DET_CKPT=/path/to.ckpt bash scripts/run_sae_warmstart.sh a|b|both [SEED]"
    exit 1
    ;;
esac
