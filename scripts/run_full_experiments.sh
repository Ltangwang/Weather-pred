#!/usr/bin/env bash
# Full WeatherBench eval pipeline — NO batch limits on val/test.
#
# Usage:
#   bash scripts/run_full_experiments.sh simvp 42          # SimVP 3-stage, seed 42
#   bash scripts/run_full_experiments.sh simvp 43
#   bash scripts/run_full_experiments.sh simvp 44
#   bash scripts/run_full_experiments.sh convlstm 42       # ConvLSTM Stage-1 only
#   bash scripts/run_full_experiments.sh post              # naive + aggregate + fig
#
# ASSUMPTION: data at /root/autodl-tmp/OpenSTL/data ; results under autodl-fs.

set -euo pipefail

ROOT="/root/autodl-tmp/Weather-pred"
DATA="/root/autodl-tmp/OpenSTL/data"
OUT="/root/autodl-fs/Weather-pred-results"

cd "$ROOT"

_run_simvp_seed() {
  local SEED="$1"
  local S1="${OUT}/simvp_prob_crps_mf_s${SEED}"
  local S2="${OUT}/simvp_prob_nll_ft_mf_s${SEED}"
  local S3="${OUT}/simvp_prob_nll_ft_mf_sweep_s${SEED}"

  echo "========== SimVP seed=${SEED} Stage-1 CRPS (full val/test) =========="
  python scripts/run_probabilistic.py \
    --backbone SimVP --loss crps \
    --lr 2e-4 --pct_start 0.2 \
    --grad_clip 1.0 --patience 10 --min_epochs 15 \
    --epochs 50 --multi_frame \
    --num_workers 0 --batch_size 16 --val_batch_size 4 \
    --data_root "$DATA" \
    --output_dir "$S1" \
    --seed "$SEED"

  echo "========== SimVP seed=${SEED} Stage-2 NLL-FT =========="
  python scripts/run_finetune_nll.py \
    --backbone SimVP \
    --init_from "${S1}/checkpoints/best.pth" \
    --multi_frame \
    --num_workers 0 --batch_size 16 --val_batch_size 4 \
    --data_root "$DATA" \
    --output_dir "$S2" \
    --epochs 10 --lr 1e-4 --seed "$SEED"

  echo "========== SimVP seed=${SEED} Stage-3 Temperature scaling =========="
  python scripts/temperature_scaling.py \
    --backbone SimVP \
    --init_from "${S2}/checkpoints/best.pth" \
    --multi_frame \
    --num_workers 4 --val_batch_size 4 \
    --data_root "$DATA" \
    --output_dir "$S3" \
    --sweep --sweep_min 0.5 --sweep_max 3.0 --sweep_n 51 \
    --seed "$SEED"
}

_run_convlstm() {
  local SEED="${1:-42}"
  echo "========== ConvLSTM seed=${SEED} Stage-1 CRPS (full val/test) =========="
  python scripts/run_probabilistic.py \
    --backbone ConvLSTM --loss crps \
    --lr 5e-4 --pct_start 0.2 \
    --grad_clip 1.0 --patience 10 --min_epochs 15 \
    --epochs 50 --multi_frame \
    --num_workers 0 --batch_size 16 --val_batch_size 4 \
    --data_root "$DATA" \
    --output_dir "${OUT}/convlstm_h32l2_s${SEED}" \
    --seed "$SEED"
}

_run_post() {
  echo "========== Naive baselines (full test) =========="
  python scripts/eval_naive_baselines.py \
    --data_root "$DATA" \
    --norm_stats "${OUT}/simvp_prob_crps_mf_s42/norm_stats.json" \
    --multi_frame \
    --output_dir "${OUT}/naive_baselines"

  echo "========== 3-seed aggregate (Stage-3) =========="
  python scripts/aggregate_seeds.py \
    --inputs \
      "${OUT}/simvp_prob_nll_ft_mf_sweep_s42" \
      "${OUT}/simvp_prob_nll_ft_mf_sweep_s43" \
      "${OUT}/simvp_prob_nll_ft_mf_sweep_s44" \
    --output_dir "${OUT}/_aggregated_3seed" \
    --label "Ours (SimVP, Stage-3, full test)"

  echo "========== Reliability diagram (seed 42 example) =========="
  mkdir -p "${OUT}/figures"
  python scripts/plot_reliability_compare.py \
    --init_from "${OUT}/simvp_prob_nll_ft_mf_s42/checkpoints/best.pth" \
    --temperature_json "${OUT}/simvp_prob_nll_ft_mf_sweep_s42/temperature.json" \
    --data_root "$DATA" \
    --multi_frame \
    --output "${OUT}/figures/reliability_stage2_vs_stage3_s42.png"

  echo "========== Lead-time plot (seed 42 Stage-1) =========="
  python scripts/plot_leadtime.py \
    --csv "${OUT}/simvp_prob_crps_mf_s42/per_leadtime.csv" \
    --hours_per_step 6 \
    --label "SimVP + CRPS (full test)" \
    --output_dir "${OUT}/simvp_prob_crps_mf_s42/figures"
}

case "${1:-}" in
  simvp)   _run_simvp_seed "${2:?usage: run_full_experiments.sh simvp <42|43|44>}" ;;
  convlstm) _run_convlstm "${2:-42}" ;;
  post)    _run_post ;;
  *)
    echo "Usage:"
    echo "  bash scripts/run_full_experiments.sh simvp 42|43|44"
    echo "  bash scripts/run_full_experiments.sh convlstm 42"
    echo "  bash scripts/run_full_experiments.sh post"
    exit 1
    ;;
esac
