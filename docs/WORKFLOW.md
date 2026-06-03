# Project workflow

End-to-end guide for reproducing ProbWrapper experiments on WeatherBench `t2m` (5.625°, 12→12 frames, 6 h per step).

## 1. Environment

```bash
git clone <this-repo> Weather-pred && cd Weather-pred
git clone https://github.com/chengtan990/OpenSTL.git ../OpenSTL
cd ../OpenSTL && pip install -e .
cd ../Weather-pred
pip install -r requirements.txt
pytest tests/ -q
```

Set paths (adjust to your machine):

```bash
export REPO_ROOT="$(pwd)"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/../OpenSTL/data}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/results}"
```

## 2. Data

Download `t2m` NetCDF files into OpenSTL’s expected tree:

```bash
bash scripts/download_weatherbench_t2m_5625deg.sh "$DATA_ROOT"
# → $DATA_ROOT/weather_5_625deg/2m_temperature/*.nc
```

Splits and normalization stats are handled by `src/dataset.py` (train mean/std cached in each run’s `norm_stats.json`).

## 3. Code flow (library)

```
WeatherBenchDataset  →  (x, y) batches in z-score space
        ↓
ProbWrapper(backbone)  →  (B, 2, [T,] C, H, W)  mean + log_var
        ↓
gaussian_crps / gaussian_nll  (training)
        ↓
MetricAccumulator  →  RMSE, MAE, CRPS, NLL, ECE in Kelvin (eval)
```

- **Training** lives in `scripts/run_probabilistic.py` and `scripts/run_finetune_nll.py`.
- **Inference**: `var = exp(log_var).clamp(min=1e-6)` in physical units after denormalization.

## 4. Recommended experiment pipeline

### Step A — Smoke test (minutes)

```bash
bash examples/smoke_probabilistic.sh
```

Uses `limit_val_batches` / `limit_test_batches` to verify GPU, data paths, and shapes.

### Step B — Main method (SimVP, ~1–2 days on RTX 3060)

```bash
SEED=42
S1="$OUT_DIR/simvp_prob_crps_mf_s${SEED}"
S2="$OUT_DIR/simvp_prob_nll_ft_mf_s${SEED}"

# Stage 1: CRPS, full val/test
python scripts/run_probabilistic.py \
  --backbone SimVP --loss crps --multi_frame \
  --lr 2e-4 --pct_start 0.2 --grad_clip 1.0 \
  --patience 10 --min_epochs 15 --epochs 50 \
  --num_workers 0 --batch_size 16 --val_batch_size 4 \
  --data_root "$DATA_ROOT" --output_dir "$S1" --seed "$SEED"

# Stage 2: NLL fine-tune (~2.5 h)
python scripts/run_finetune_nll.py \
  --backbone SimVP --multi_frame \
  --init_from "$S1/checkpoints/best.pth" \
  --epochs 10 --lr 1e-4 \
  --num_workers 0 --batch_size 16 --val_batch_size 4 \
  --data_root "$DATA_ROOT" --output_dir "$S2" --seed "$SEED"
```

Artifacts per run:

- `checkpoints/best.pth`
- `paper_eval_summary.txt` — headline test metrics (Kelvin)
- `per_leadtime.csv` — metrics per forecast step
- `figures/reliability.png`, `figures/pit_histogram.png`

### Step C — Multi-seed aggregation (optional, paper)

Repeat Step B with `SEED=43,44`, then:

```bash
python scripts/aggregate_seeds.py \
  --inputs "$OUT_DIR/simvp_prob_nll_ft_mf_s42" \
           "$OUT_DIR/simvp_prob_nll_ft_mf_s43" \
           "$OUT_DIR/simvp_prob_nll_ft_mf_s44" \
  --output_dir "$OUT_DIR/_aggregated_3seed" \
  --label "Ours (SimVP, Stage-2)"
```

### Step D — Probabilistic baselines (no extra training for Deep Ensemble)

```bash
# Naive baselines (full test)
python scripts/eval_naive_baselines.py \
  --data_root "$DATA_ROOT" --multi_frame \
  --output_dir "$OUT_DIR/naive_baselines"

# MC Dropout (loads Stage-2 or Stage-1 checkpoint)
python scripts/run_mc_dropout.py \
  --backbone SimVP --multi_frame \
  --init_from "$S2/checkpoints/best.pth" \
  --mc_samples 10 --mc_dropout_p 0.1 \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_DIR/baseline_mc_dropout" --seed 42

# Deep Ensemble (combine 3 Stage-2 checkpoints)
python scripts/run_deep_ensemble.py \
  --backbone SimVP --multi_frame \
  --members \
    "$OUT_DIR/simvp_prob_nll_ft_mf_s42/checkpoints/best.pth" \
    "$OUT_DIR/simvp_prob_nll_ft_mf_s43/checkpoints/best.pth" \
    "$OUT_DIR/simvp_prob_nll_ft_mf_s44/checkpoints/best.pth" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_DIR/baseline_deep_ensemble"
```

### Step E — Backbone generalization (single seed 42)

Same two-stage protocol; swap `--backbone`:

| Backbone | Notes | Rough Stage-1 time (3060) |
|----------|--------|---------------------------|
| `TAU` | Drop-in SimVP family (`model_type=tau`) | ~24 h / 50 epochs |
| `ConvLSTM` | Lightweight H=32, L=2 | ~28 h / 50 epochs |
| `PredRNN` | Very slow (autoregressive); optional | days — often omitted |

```bash
bash scripts/run_backbone_ablation.sh TAU 42
# ConvLSTM Stage-2 only (if Stage-1 already done):
python scripts/run_finetune_nll.py --backbone ConvLSTM --multi_frame \
  --init_from "$OUT_DIR/convlstm_h32l2_s42/checkpoints/best.pth" \
  ...
```

### Step F — Figures and LaTeX tables

```bash
python scripts/build_paper_tables.py \
  --results_root "$OUT_DIR" --output_dir "$OUT_DIR/_paper"

python scripts/plot_leadtime_compare.py \
  --results_root "$OUT_DIR" \
  --output "$OUT_DIR/_paper/leadtime_compare.png" \
  --series "Ours,$OUT_DIR/simvp_prob_nll_ft_mf_s42/per_leadtime.csv" \
  --series "Deep Ens.,$OUT_DIR/baseline_deep_ensemble/per_leadtime.csv"
```

### Step G — Ablations

| Ablation | Command idea |
|----------|----------------|
| NLL from scratch | `run_probabilistic.py --loss nll` (no Stage 2) |
| Stage 1 only | stop after Step B Stage 1 |
| Temperature scaling | `scripts/temperature_scaling.py` (optional Stage 3) |

## 5. Deterministic baselines (OpenSTL native)

For comparison with MSE-trained SimVP / ConvLSTM:

```bash
python scripts/run_deterministic.py --backbone simvp --mode full
```

See `configs/openstl_baseline.yaml` for batch limits and hardware.

## 6. What we do **not** ship in git

- Trained checkpoints (`.pth`), logs, zip packs of results
- WeatherBench NetCDF files
- OpenSTL source (clone separately)

Keep large outputs under `OUT_DIR` locally or on cloud storage.

## 7. Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on full test eval | Use default streaming metrics; avoid storing full `_mu` tensors |
| `import openstl` fails | `pip install -e ../OpenSTL` |
| MC Dropout nonsense RMSE | Must use full `ProbWrapper` + latent dropout (see `run_mc_dropout.py`) |
| PredRNN too slow | Skip; 3 backbones (SimVP, TAU, ConvLSTM) suffice for generality |
