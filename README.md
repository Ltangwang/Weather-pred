# ProbWrapper — Plug-and-Play Probabilistic Spatiotemporal Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ProbWrapper** is a lightweight Gaussian head that wraps existing deterministic spatiotemporal backbones from [OpenSTL](https://github.com/chengtan990/OpenSTL) (SimVP, TAU, ConvLSTM, PredRNN) without modifying their architectures. It outputs per-pixel predictive mean and log-variance for short-range weather forecasting on [WeatherBench](https://github.com/pangeo-data/WeatherBench) 2 m temperature (`t2m`, 5.625°).

The recommended training recipe is a **two-stage CRPS → NLL** schedule: Stage 1 trains with Gaussian CRPS (stable cold start); Stage 2 fine-tunes the variance head with NLL while freezing the backbone.

## Highlights

- **Backbone-agnostic**: same `ProbWrapper` + training code for SimVP, TAU, ConvLSTM (and PredRNN via a thin inference shim).
- **Physical-unit metrics**: RMSE, MAE, CRPS, NLL in **Kelvin**; ECE with 10 equal-width bins.
- **Output shape** (enforced): `(B, 2, C, H, W)` — index `0` = mean, `1` = log_var.
- **Probabilistic baselines**: naive climatology/persistence, MC Dropout, Deep Ensemble (from existing checkpoints).
- **Streaming evaluation** on the full test split (no RAM blow-up on 17k+ samples).

## Repository layout

```
src/              Core library (dataset, ProbWrapper, losses, metrics, calibration)
scripts/          CLI entry points (train, fine-tune, baselines, figures, tables)
configs/          YAML defaults (optimizer, data paths template)
tests/            pytest unit tests
examples/         Short runnable examples (smoke + full SimVP pipeline)
docs/             End-to-end workflow (WORKFLOW.md)
paper/            Optional LaTeX intro/conclusion snippets
```

**Not included in this repo** (install separately):

- [OpenSTL](https://github.com/chengtan990/OpenSTL) — backbone implementations
- WeatherBench NetCDF data under `data/weather_5_625deg/2m_temperature/`

## Requirements

- Python 3.10+
- CUDA GPU recommended (tested on RTX 3060 12 GB)
- OpenSTL installed editable: `pip install -e /path/to/OpenSTL`

```bash
pip install -r requirements.txt
pip install -e /path/to/OpenSTL   # sibling clone, not vendored here
```

## Quick start

```bash
# 1) Unit tests (no data required)
pytest tests/ -q

# 2) Smoke probabilistic run (few batches — see examples/)
bash examples/smoke_probabilistic.sh

# 3) Full pipeline documentation
#    See docs/WORKFLOW.md
```

## Minimal API usage

```python
from openstl.models import SimVP_Model
from src.model import ProbWrapper
import torch

backbone = SimVP_Model(
    in_shape=(12, 1, 32, 64),
    hid_S=32, hid_T=256, N_S=2, N_T=8,
    model_type="gSTA", mlp_ratio=8.0, drop=0.0, drop_path=0.1,
    spatio_kernel_enc=3, spatio_kernel_dec=3,
)
model = ProbWrapper(backbone, out_channels=1, multi_frame=True)
x = torch.randn(2, 12, 1, 32, 64)   # (B, T_in, C, H, W)
out = model(x)                       # (B, 2, T_out, C, H, W)
mean, log_var = out[:, 0], out[:, 1]
```

## Main training commands (full dataset)

Set paths for your machine:

```bash
export DATA_ROOT=/path/to/OpenSTL/data
export OUT_DIR=/path/to/results
```

**Stage 1 — CRPS (select best by val CRPS):**

```bash
python scripts/run_probabilistic.py \
  --backbone SimVP --loss crps --multi_frame \
  --epochs 50 --lr 2e-4 --pct_start 0.2 \
  --batch_size 16 --val_batch_size 4 --num_workers 0 \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_DIR/simvp_prob_crps_mf_s42" --seed 42
```

**Stage 2 — NLL fine-tune (frozen backbone):**

```bash
python scripts/run_finetune_nll.py \
  --backbone SimVP --multi_frame \
  --init_from "$OUT_DIR/simvp_prob_crps_mf_s42/checkpoints/best.pth" \
  --epochs 10 --lr 1e-4 \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_DIR/simvp_prob_nll_ft_mf_s42" --seed 42
```

## Scripts overview

| Script | Role |
|--------|------|
| `run_probabilistic.py` | Stage 1 (CRPS or NLL from scratch) |
| `run_finetune_nll.py` | Stage 2 NLL fine-tune |
| `run_deterministic.py` | OpenSTL deterministic baselines (SimVP / ConvLSTM) |
| `eval_naive_baselines.py` | Climatology / persistence probabilistic baselines |
| `run_mc_dropout.py` | MC Dropout baseline (full ProbWrapper + latent dropout) |
| `run_deep_ensemble.py` | Deep ensemble from multiple `best.pth` checkpoints |
| `aggregate_seeds.py` | Mean ± std over seeds → LaTeX table |
| `build_paper_tables.py` | Consolidated paper tables |
| `plot_leadtime.py` / `plot_leadtime_compare.py` | Lead-time curves |
| `temperature_scaling.py` | Optional post-hoc calibration (Stage 3) |
| `run_backbone_ablation.sh` | TAU / PredRNN full runs |

Optional / paper helpers: `plot_reliability_compare.py`, `run_full_experiments.sh`.

## Data download

WeatherBench 1 `t2m` @ 5.625° (matches OpenSTL layout):

```bash
bash scripts/download_weatherbench_t2m_5625deg.sh /path/to/OpenSTL/data
```

Expected layout: `data/weather_5_625deg/2m_temperature/*.nc`

## Citation

If you use this code, please cite the ProbWrapper paper together with the OpenSTL and WeatherBench resources it builds on.

**ProbWrapper (this method):**

```bibtex
@article{hou2026probwrapper,
  title   = {ProbWrapper: Backbone-Agnostic Probabilistic Neural Spatiotemporal
             Weather Forecasting with Two-Stage CRPS--NLL Training},
  author  = {Hou, WeiDong and Kasmiran, Khairul Azhar},
  year    = {2026},
  note    = {Manuscript}
}
```

Please also cite the backbones/benchmark this work depends on:

- **OpenSTL** — Tan et al., *OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning*, NeurIPS Datasets and Benchmarks, 2023.
- **WeatherBench** — Rasp et al., *WeatherBench: A Benchmark Data Set for Data-Driven Weather Forecasting*, JAMES, 2020.

## License

MIT — see [LICENSE](LICENSE).
