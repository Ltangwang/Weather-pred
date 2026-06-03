#!/usr/bin/env python3
"""Deep ensemble baseline: merge multiple ProbWrapper checkpoints at test time."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_probabilistic import (
    _build_backbone,
    _paper_eval_banner_probabilistic,
    _select_target_frame,
)
from src.calibration import pit_histogram, reliability_diagram
from src.dataset import WeatherBenchDataset
from src.metrics import MetricAccumulator
from src.model import ProbWrapper
from src.utils import get_logger, set_seed


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the deep-ensemble baseline."""

    p = argparse.ArgumentParser(
        description="Deep Ensemble of trained ProbWrapper checkpoints.",
    )
    p.add_argument("--members", nargs="+", required=True,
                   help="Paths to ProbWrapper best.pth checkpoints (>=2).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--data_split", type=str, default="5_625")
    p.add_argument("--variable", type=str, default="t2m")
    p.add_argument("--backbone", type=str, default="SimVP",
                   choices=["SimVP", "TAU", "PredRNN", "ConvLSTM"])
    p.add_argument("--in_len", type=int, default=12)
    p.add_argument("--out_len", type=int, default=12)
    p.add_argument("--multi_frame", action="store_true")
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--min_var", type=float, default=1e-6,
                   help="Numerical floor on predictive variance (K^2).")
    p.add_argument("--figure_max_scalars", type=int, default=200_000)
    p.add_argument("--limit_test_batches", type=int, default=None,
                   help="Cap test batches (None = full test split).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load_member(path: Path, backbone_name: str, in_len: int,
                 C: int, H: int, W: int, multi_frame: bool,
                 device: torch.device) -> ProbWrapper:
    """Build a ProbWrapper and load one member's weights."""

    backbone = _build_backbone(backbone_name, in_len, C, H, W)
    model = ProbWrapper(backbone, out_channels=C,
                        multi_frame=multi_frame).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def _ensemble_moments(models, x, mean: float, std: float, log_std2: float,
                      min_var: float):
    """Return mixture (mu_K, log_var_K) over ensemble members in Kelvin."""

    sum_mu = None
    sum_second = None  # sum_i (sigma_i^2 + mu_i^2)
    m = len(models)
    for model in models:
        out = model(x)
        mu_i = out[:, 0] * std + mean                     # Kelvin
        var_i = (out[:, 1] + log_std2).exp()              # Kelvin^2
        second_i = var_i + mu_i * mu_i
        sum_mu = mu_i if sum_mu is None else sum_mu + mu_i
        sum_second = second_i if sum_second is None else sum_second + second_i
    mu = sum_mu / m
    var = (sum_second / m) - mu * mu
    var = var.clamp(min=min_var)
    return mu.float().cpu(), var.log().float().cpu()


def main() -> None:
    """Run the deep-ensemble baseline end-to-end and write a paper banner."""

    args = _parse_args()
    set_seed(args.seed)
    if len(args.members) < 2:
        raise ValueError("Deep Ensemble needs >=2 member checkpoints.")

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "deep_ensemble.log"
    summary_path = output_dir / "paper_eval_summary.txt"
    norm_stats_path = output_dir / "norm_stats.json"
    logger = get_logger("deep_ens", str(log_path))
    logger.info("Args: %s", json.dumps(vars(args), indent=2))

    device = torch.device(args.device)

    logger.info("Building datasets ...")
    train_ds = WeatherBenchDataset(
        data_root=args.data_root, split="train",
        in_len=args.in_len, out_len=args.out_len,
        variable=args.variable, data_split=args.data_split,
        norm_stats_path=str(norm_stats_path),
    )
    norm_stats = (train_ds.mean, train_ds.std)
    test_ds = WeatherBenchDataset(
        data_root=args.data_root, split="test",
        in_len=args.in_len, out_len=args.out_len,
        variable=args.variable, data_split=args.data_split,
        norm_stats=norm_stats,
    )
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    logger.info("Test=%d  Norm: mean=%.6f std=%.6f",
                len(test_ds), train_ds.mean, train_ds.std)

    sample_x, _ = train_ds[0]
    _, C, H, W = sample_x.shape

    models = []
    for mp in args.members:
        path = Path(mp).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"member not found: {path}")
        models.append(_load_member(path, args.backbone, args.in_len,
                                   C, H, W, args.multi_frame, device))
    logger.info("Loaded %d ensemble members.", len(models))

    log_std2 = math.log(train_ds.std * train_ds.std)
    accum = MetricAccumulator()
    per_lt = {} if args.multi_frame else None

    # Figure subsample buffers.
    fig_mu_parts, fig_lv_parts, fig_tg_parts = [], [], []
    n_fig = 0

    for i, (x, y) in enumerate(test_loader):
        if args.limit_test_batches is not None and i >= args.limit_test_batches:
            break
        x = x.to(device, non_blocking=True)
        y_sup = _select_target_frame(y, multi_frame=args.multi_frame).to(
            device, non_blocking=True)
        mu, lv = _ensemble_moments(models, x, train_ds.mean, train_ds.std,
                                   log_std2, args.min_var)
        tg = (y_sup * train_ds.std + train_ds.mean).float().cpu()
        accum.update(mu, lv, tg)
        if per_lt is not None and mu.dim() == 5:
            for t in range(mu.shape[1]):
                per_lt.setdefault(t, MetricAccumulator()).update(
                    mu[:, t], lv[:, t], tg[:, t])
        # Subsample scalars for reliability/PIT.
        if n_fig < args.figure_max_scalars:
            mflat = mu.reshape(-1)
            remaining = args.figure_max_scalars - n_fig
            take = min(remaining, mflat.numel())
            fig_mu_parts.append(mflat[:take])
            fig_lv_parts.append(lv.reshape(-1)[:take])
            fig_tg_parts.append(tg.reshape(-1)[:take])
            n_fig += take
        if (i + 1) % 200 == 0:
            logger.info("  processed %d batches", i + 1)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    test = accum.finalize()
    logger.info(
        "Deep Ensemble (M=%d) TEST  rmse=%.4f mae=%.4f crps=%.4f nll=%.4f ece=%.4f",
        len(models), test["rmse"], test["mae"], test["crps"],
        test["nll"], test["ece"],
    )

    if per_lt is not None:
        lt_csv = output_dir / "per_leadtime.csv"
        with lt_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["leadtime", "rmse", "mae", "crps", "nll", "ece"])
            for t, acc in sorted(per_lt.items()):
                m = acc.finalize()
                w.writerow([t, f"{m['rmse']:.6f}", f"{m['mae']:.6f}",
                            f"{m['crps']:.6f}", f"{m['nll']:.6f}",
                            f"{m['ece']:.6f}"])
        logger.info("Per lead-time metrics written to %s", lt_csv)

    # Figures.
    shape = (n_fig, 1, 1, 1)
    fig_mu = torch.cat(fig_mu_parts).reshape(shape)
    fig_lv = torch.cat(fig_lv_parts).reshape(shape)
    fig_tg = torch.cat(fig_tg_parts).reshape(shape)
    reliability_diagram(fig_mu, fig_lv, fig_tg,
                        save_path=str(output_dir / "figures" / "reliability.png"))
    pit_histogram(fig_mu, fig_lv, fig_tg,
                  save_path=str(output_dir / "figures" / "pit_histogram.png"))

    dataname = f"weather_{args.variable}_{args.data_split}"
    banner = _paper_eval_banner_probabilistic(
        test, dataname=dataname, loss_name=f"Deep-Ensemble (M={len(models)})",
        backbone=args.backbone, epoch=-1,
    )
    print("\n" + banner)
    print(f"compact: rmse:{test['rmse']:.6f}, mae:{test['mae']:.6f}, "
          f"crps:{test['crps']:.6f}, nll:{test['nll']:.6f}, "
          f"ece:{test['ece']:.6f}")

    with summary_path.open("w") as f:
        f.write(banner + "\n\n")
        f.write("Ensemble members:\n")
        for mp in args.members:
            f.write(f"  {mp}\n")
        f.write(
            f"\ncompact: rmse:{test['rmse']}, mae:{test['mae']}, "
            f"crps:{test['crps']}, nll:{test['nll']}, ece:{test['ece']}\n"
        )
    logger.info("Wrote paper summary to %s", summary_path)


if __name__ == "__main__":
    main()
