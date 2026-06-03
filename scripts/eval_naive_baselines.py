#!/usr/bin/env python3
"""Climatology and persistence probabilistic baselines (test split)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset import WeatherBenchDataset, load_norm_stats
from src.metrics import MetricAccumulator
from src.utils import set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Naive probabilistic baselines.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--norm_stats", type=str, required=True,
                   help="JSON with train mean/std.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--in_len", type=int, default=12)
    p.add_argument("--out_len", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--multi_frame", action="store_true",
                   help="Supervise all output frames (matches main experiments).")
    p.add_argument("--limit_test_batches", type=int, default=None,
                   help="Cap test batches (None = full test split).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _denorm(t: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return t * std + mean


@torch.no_grad()
def _eval_baseline(
    loader: DataLoader,
    mean_k: float,
    std_k: float,
    mode: str,
    multi_frame: bool,
    limit_batches: int | None,
) -> dict[str, float]:
    """Run one naive baseline over the test loader (streaming metrics)."""
    log_var_k = math.log(std_k * std_k)
    accum = MetricAccumulator()

    for bi, (x, y) in enumerate(loader):
        if limit_batches is not None and bi >= limit_batches:
            break
        y_sup = y if multi_frame else y[:, -1:]
        tg = _denorm(y_sup, mean_k, std_k).float()

        if mode == "climatology":
            mu = torch.full_like(tg, mean_k)
        elif mode == "persistence":
            if multi_frame:
                last = _denorm(x[:, -1:], mean_k, std_k).float()
                mu = last.expand_as(tg)
            else:
                mu = _denorm(x[:, -1:], mean_k, std_k).float()
        else:
            raise ValueError(mode)

        lv = torch.full_like(mu, log_var_k)
        accum.update(mu.cpu(), lv.cpu(), tg.cpu())

    return accum.finalize()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_k, std_k = load_norm_stats(args.norm_stats)
    train_ds = WeatherBenchDataset(
        args.data_root, split="train",
        in_len=args.in_len, out_len=args.out_len,
        norm_stats=(mean_k, std_k),
    )
    test_ds = WeatherBenchDataset(
        args.data_root, split="test",
        in_len=args.in_len, out_len=args.out_len,
        norm_stats=(train_ds.mean, train_ds.std),
    )
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    results = {}
    for mode in ("climatology", "persistence"):
        results[mode] = _eval_baseline(
            loader, train_ds.mean, train_ds.std, mode,
            args.multi_frame, args.limit_test_batches,
        )

    lines = [
        "NAIVE PROBABILISTIC BASELINES (denormalized Kelvin)",
        f"  mean={train_ds.mean:.4f} K  std={train_ds.std:.4f} K",
        f"  multi_frame={args.multi_frame}",
        "-" * 60,
    ]
    for mode, m in results.items():
        lines.append(
            f"  {mode:16s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
            f"CRPS={m['crps']:.4f}  NLL={m['nll']:.4f}  ECE={m['ece']:.4f}"
        )
        lines.append(
            f"  compact ({mode}): rmse:{m['rmse']}, mae:{m['mae']}, "
            f"crps:{m['crps']}, nll:{m['nll']}, ece:{m['ece']}"
        )

    text = "\n".join(lines)
    print(text)
    (out_dir / "naive_baselines.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "naive_baselines.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    print(f"\nWrote {out_dir}/naive_baselines.txt")


if __name__ == "__main__":
    main()
