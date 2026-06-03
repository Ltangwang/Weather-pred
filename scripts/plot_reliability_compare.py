#!/usr/bin/env python3
"""Compare reliability diagrams with and without temperature scaling."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.temperature_scaling import _collect
from src.calibration import compute_ece, reliability_diagram
from src.dataset import WeatherBenchDataset
from src.model import ProbWrapper
from src.utils import set_seed
from scripts.run_probabilistic import _build_backbone


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay reliability: before vs after temperature scaling.",
    )
    p.add_argument("--init_from", type=str, required=True,
                   help="Stage-2 NLL-FT best.pth.")
    p.add_argument("--temperature_json", type=str, required=True,
                   help="Stage-3 temperature.json containing scalar T.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--backbone", type=str, default="SimVP",
                   choices=["SimVP", "ConvLSTM"])
    p.add_argument("--multi_frame", action="store_true")
    p.add_argument("--in_len", type=int, default=12)
    p.add_argument("--out_len", type=int, default=12)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--limit_test_batches", type=int, default=None,
                   help="Optional cap for smoke tests only; default full test.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _coverage_curve(mean: torch.Tensor, log_var: torch.Tensor,
                    target: torch.Tensor) -> tuple[list[float], list[float]]:
    """Return (nominal confidences, empirical coverages) for 10%,…,90%."""
    sigma = (0.5 * log_var).exp()
    z = ((target - mean) / sigma).abs()
    confidences = np.linspace(0.1, 0.9, 9)
    coverages = []
    for alpha in confidences:
        half_width = math.sqrt(2.0) * float(
            torch.erfinv(torch.tensor(alpha)).item())
        coverages.append((z <= half_width).float().mean().item())
    return confidences.tolist(), coverages


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.temperature_json).open() as f:
        T = float(json.load(f)["T"])

    norm_stats_path = Path(args.init_from).resolve().parents[1] / "norm_stats.json"
    train_ds = WeatherBenchDataset(
        args.data_root, split="train",
        in_len=args.in_len, out_len=args.out_len,
        norm_stats_path=str(norm_stats_path),
    )
    test_ds = WeatherBenchDataset(
        args.data_root, split="test",
        in_len=args.in_len, out_len=args.out_len,
        norm_stats=(train_ds.mean, train_ds.std),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    C = 1
    backbone = _build_backbone(
        args.backbone, in_len=args.in_len, channels=C,
        height=32, width=64,
    )
    model = ProbWrapper(backbone, out_channels=C, multi_frame=args.multi_frame)
    ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    mu, lv, tg = _collect(
        model, test_loader, device,
        train_ds.mean, train_ds.std,
        multi_frame=args.multi_frame,
        limit_batches=args.limit_test_batches,
    )
    if args.multi_frame and mu.dim() == 5:
        n_flat, n_lt, c_f, h_f, w_f = mu.shape
        mu = mu.reshape(n_flat * n_lt, c_f, h_f, w_f)
        lv = lv.reshape(n_flat * n_lt, c_f, h_f, w_f)
        tg = tg.reshape(n_flat * n_lt, c_f, h_f, w_f)

    lv_cal = lv + 2.0 * math.log(T)
    ece_pre = compute_ece(mu, lv, tg)
    ece_post = compute_ece(mu, lv_cal, tg)
    conf_pre, cov_pre = _coverage_curve(mu, lv, tg)
    conf_post, cov_post = _coverage_curve(mu, lv_cal, tg)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax.plot(conf_pre, cov_pre, "o-", color="C1", linewidth=2,
            label=f"Stage-2 (NLL-FT, ECE={ece_pre:.3f})")
    ax.plot(conf_post, cov_post, "s-", color="C0", linewidth=2,
            label=f"Stage-3 (T={T:.2f}, ECE={ece_post:.3f})")
    ax.set_xlabel("Nominal confidence", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Reliability diagram (test set)", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved {out_path}")
    print(f"Stage-2 ECE={ece_pre:.4f}  Stage-3 ECE={ece_post:.4f}  T={T:.4f}")


if __name__ == "__main__":
    main()
