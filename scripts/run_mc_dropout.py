#!/usr/bin/env python3
"""MC Dropout baseline: latent dropout on a trained ProbWrapper checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    p = argparse.ArgumentParser(description="MC Dropout probabilistic baseline.")
    p.add_argument("--init_from", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--data_split", type=str, default="5_625")
    p.add_argument("--variable", type=str, default="t2m")
    p.add_argument("--backbone", type=str, default="SimVP",
                   choices=["SimVP", "TAU", "PredRNN", "ConvLSTM"])
    p.add_argument("--in_len", type=int, default=12)
    p.add_argument("--out_len", type=int, default=12)
    p.add_argument("--multi_frame", action="store_true")
    p.add_argument("--mc_samples", type=int, default=20)
    p.add_argument("--mc_dropout_p", type=float, default=0.1)
    p.add_argument("--min_var", type=float, default=1e-4)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--limit_test_batches", type=int, default=None)
    p.add_argument("--figure_max_scalars", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _enable_mc_dropout(model: nn.Module, p: float) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            if getattr(m, "p", 0.0) == 0.0:
                m.p = p
            m.train()
            n += 1
    return n


def _attach_latent_dropout(backbone: nn.Module, p: float):
    target = getattr(backbone, "hid", None)
    if target is None:
        return None

    def _hook(_module, _inputs, output):
        return F.dropout(output, p=p, training=True)

    return target.register_forward_hook(_hook)


@torch.no_grad()
def _mc_predict(model: nn.Module, x: torch.Tensor, n: int,
                multi_frame: bool) -> tuple[torch.Tensor, torch.Tensor]:
    samples = []
    for _ in range(n):
        out = model(x)
        mu = out[:, 0]
        if not multi_frame and mu.dim() == 5:
            mu = mu[:, -1]
        samples.append(mu)
    s = torch.stack(samples, dim=0)
    mean = s.mean(dim=0)
    var = s.var(dim=0, unbiased=True).clamp(min=1e-8)
    return mean, torch.log(var)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("mc_dropout", str(output_dir / "mc_dropout.log"))
    logger.info("Args: %s", json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    train_ds = WeatherBenchDataset(
        data_root=args.data_root, split="train",
        in_len=args.in_len, out_len=args.out_len,
        variable=args.variable, data_split=args.data_split,
        norm_stats_path=str(output_dir / "norm_stats.json"),
    )
    test_ds = WeatherBenchDataset(
        data_root=args.data_root, split="test",
        in_len=args.in_len, out_len=args.out_len,
        variable=args.variable, data_split=args.data_split,
        norm_stats=(train_ds.mean, train_ds.std),
    )
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    sample_x, _ = train_ds[0]
    _, C, H, W = sample_x.shape
    backbone = _build_backbone(args.backbone, args.in_len, C, H, W)
    model = ProbWrapper(backbone, out_channels=C,
                        multi_frame=args.multi_frame).to(device)

    ckpt = torch.load(Path(args.init_from).resolve(),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    if args.backbone == "SimVP":
        handle = _attach_latent_dropout(model.backbone, args.mc_dropout_p)
        n_drop = 1 if handle is not None else 0
    else:
        n_drop = _enable_mc_dropout(model.backbone, args.mc_dropout_p)
    logger.info("MC dropout points: %d (p=%.3f)", n_drop, args.mc_dropout_p)

    log_std2 = math.log(train_ds.std * train_ds.std)
    log_min_var = math.log(args.min_var)
    accum = MetricAccumulator()
    per_lt = {} if args.multi_frame else None
    fig_mu, fig_lv, fig_tg = [], [], []
    n_fig = 0

    for i, (x, y) in enumerate(test_loader):
        if args.limit_test_batches is not None and i >= args.limit_test_batches:
            break
        x = x.to(device, non_blocking=True)
        y_sup = _select_target_frame(y, multi_frame=args.multi_frame).to(device)
        mu_z, lv_z = _mc_predict(model, x, args.mc_samples, args.multi_frame)
        mu = (mu_z.float() * train_ds.std + train_ds.mean).cpu()
        lv = (lv_z.float() + log_std2).clamp(min=log_min_var - log_std2).cpu()
        tg = (y_sup * train_ds.std + train_ds.mean).float().cpu()
        accum.update(mu, lv, tg)
        if per_lt is not None and mu.dim() == 5:
            for t in range(mu.shape[1]):
                per_lt.setdefault(t, MetricAccumulator()).update(
                    mu[:, t], lv[:, t], tg[:, t])
        if n_fig < args.figure_max_scalars:
            mflat = mu.reshape(-1)
            take = min(args.figure_max_scalars - n_fig, mflat.numel())
            fig_mu.append(mflat[:take])
            fig_lv.append(lv.reshape(-1)[:take])
            fig_tg.append(tg.reshape(-1)[:take])
            n_fig += take
        if (i + 1) % 100 == 0:
            logger.info("  %d batches", i + 1)

    test = accum.finalize()
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

    shape = (n_fig, 1, 1, 1)
    reliability_diagram(torch.cat(fig_mu).reshape(shape),
                        torch.cat(fig_lv).reshape(shape),
                        torch.cat(fig_tg).reshape(shape),
                        save_path=str(output_dir / "figures" / "reliability.png"))
    pit_histogram(torch.cat(fig_mu).reshape(shape),
                  torch.cat(fig_lv).reshape(shape),
                  torch.cat(fig_tg).reshape(shape),
                  save_path=str(output_dir / "figures" / "pit_histogram.png"))

    banner = _paper_eval_banner_probabilistic(
        test, dataname=f"weather_{args.variable}_{args.data_split}",
        loss_name=f"MC-Dropout (N={args.mc_samples})",
        backbone=args.backbone, epoch=int(ckpt.get("epoch", -1)),
    )
    print("\n" + banner)
    (output_dir / "paper_eval_summary.txt").write_text(banner + "\n")
    logger.info("Wrote %s", output_dir / "paper_eval_summary.txt")


if __name__ == "__main__":
    main()
