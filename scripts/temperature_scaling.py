#!/usr/bin/env python3
"""Post-hoc temperature scaling on the validation set (optional calibration)."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
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
    """Parse CLI arguments for temperature scaling."""

    p = argparse.ArgumentParser(
        description="Fit a scalar temperature T on the validation set and "
                    "re-evaluate the test split with the calibrated variance.",
    )
    p.add_argument("--init_from", type=str, required=True,
                   help="Path to a trained ProbWrapper ``best.pth`` "
                        "(CRPS or NLL-FT).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--data_split", type=str, default="5_625")
    p.add_argument("--variable", type=str, default="t2m")

    p.add_argument("--backbone", type=str, default="SimVP",
                   choices=["SimVP", "TAU", "PredRNN", "ConvLSTM"])
    p.add_argument("--in_len", type=int, default=12)
    p.add_argument("--out_len", type=int, default=12)
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--multi_frame", action="store_true",
                   help="Must match the flag used when the init checkpoint "
                        "was trained.")
    p.add_argument("--max_iter", type=int, default=200,
                   help="LBFGS max iterations for the 1-D fit.")
    p.add_argument("--lr", type=float, default=0.1,
                   help="LBFGS learning rate for the 1-D fit.")
    p.add_argument("--sweep", action="store_true",
                   help="Grid-search T on validation scalars; pick ECE-min T.")
    p.add_argument("--sweep_min", type=float, default=0.5)
    p.add_argument("--sweep_max", type=float, default=3.0)
    p.add_argument("--sweep_n", type=int, default=51)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit_val_batches", type=int, default=None)
    p.add_argument("--limit_test_batches", type=int, default=None)
    p.add_argument("--figure_max_scalars", type=int, default=200_000,
                   help="Max scalar samples for reliability/PIT (RAM-safe).")
    p.add_argument("--fit_max_scalars", type=int, default=500_000,
                   help="Validation scalars kept for T fit (subsample; RAM-safe).")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip T fit; load ``temperature.json`` and run test.")
    return p.parse_args()


@torch.no_grad()
def _collect_val_flat(model: ProbWrapper, loader: DataLoader, device: torch.device,
                      mean: float, std: float, multi_frame: bool,
                      max_scalars: int,
                      limit_batches: int | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect up to ``max_scalars`` validation predictions (float16, flat)."""

    model.eval()
    means, log_vars, targets = [], [], []
    n_scalar = 0
    log_std2 = math.log(std * std)
    for i, (x, y) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        if n_scalar >= max_scalars:
            break
        x = x.to(device, non_blocking=True)
        y_sup = _select_target_frame(y, multi_frame=multi_frame).to(
            device, non_blocking=True)
        out = model(x)
        mu = (out[:, 0] * std + mean).float().reshape(-1).cpu().to(torch.float16)
        lv = (out[:, 1].float() + log_std2).reshape(-1).cpu().to(torch.float16)
        tg = (y_sup * std + mean).float().reshape(-1).cpu().to(torch.float16)
        remaining = max_scalars - n_scalar
        if mu.numel() > remaining:
            mu, lv, tg = mu[:remaining], lv[:remaining], tg[:remaining]
        means.append(mu)
        log_vars.append(lv)
        targets.append(tg)
        n_scalar += mu.numel()
        if (i + 1) % 500 == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(means), torch.cat(log_vars), torch.cat(targets)


def _chunk_nll_sum(mu: torch.Tensor, lv: torch.Tensor, tg: torch.Tensor,
                     log_T: torch.Tensor, chunk: int) -> torch.Tensor:
    """Sum Gaussian NLL over flat tensors in chunks (only ``log_T`` grad)."""

    offset = 2.0 * log_T
    total = torch.zeros((), dtype=torch.float32)
    n = mu.numel()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        mu_c = mu[start:end].float()
        lv_c = lv[start:end].float() + offset
        tg_c = tg[start:end].float()
        total = total + 0.5 * (
            lv_c + (tg_c - mu_c).pow(2) / lv_c.exp()
        ).sum()
    return total / max(1, n)


def _fit_temperature(mu: torch.Tensor, lv: torch.Tensor, tg: torch.Tensor,
                     lr: float, max_iter: int, chunk: int, logger) -> float:
    """Fit scalar T (via log_T) by minimizing chunked Gaussian NLL."""

    log_T = torch.zeros(1, requires_grad=True)
    optim = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter,
                              line_search_fn="strong_wolfe")

    with torch.no_grad():
        nll_before = _chunk_nll_sum(
            mu, lv, tg, torch.zeros(1), chunk).item()

    def closure() -> torch.Tensor:
        optim.zero_grad()
        nll = _chunk_nll_sum(mu, lv, tg, log_T, chunk)
        nll.backward()
        return nll

    optim.step(closure)
    T = float(log_T.detach().exp().item())
    with torch.no_grad():
        nll_after = _chunk_nll_sum(
            mu, lv, tg, log_T.detach(), chunk).item()
    logger.info("Temperature fit: T=%.4f  (val NLL: %.4f -> %.4f)",
                T, nll_before, nll_after)
    return T


def _sweep_temperature(mu: torch.Tensor, lv: torch.Tensor, tg: torch.Tensor,
                       t_min: float, t_max: float, n: int, chunk: int,
                       logger) -> tuple[float, list[dict]]:
    """Grid-search T; pick ECE-min on flattened validation scalars."""

    Ts = torch.linspace(t_min, t_max, n).tolist()
    rows: list[dict] = []
    best_ece, best_ece_T = float("inf"), 1.0
    best_nll, best_nll_T = float("inf"), 1.0
    accum = MetricAccumulator()
    for T in Ts:
        accum = MetricAccumulator()
        offset = 2.0 * math.log(T)
        n_scalars = mu.numel()
        for start in range(0, n_scalars, chunk):
            end = min(start + chunk, n_scalars)
            mu_c = mu[start:end].float().reshape(-1, 1, 1, 1)
            lv_c = (lv[start:end].float() + offset).reshape(-1, 1, 1, 1)
            tg_c = tg[start:end].float().reshape(-1, 1, 1, 1)
            accum.update(mu_c, lv_c, tg_c)
        m = accum.finalize()
        row = {"T": T, **m}
        rows.append(row)
        if m["ece"] < best_ece:
            best_ece, best_ece_T = m["ece"], T
        if m["nll"] < best_nll:
            best_nll, best_nll_T = m["nll"], T
    base = next((r for r in rows if abs(r["T"] - 1.0) < 1e-6), None)
    if base is not None:
        logger.info(
            "Sweep baseline T=1.000  rmse=%.4f crps=%.4f nll=%.4f ece=%.4f",
            base["rmse"], base["crps"], base["nll"], base["ece"],
        )
    logger.info("Sweep best by ECE : T=%.4f  ece=%.4f", best_ece_T, best_ece)
    logger.info("Sweep best by NLL : T=%.4f  nll=%.4f", best_nll_T, best_nll)
    return best_ece_T, rows


@torch.no_grad()
def _eval_streaming(model: ProbWrapper, loader: DataLoader, device: torch.device,
                    mean: float, std: float, multi_frame: bool,
                    log_T: float = 1.0,
                    limit_batches: int | None = None) -> dict:
    """Stream denormalized metrics; optionally apply ``log_var += 2 log T``."""

    model.eval()
    accum = MetricAccumulator()
    per_lt: dict[int, MetricAccumulator] | None = (
        {} if multi_frame else None)
    log_std2 = math.log(std * std)
    lv_offset = 2.0 * math.log(log_T)
    for i, (x, y) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y_sup = _select_target_frame(y, multi_frame=multi_frame).to(
            device, non_blocking=True)
        out = model(x)
        mu = (out[:, 0] * std + mean).float().cpu()
        lv = (out[:, 1].float() + log_std2 + lv_offset).cpu()
        tg = (y_sup * std + mean).float().cpu()
        accum.update(mu, lv, tg)
        if per_lt is not None and mu.dim() == 5:
            for t in range(mu.shape[1]):
                per_lt.setdefault(t, MetricAccumulator()).update(
                    mu[:, t], lv[:, t], tg[:, t])
        if (i + 1) % 500 == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result = accum.finalize()
    if per_lt is not None:
        result["per_leadtime"] = {
            t: acc.finalize() for t, acc in sorted(per_lt.items())}
    return result


@torch.no_grad()
def _collect_figure_tensors(
    model: ProbWrapper, loader: DataLoader, device: torch.device,
    mean: float, std: float, multi_frame: bool, log_T: float,
    max_scalars: int = 200_000,
    limit_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subsample calibrated scalars for reliability/PIT plots."""

    model.eval()
    mu_parts: list[torch.Tensor] = []
    lv_parts: list[torch.Tensor] = []
    tg_parts: list[torch.Tensor] = []
    n_scalar = 0
    log_std2 = math.log(std * std)
    lv_offset = 2.0 * math.log(log_T)
    for i, (x, y) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y_sup = _select_target_frame(y, multi_frame=multi_frame).to(
            device, non_blocking=True)
        out = model(x)
        mu = (out[:, 0] * std + mean).float().reshape(-1).cpu()
        lv = (out[:, 1].float() + log_std2 + lv_offset).reshape(-1).cpu()
        tg = (y_sup * std + mean).float().reshape(-1).cpu()
        remaining = max_scalars - n_scalar
        if remaining <= 0:
            break
        if mu.numel() > remaining:
            mu, lv, tg = mu[:remaining], lv[:remaining], tg[:remaining]
        mu_parts.append(mu)
        lv_parts.append(lv)
        tg_parts.append(tg)
        n_scalar += mu.numel()
    if not mu_parts:
        raise ValueError("No samples collected for figure tensors.")
    shape = (n_scalar, 1, 1, 1)
    return (
        torch.cat(mu_parts).reshape(shape),
        torch.cat(lv_parts).reshape(shape),
        torch.cat(tg_parts).reshape(shape),
    )


def main() -> None:
    """Run temperature scaling end-to-end and write a paper banner."""

    args = _parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "temperature.log"
    summary_path = output_dir / "paper_eval_summary.txt"
    norm_stats_path = output_dir / "norm_stats.json"
    temp_json_path = output_dir / "temperature.json"
    logger = get_logger("temp_scale", str(log_path))
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
    val_ds = WeatherBenchDataset(
        data_root=args.data_root, split="val",
        in_len=args.in_len, out_len=args.out_len,
        variable=args.variable, data_split=args.data_split,
        norm_stats=norm_stats,
    )
    test_ds = WeatherBenchDataset(
        data_root=args.data_root, split="test",
        in_len=args.in_len, out_len=args.out_len,
        variable=args.variable, data_split=args.data_split,
        norm_stats=norm_stats,
    )
    logger.info("Val=%d  Test=%d", len(val_ds), len(test_ds))
    logger.info("Norm stats: mean=%.6f std=%.6f", train_ds.mean, train_ds.std)

    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    sample_x, _ = train_ds[0]
    _, C, H, W = sample_x.shape

    logger.info("Building model: %s + ProbWrapper (multi_frame=%s)",
                args.backbone, args.multi_frame)
    backbone = _build_backbone(args.backbone, args.in_len, C, H, W)
    model = ProbWrapper(backbone, out_channels=C,
                        multi_frame=args.multi_frame).to(device)

    init_path = Path(args.init_from).resolve()
    if not init_path.is_file():
        raise FileNotFoundError(f"--init_from not found: {init_path}")
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded init from %s", init_path)

    if args.eval_only:
        if not temp_json_path.is_file():
            raise FileNotFoundError(
                f"--eval_only requires {temp_json_path}; fit T first.")
        with temp_json_path.open() as f:
            T = float(json.load(f)["T"])
        logger.info("eval_only: loaded T=%.4f from %s", T, temp_json_path)
    else:
        logger.info("Collecting validation subsample for T fit (max=%d) ...",
                    args.fit_max_scalars)
        mu_v, lv_v, tg_v = _collect_val_flat(
            model, val_loader, device, train_ds.mean, train_ds.std,
            multi_frame=args.multi_frame,
            max_scalars=args.fit_max_scalars,
            limit_batches=args.limit_val_batches,
        )
        logger.info("Val fit subsample: n=%d scalars", mu_v.numel())

        if args.sweep:
            T, sweep_rows = _sweep_temperature(
                mu_v, lv_v, tg_v,
                t_min=args.sweep_min, t_max=args.sweep_max, n=args.sweep_n,
                chunk=min(args.fit_max_scalars, 500_000), logger=logger,
            )
            sweep_csv = output_dir / "temperature_sweep.csv"
            with sweep_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["T", "val_rmse", "val_mae", "val_crps",
                            "val_nll", "val_ece"])
                for row in sweep_rows:
                    w.writerow([f"{row['T']:.4f}",
                                f"{row['rmse']:.6f}", f"{row['mae']:.6f}",
                                f"{row['crps']:.6f}", f"{row['nll']:.6f}",
                                f"{row['ece']:.6f}"])
            logger.info("Sweep results written to %s", sweep_csv)
        else:
            T = _fit_temperature(
                mu_v, lv_v, tg_v, lr=args.lr, max_iter=args.max_iter,
                chunk=min(mu_v.numel(), 500_000), logger=logger)

        del mu_v, lv_v, tg_v
        gc.collect()

        val_post = _eval_streaming(
            model, val_loader, device, train_ds.mean, train_ds.std,
            multi_frame=args.multi_frame, log_T=T,
            limit_batches=args.limit_val_batches)
        logger.info(
            "POST val_rmse=%.4fK val_mae=%.4fK val_crps=%.4fK "
            "val_nll=%.4f val_ece=%.4f",
            val_post["rmse"], val_post["mae"], val_post["crps"],
            val_post["nll"], val_post["ece"],
        )

        with temp_json_path.open("w") as f:
            json.dump({"T": T, "init_from": str(init_path)}, f, indent=2)
        logger.info("Saved T=%.4f to %s", T, temp_json_path)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Evaluating TEST (T=%.4f, streaming) ...", T)
    test_pre = _eval_streaming(
        model, test_loader, device, train_ds.mean, train_ds.std,
        multi_frame=args.multi_frame, log_T=1.0,
        limit_batches=args.limit_test_batches)
    test_post = _eval_streaming(
        model, test_loader, device, train_ds.mean, train_ds.std,
        multi_frame=args.multi_frame, log_T=T,
        limit_batches=args.limit_test_batches)

    fig_mu, fig_lv, fig_tg = _collect_figure_tensors(
        model, test_loader, device, train_ds.mean, train_ds.std,
        multi_frame=args.multi_frame, log_T=T,
        max_scalars=args.figure_max_scalars,
        limit_batches=args.limit_test_batches,
    )
    reliability_diagram(
        fig_mu, fig_lv, fig_tg,
        save_path=str(output_dir / "figures" / "reliability.png"),
    )
    pit_histogram(
        fig_mu, fig_lv, fig_tg,
        save_path=str(output_dir / "figures" / "pit_histogram.png"),
    )

    if "per_leadtime" in test_post:
        lt_csv = output_dir / "per_leadtime.csv"
        with lt_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["leadtime", "rmse", "mae", "crps", "nll", "ece"])
            for t, m in sorted(test_post["per_leadtime"].items()):
                w.writerow([t, f"{m['rmse']:.6f}", f"{m['mae']:.6f}",
                            f"{m['crps']:.6f}", f"{m['nll']:.6f}",
                            f"{m['ece']:.6f}"])
        logger.info("Per lead-time metrics written to %s", lt_csv)

    dataname = f"weather_{args.variable}_{args.data_split}"
    banner = _paper_eval_banner_probabilistic(
        test_post, dataname=dataname, loss_name=f"temp-scaled (T={T:.4f})",
        backbone=args.backbone, epoch=int(ckpt.get("epoch", -1)),
    )
    print("\n" + banner)
    print(f"compact: rmse:{test_post['rmse']:.6f}, mae:{test_post['mae']:.6f}, "
          f"crps:{test_post['crps']:.6f}, nll:{test_post['nll']:.6f}, "
          f"ece:{test_post['ece']:.6f}")

    with summary_path.open("w") as f:
        f.write(banner + "\n\n")
        f.write("Calibration summary (temperature scaling)\n")
        f.write(f"  T = {T:.6f}\n")
        f.write("  TEST before  : "
                f"rmse={test_pre['rmse']:.4f} mae={test_pre['mae']:.4f} "
                f"crps={test_pre['crps']:.4f} nll={test_pre['nll']:.4f} "
                f"ece={test_pre['ece']:.4f}\n")
        f.write("  TEST after   : "
                f"rmse={test_post['rmse']:.4f} mae={test_post['mae']:.4f} "
                f"crps={test_post['crps']:.4f} nll={test_post['nll']:.4f} "
                f"ece={test_post['ece']:.4f}\n\n")
        f.write(
            f"compact: rmse:{test_post['rmse']}, mae:{test_post['mae']}, "
            f"crps:{test_post['crps']}, nll:{test_post['nll']}, "
            f"ece:{test_post['ece']}\n"
        )

    logger.info("Wrote paper summary to %s (T=%.4f)", summary_path, T)
    logger.info("All artifacts under: %s", output_dir)


if __name__ == "__main__":
    main()
