#!/usr/bin/env python3
"""Train ProbWrapper with Gaussian NLL or CRPS on WeatherBench t2m."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.calibration import pit_histogram, reliability_diagram
from src.dataset import WeatherBenchDataset
from src.losses import beta_gaussian_nll, gaussian_crps, gaussian_nll
from src.metrics import MetricAccumulator
from src.model import ProbWrapper
from src.utils import get_logger, save_checkpoint, set_seed


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the probabilistic training script."""

    p = argparse.ArgumentParser(
        description="Train a Gaussian probabilistic forecaster (ProbWrapper).",
    )
    p.add_argument("--backbone", type=str, default="SimVP",
                   choices=["SimVP", "TAU", "PredRNN", "ConvLSTM"])
    p.add_argument("--loss", type=str, default="nll",
                   choices=["nll", "crps", "beta_nll"],
                   help="Training objective.")
    p.add_argument("--beta", type=float, default=0.5,
                   help="β exponent for --loss beta_nll (Seitzer et al.; default 0.5).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="OneCycleLR max learning rate.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pct_start", type=float, default=0.2,
                   help="OneCycleLR warmup fraction.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping max norm (0 disables).")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience on val CRPS (0 disables).")
    p.add_argument("--min_epochs", type=int, default=15)
    p.add_argument("--multi_frame", action="store_true",
                   help="Supervise all output frames.")
    p.add_argument("--in_len", type=int, default=12)
    p.add_argument("--out_len", type=int, default=12)
    p.add_argument("--data_root", type=str, required=True,
                   help="Path containing ``weather_5_625deg/`` (e.g. ``OpenSTL/data``).")
    p.add_argument("--data_split", type=str, default="5_625")
    p.add_argument("--variable", type=str, default="t2m",
                   choices=["t2m", "z", "z500"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed precision (default: AMP fp16 on CUDA).")
    p.add_argument("--limit_train_batches", type=int, default=None,
                   help="Optional cap for smoke tests.")
    p.add_argument("--limit_val_batches", type=int, default=None)
    p.add_argument("--limit_test_batches", type=int, default=None)
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; load best.pth and run full test eval.")
    p.add_argument("--figure_max_scalars", type=int, default=200_000,
                   help="Max scalar samples for reliability/PIT figures.")
    p.add_argument("--init_from", type=str, default=None,
                   help="Optional checkpoint to warm-start ProbWrapper (e.g. "
                        "deterministic SimVP MSE ``best.ckpt``).")
    p.add_argument("--init_backbone_only", action="store_true", default=True,
                   help="Load only backbone weights from ``--init_from`` "
                        "(default: True).")
    p.add_argument("--no_init_backbone_only", action="store_false",
                   dest="init_backbone_only",
                   help="Load full ProbWrapper state dict from ``--init_from``.")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Train ``prob_head`` only (retrofit / warm-start variant b).")
    return p.parse_args()


class _PredRNNBackbone(torch.nn.Module):
    """Adapter so OpenSTL PredRNN matches the (B,T,C,H,W) backbone API."""

    def __init__(self, in_len: int, channels: int, height: int, width: int,
                 out_len: int, num_layers: int = 4, num_hidden: int = 64):
        super().__init__()
        from types import SimpleNamespace

        from openstl.models import PredRNN_Model

        self.in_len = in_len
        self.out_len = out_len
        self.channels = channels
        self.configs = SimpleNamespace(
            in_shape=(in_len, channels, height, width),
            patch_size=1, filter_size=5, stride=1, layer_norm=True,
            pre_seq_length=in_len, aft_seq_length=out_len,
            reverse_scheduled_sampling=0,
        )
        self.net = PredRNN_Model(num_layers, [num_hidden] * num_layers,
                                 self.configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T_in, C, H, W)`` to ``(B, T_out, C, H, W)``."""
        B, T_in, C, H, W = x.shape
        total = self.configs.pre_seq_length + self.configs.aft_seq_length
        frames = x.new_zeros(B, total, H, W, C)
        frames[:, :T_in] = x.permute(0, 1, 3, 4, 2)
        mask = x.new_zeros(B, self.configs.aft_seq_length - 1, H, W, C)
        out, _ = self.net(frames, mask, return_loss=False)  # (B, total-1, H,W,C)
        out = out[:, -self.out_len:].permute(0, 1, 4, 2, 3).contiguous()
        return out  # (B, T_out, C, H, W)


def _build_backbone(name: str, in_len: int, channels: int,
                    height: int, width: int) -> torch.nn.Module:
    """Build an OpenSTL backbone (unchanged architecture)."""

    if name == "SimVP":
        from openstl.models import SimVP_Model
        return SimVP_Model(
            in_shape=(in_len, channels, height, width),
            hid_S=32, hid_T=256, N_S=2, N_T=8,
            model_type="gSTA", mlp_ratio=8.0, drop=0.0, drop_path=0.1,
            spatio_kernel_enc=3, spatio_kernel_dec=3,
        )
    if name == "TAU":
        from openstl.models import SimVP_Model
        return SimVP_Model(
            in_shape=(in_len, channels, height, width),
            hid_S=32, hid_T=256, N_S=2, N_T=8,
            model_type="tau", mlp_ratio=8.0, drop=0.0, drop_path=0.1,
            spatio_kernel_enc=3, spatio_kernel_dec=3,
        )
    if name == "PredRNN":
        return _PredRNNBackbone(in_len, channels, height, width,
                                out_len=in_len)
    if name == "ConvLSTM":
        from src.model import SimpleConvLSTM
        return SimpleConvLSTM(
            in_shape=(in_len, channels, height, width),
            hidden_channels=32, num_layers=2, kernel_size=3,
        )
    raise ValueError(f"Unsupported backbone: {name}")


def _select_target_frame(y: torch.Tensor, multi_frame: bool = False
                         ) -> torch.Tensor:
    """Pick the supervision tensor matching ProbWrapper's output shape.

    Args:
        y: Ground truth ``(B, T_out, C, H, W)``.
        multi_frame: If True, supervise every output frame; otherwise only
            the last (legacy single-frame behavior).

    Returns:
        ``(B, T_out, C, H, W)`` when ``multi_frame`` else ``(B, C, H, W)``.
    """

    return y if multi_frame else y[:, -1]


def _denorm(t: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return t * std + mean


def _freeze_backbone(model: ProbWrapper) -> int:
    """Freeze backbone; leave ``prob_head`` trainable."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.prob_head.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_init_checkpoint(model: ProbWrapper, init_path: Path,
                          *, backbone_only: bool,
                          logger) -> tuple[list[str], list[str]]:
    """Load warm-start weights into ``ProbWrapper``."""
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unrecognized checkpoint format: {init_path}")

    if backbone_only:
        backbone_sd = model.backbone.state_dict()
        mapped: dict[str, torch.Tensor] = {}
        for key, val in state_dict.items():
            candidates = [key]
            if key.startswith("backbone."):
                candidates.append(key[len("backbone."):])
            if key.startswith("model."):
                candidates.append(key[len("model."):])
            if key.startswith("method.model."):
                candidates.append(key[len("method.model."):])
            for cand in candidates:
                if cand in backbone_sd:
                    mapped[cand] = val
                    break
        missing, unexpected = model.backbone.load_state_dict(mapped, strict=False)
        logger.info(
            "Warm-start backbone from %s (loaded=%d, missing=%d, unexpected=%d)",
            init_path, len(mapped), len(missing), len(unexpected),
        )
        if missing:
            logger.warning("Backbone missing keys (first 5): %s", missing[:5])
        return list(missing), list(unexpected)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Warm-start full model from %s (missing=%d, unexpected=%d)",
        init_path, len(missing), len(unexpected),
    )
    if missing:
        logger.warning("Missing keys (first 5): %s", missing[:5])
    return list(missing), list(unexpected)


def _run_validation(model: ProbWrapper, loader: DataLoader,
                    device: torch.device, mean: float, std: float,
                    limit_batches: int | None = None,
                    multi_frame: bool = False,
                    store_tensors: bool = False) -> dict:
    """Compute denormalized RMSE / MAE / CRPS / NLL / ECE on a loader.

    Uses online ``MetricAccumulator`` by default so full val/test splits do
    not exhaust system RAM. Set ``store_tensors=True`` only for the final
    test pass when reliability/PIT figures are needed.

    Args:
        model: Trained ``ProbWrapper`` (eval mode entered inside).
        loader: Validation/test DataLoader yielding ``(x, y)``.
        device: Compute device.
        mean: WeatherBench training mean used for denormalization.
        std: WeatherBench training std used for denormalization.
        limit_batches: Optional cap (smoke).
        multi_frame: If True, also report per lead-time metrics under
            ``per_leadtime``.
        store_tensors: If True, also return ``_mu, _lv, _tg`` (memory heavy).

    Returns:
        Dict with keys ``rmse, mae, crps, nll, ece`` (averaged across
        time when ``multi_frame``), optional ``per_leadtime``, and optional
        ``_mu, _lv, _tg`` tensors.
    """

    model.eval()
    accum = MetricAccumulator()
    per_lt: dict[int, MetricAccumulator] | None = (
        {} if multi_frame else None)
    means: list[torch.Tensor] = []
    log_vars: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    log_std2 = math.log(std * std)
    import gc
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            x = x.to(device, non_blocking=True)
            y_sup = _select_target_frame(y, multi_frame=multi_frame).to(
                device, non_blocking=True)
            out = model(x)
            mu = _denorm(out[:, 0], mean, std).float().cpu()
            lv = (out[:, 1].float() + log_std2).cpu()
            tg = _denorm(y_sup, mean, std).float().cpu()
            accum.update(mu, lv, tg)
            if per_lt is not None and mu.dim() == 5:
                for t in range(mu.shape[1]):
                    per_lt.setdefault(t, MetricAccumulator()).update(
                        mu[:, t], lv[:, t], tg[:, t])
            if store_tensors:
                means.append(mu)
                log_vars.append(lv)
                targets.append(tg)
            if (i + 1) % 500 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    result = accum.finalize()
    if per_lt is not None:
        result["per_leadtime"] = {
            t: acc.finalize() for t, acc in sorted(per_lt.items())}
    if store_tensors:
        result["_mu"] = torch.cat(means, dim=0)
        result["_lv"] = torch.cat(log_vars, dim=0)
        result["_tg"] = torch.cat(targets, dim=0)
    return result


def _collect_figure_tensors(
    model: ProbWrapper, loader: DataLoader, device: torch.device,
    mean: float, std: float, multi_frame: bool,
    max_scalars: int = 200_000,
    limit_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subsample scalar predictions for reliability/PIT plots (RAM-safe)."""

    model.eval()
    mu_parts: list[torch.Tensor] = []
    lv_parts: list[torch.Tensor] = []
    tg_parts: list[torch.Tensor] = []
    n_scalar = 0
    log_std2 = math.log(std * std)
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            x = x.to(device, non_blocking=True)
            y_sup = _select_target_frame(y, multi_frame=multi_frame).to(
                device, non_blocking=True)
            out = model(x)
            mu = _denorm(out[:, 0], mean, std).float().reshape(-1).cpu()
            lv = (out[:, 1].float() + log_std2).reshape(-1).cpu()
            tg = _denorm(y_sup, mean, std).float().reshape(-1).cpu()
            remaining = max_scalars - n_scalar
            if remaining <= 0:
                break
            if mu.numel() > remaining:
                mu, lv, tg = mu[:remaining], lv[:remaining], tg[:remaining]
            mu_parts.append(mu)
            lv_parts.append(lv)
            tg_parts.append(tg)
            n_scalar += mu.numel()
    mu_all = torch.cat(mu_parts)
    # Reshape to (N, 1, 1, 1) so calibration helpers accept (N, C, H, W).
    n = mu_all.numel()
    shape = (n, 1, 1, 1)
    return (
        mu_all.reshape(shape),
        torch.cat(lv_parts).reshape(shape),
        torch.cat(tg_parts).reshape(shape),
    )


def _paper_eval_banner_probabilistic(metrics: dict, *, dataname: str,
                                     loss_name: str, backbone: str,
                                     epoch: int,
                                     variable: str = "t2m",
                                     unit_label: str = "K",
                                     display_name: str = "t2m (2 m temperature)"
                                     ) -> str:
    """Render a paper-style summary block for the probabilistic test pass."""

    lines = [
        "=" * 78,
        f"PAPER EVAL — ProbWrapper({backbone}) + {loss_name.upper()} (physical units)",
        "=" * 78,
        f"  dataname   : {dataname}",
        f"  variable   : {display_name}",
        "  scaling    : denormalized (WeatherBench train mean / std)",
        f"  best epoch : {epoch}  (selected by val CRPS)",
        "-" * 78,
        "  Point forecasts (from predictive mean):",
        f"    MAE      ({unit_label}) : {metrics['mae']:.6f}",
        f"    RMSE     ({unit_label}) : {metrics['rmse']:.6f}",
        "-" * 78,
        "  Probabilistic metrics (Gaussian predictive distribution):",
        f"    CRPS     ({unit_label}) : {metrics['crps']:.6f}",
        f"    NLL              : {metrics['nll']:.6f}",
        f"    ECE      [0,1]   : {metrics['ece']:.6f}",
        "=" * 78,
    ]
    return "\n".join(lines)


def main() -> None:
    """Train, validate, and test a probabilistic forecaster end-to-end."""

    args = _parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "paper_eval_summary.txt"
    norm_stats_path = output_dir / "norm_stats.json"
    logger = get_logger("prob", str(log_path))
    logger.info("Args: %s", json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"

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
    logger.info("Train=%d  Val=%d  Test=%d", len(train_ds), len(val_ds), len(test_ds))
    logger.info("Norm stats: mean=%.6f std=%.6f", train_ds.mean, train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    sample_x, sample_y = train_ds[0]
    _, C, H, W = sample_x.shape
    logger.info("Sample shapes  x=%s  y=%s", tuple(sample_x.shape), tuple(sample_y.shape))

    logger.info("Building model: %s + ProbWrapper (multi_frame=%s)",
                args.backbone, args.multi_frame)
    backbone = _build_backbone(args.backbone, args.in_len, C, H, W)
    model = ProbWrapper(backbone, out_channels=C,
                        multi_frame=args.multi_frame).to(device)

    if args.init_from:
        _load_init_checkpoint(
            model, Path(args.init_from).resolve(),
            backbone_only=args.init_backbone_only, logger=logger,
        )
    if args.freeze_backbone:
        n_trainable = _freeze_backbone(model)
        logger.info("Frozen backbone; trainable params: %.2fM / %.2fM",
                    n_trainable / 1e6,
                    sum(p.numel() for p in model.parameters()) / 1e6)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM (trainable %.2fM)",
                n_params / 1e6, sum(p.numel() for p in trainable_params) / 1e6)

    optim = torch.optim.AdamW(trainable_params, lr=args.lr,
                              weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // max(1, args.accum_steps))
    if args.limit_train_batches is not None:
        steps_per_epoch = max(1, min(steps_per_epoch,
                                     args.limit_train_batches // max(1, args.accum_steps)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=args.pct_start,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.loss == "nll":
        loss_fn = gaussian_nll
    elif args.loss == "crps":
        loss_fn = gaussian_crps
    else:
        beta = args.beta
        loss_fn = lambda mean, log_var, target: beta_gaussian_nll(
            mean, log_var, target, beta=beta)
    loss_tag = args.loss if args.loss != "beta_nll" else f"beta_nll(b={args.beta})"

    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_rmse", "val_mae",
            "val_crps", "val_nll", "val_ece", "lr", "elapsed_sec",
        ])

    best_crps = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    dataname = f"weather_{args.variable}_{args.data_split}"

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            optim.zero_grad(set_to_none=True)
            t0 = time.time()
            running = 0.0
            n_seen = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]",
                        file=sys.stdout, dynamic_ncols=True)
            for step, (x, y) in enumerate(pbar):
                if args.limit_train_batches is not None and step >= args.limit_train_batches:
                    break
                x = x.to(device, non_blocking=True)
                y_sup = _select_target_frame(y, multi_frame=args.multi_frame).to(
                    device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                        enabled=use_amp):
                    out = model(x)
                    loss = loss_fn(out[:, 0], out[:, 1], y_sup) / args.accum_steps
                scaler.scale(loss).backward()
                running += float(loss.detach()) * args.accum_steps * x.size(0)
                n_seen += x.size(0)
                if (step + 1) % args.accum_steps == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=args.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    if scheduler.last_epoch < scheduler.total_steps - 1:
                        scheduler.step()
                pbar.set_postfix({
                    "loss": f"{running / max(1, n_seen):.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                })
            train_loss = running / max(1, n_seen)

            val = _run_validation(model, val_loader, device,
                                  train_ds.mean, train_ds.std,
                                  limit_batches=args.limit_val_batches,
                                  multi_frame=args.multi_frame)
            elapsed = time.time() - t0
            epoch_line = (
                f"Epoch {epoch:03d} | train_{loss_tag}={train_loss:.4f} "
                f"| val_rmse={val['rmse']:.4f}{train_ds.unit_label} "
                f"val_mae={val['mae']:.4f}{train_ds.unit_label} "
                f"val_crps={val['crps']:.4f}{train_ds.unit_label} val_nll={val['nll']:.4f} "
                f"val_ece={val['ece']:.4f} | lr={optim.param_groups[0]['lr']:.2e} "
                f"| {elapsed:.1f}s"
            )
            logger.info(epoch_line)

            with csv_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, f"{train_loss:.6f}", f"{val['rmse']:.6f}",
                    f"{val['mae']:.6f}", f"{val['crps']:.6f}", f"{val['nll']:.6f}",
                    f"{val['ece']:.6f}", f"{optim.param_groups[0]['lr']:.6e}",
                    f"{elapsed:.2f}",
                ])

            is_best = val["crps"] < best_crps
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "val_crps": val["crps"],
                "args": vars(args),
                "norm_stats": {"mean": train_ds.mean, "std": train_ds.std},
            }
            save_checkpoint(state, str(output_dir / "checkpoints" / "last.pth"),
                            is_best=is_best)
            if is_best:
                best_crps = val["crps"]
                best_epoch = epoch
                epochs_since_improve = 0
                logger.info("  ↳ new best (val CRPS=%.4f K) saved to best.pth", best_crps)
            else:
                epochs_since_improve += 1

            if (args.patience > 0 and epoch >= args.min_epochs
                    and epochs_since_improve >= args.patience):
                logger.info(
                    "Early stop at epoch %d: val_crps has not improved for %d "
                    "epochs (best=%d, best_val_crps=%.4f K).",
                    epoch, epochs_since_improve, best_epoch, best_crps,
                )
                break

        logger.info("Training done. best_epoch=%d best_val_crps=%.4f K",
                    best_epoch, best_crps)
    else:
        ckpt_meta = torch.load(
            output_dir / "checkpoints" / "best.pth",
            map_location="cpu", weights_only=False)
        best_epoch = int(ckpt_meta.get("epoch", -1))
        best_crps = float(ckpt_meta.get("val_crps", float("nan")))
        logger.info("eval_only: skipping training (best_epoch=%d val_crps=%.4f K)",
                    best_epoch, best_crps)

    logger.info("Loading best.pth and running TEST evaluation ...")
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ckpt = torch.load(output_dir / "checkpoints" / "best.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test = _run_validation(model, test_loader, device,
                           train_ds.mean, train_ds.std,
                           limit_batches=args.limit_test_batches,
                           multi_frame=args.multi_frame,
                           store_tensors=False)
    fig_mu, fig_lv, fig_tg = _collect_figure_tensors(
        model, test_loader, device, train_ds.mean, train_ds.std,
        multi_frame=args.multi_frame,
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

    if "per_leadtime" in test:
        lt_csv = output_dir / "per_leadtime.csv"
        with lt_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["leadtime", "rmse", "mae", "crps", "nll", "ece"])
            for t, m in sorted(test["per_leadtime"].items()):
                w.writerow([t, f"{m['rmse']:.6f}", f"{m['mae']:.6f}",
                            f"{m['crps']:.6f}", f"{m['nll']:.6f}",
                            f"{m['ece']:.6f}"])
        logger.info("Per lead-time metrics written to %s", lt_csv)

    banner = _paper_eval_banner_probabilistic(
        test, dataname=dataname, loss_name=loss_tag,
        backbone=args.backbone, epoch=best_epoch,
        variable=args.variable, unit_label=train_ds.unit_label,
        display_name=train_ds.display_name,
    )
    print("\n" + banner)
    print(f"compact: rmse:{test['rmse']:.6f}, mae:{test['mae']:.6f}, "
          f"crps:{test['crps']:.6f}, nll:{test['nll']:.6f}, "
          f"ece:{test['ece']:.6f}")

    with summary_path.open("w") as f:
        f.write(banner + "\n\n")
        f.write(
            f"compact: rmse:{test['rmse']}, mae:{test['mae']}, "
            f"crps:{test['crps']}, nll:{test['nll']}, ece:{test['ece']}\n"
        )
    logger.info("Wrote paper summary to %s", summary_path)
    logger.info("All artifacts under: %s", output_dir)


if __name__ == "__main__":
    main()
