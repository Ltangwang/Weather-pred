#!/usr/bin/env python3
"""Stage-2 NLL fine-tuning for a CRPS-trained ProbWrapper (frozen backbone)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_probabilistic import (
    _build_backbone,
    _collect_figure_tensors,
    _paper_eval_banner_probabilistic,
    _run_validation,
    _select_target_frame,
)
from src.calibration import pit_histogram, reliability_diagram
from src.dataset import WeatherBenchDataset
from src.losses import gaussian_nll
from src.model import ProbWrapper
from src.utils import get_logger, save_checkpoint, set_seed


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the NLL fine-tuning script."""

    p = argparse.ArgumentParser(
        description="Stage-2 NLL fine-tune of a CRPS-trained ProbWrapper.",
    )
    p.add_argument("--init_from", type=str, default=None,
                   help="Path to the Stage-1 (CRPS) ``best.pth`` checkpoint. "
                        "Not required with ``--eval_only``.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--data_split", type=str, default="5_625")
    p.add_argument("--variable", type=str, default="t2m",
                   choices=["t2m", "z", "z500"])

    p.add_argument("--backbone", type=str, default="SimVP",
                   choices=["SimVP", "TAU", "PredRNN", "ConvLSTM"])
    p.add_argument("--in_len", type=int, default=12)
    p.add_argument("--out_len", type=int, default=12)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pct_start", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--unfreeze_backbone_last", action="store_true",
                   help="Also train the last backbone block.")
    p.add_argument("--multi_frame", action="store_true",
                   help="Must match the flag used when the init checkpoint "
                        "was trained. Required when fine-tuning a multi-frame "
                        "ProbWrapper.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--limit_train_batches", type=int, default=None)
    p.add_argument("--limit_val_batches", type=int, default=None)
    p.add_argument("--limit_test_batches", type=int, default=None)
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; load best.pth and run test + figures.")
    p.add_argument("--figure_max_scalars", type=int, default=200_000,
                   help="Max scalar samples for reliability/PIT (RAM-safe).")
    return p.parse_args()


def _freeze_backbone(model: ProbWrapper, unfreeze_last: bool) -> int:
    """Freeze all parameters except ``prob_head`` (and optionally the very
    last backbone module).

    Args:
        model: ``ProbWrapper`` wrapping an OpenSTL backbone.
        unfreeze_last: Whether to also leave the last named child of
            ``model.backbone`` trainable (commonly the decoder tail).

    Returns:
        Number of trainable parameters after freezing.
    """

    for p in model.backbone.parameters():
        p.requires_grad = False

    if unfreeze_last:
        last_name = None
        for name, _ in model.backbone.named_children():
            last_name = name
        if last_name is not None:
            for p in getattr(model.backbone, last_name).parameters():
                p.requires_grad = True

    for p in model.prob_head.parameters():
        p.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    """Run a short NLL fine-tune on top of a CRPS checkpoint."""

    args = _parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "paper_eval_summary.txt"
    norm_stats_path = output_dir / "norm_stats.json"
    logger = get_logger("nll_ft", str(log_path))
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

    sample_x, _ = train_ds[0]
    _, C, H, W = sample_x.shape

    logger.info("Building model: %s + ProbWrapper (multi_frame=%s)",
                args.backbone, args.multi_frame)
    backbone = _build_backbone(args.backbone, args.in_len, C, H, W)
    model = ProbWrapper(backbone, out_channels=C,
                        multi_frame=args.multi_frame).to(device)

    if args.eval_only:
        best_path = output_dir / "checkpoints" / "best.pth"
        if not best_path.is_file():
            raise FileNotFoundError(
                f"--eval_only requires {best_path}; run training first.")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = int(ckpt.get("epoch", -1))
        best_nll = float(ckpt.get("val_nll", float("nan")))
        logger.info("eval_only: loaded best.pth (epoch=%d val_nll=%.4f)",
                    best_epoch, best_nll)
    else:
        if not args.init_from:
            raise ValueError("--init_from is required unless --eval_only is set.")
        init_path = Path(args.init_from).resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"--init_from not found: {init_path}")
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded init from %s (missing=%d, unexpected=%d)",
                    init_path, len(missing), len(unexpected))
        if missing:
            logger.warning("Missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:5])

    n_trainable = _freeze_backbone(model, unfreeze_last=args.unfreeze_backbone_last)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params after freezing: %.2fM / %.2fM (%.1f%%)",
                n_trainable / 1e6, n_total / 1e6,
                100.0 * n_trainable / max(1, n_total))

    dataname = f"weather_{args.variable}_{args.data_split}"

    if not args.eval_only:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
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

        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_rmse", "val_mae",
                "val_crps", "val_nll", "val_ece", "lr", "elapsed_sec",
            ])

        # Stage-2 best is selected by val NLL (we are optimizing calibration).
        best_nll = float("inf")
        best_epoch = -1

        # Eval the loaded checkpoint once before any fine-tune step, so users can
        # see the delta the fine-tune produces.
        pre = _run_validation(model, val_loader, device, train_ds.mean, train_ds.std,
                              limit_batches=args.limit_val_batches,
                              multi_frame=args.multi_frame)
        logger.info(
            "PRE-FT  val_rmse=%.4fK val_mae=%.4fK val_crps=%.4fK "
            "val_nll=%.4f val_ece=%.4f",
            pre["rmse"], pre["mae"], pre["crps"], pre["nll"], pre["ece"],
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            # Backbone stays in eval mode so BN/Dropout running stats don't drift
            # while frozen.
            model.backbone.eval()
            optim.zero_grad(set_to_none=True)
            t0 = time.time()
            running = 0.0
            n_seen = 0
            pbar = tqdm(train_loader, desc=f"FT {epoch}/{args.epochs}",
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
                    loss = gaussian_nll(out[:, 0], out[:, 1], y_sup) / args.accum_steps
                scaler.scale(loss).backward()
                running += float(loss.detach()) * args.accum_steps * x.size(0)
                n_seen += x.size(0)
                if (step + 1) % args.accum_steps == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    if scheduler.last_epoch < scheduler.total_steps - 1:
                        scheduler.step()
                pbar.set_postfix({
                    "nll": f"{running / max(1, n_seen):.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                })
            train_loss = running / max(1, n_seen)

            val = _run_validation(model, val_loader, device,
                                  train_ds.mean, train_ds.std,
                                  limit_batches=args.limit_val_batches,
                                  multi_frame=args.multi_frame)
            elapsed = time.time() - t0
            logger.info(
                "FT %03d | train_nll=%.4f | val_rmse=%.4fK val_mae=%.4fK "
                "val_crps=%.4fK val_nll=%.4f val_ece=%.4f | lr=%.2e | %.1fs",
                epoch, train_loss, val["rmse"], val["mae"], val["crps"],
                val["nll"], val["ece"], optim.param_groups[0]["lr"], elapsed,
            )

            with csv_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, f"{train_loss:.6f}", f"{val['rmse']:.6f}",
                    f"{val['mae']:.6f}", f"{val['crps']:.6f}", f"{val['nll']:.6f}",
                    f"{val['ece']:.6f}", f"{optim.param_groups[0]['lr']:.6e}",
                    f"{elapsed:.2f}",
                ])

            is_best = val["nll"] < best_nll
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "val_nll": val["nll"],
                "args": vars(args),
                "norm_stats": {"mean": train_ds.mean, "std": train_ds.std},
            }
            save_checkpoint(state, str(output_dir / "checkpoints" / "last.pth"),
                            is_best=is_best)
            if is_best:
                best_nll = val["nll"]
                best_epoch = epoch
                logger.info("  ↳ new best (val NLL=%.4f) saved to best.pth", best_nll)

        logger.info("Fine-tune done. best_epoch=%d best_val_nll=%.4f",
                    best_epoch, best_nll)

    logger.info("Loading best.pth and running TEST evaluation ...")
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not args.eval_only:
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
        test, dataname=dataname, loss_name="nll-ft",
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
