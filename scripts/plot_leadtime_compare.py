#!/usr/bin/env python3
"""Plot overlaid lead-time curves from several per_leadtime.csv files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    p = argparse.ArgumentParser(description="Multi-method lead-time curves.")
    p.add_argument("--results_root", type=str, required=True)
    p.add_argument("--series", action="append", required=True,
                   help="``LABEL,relative/path/to/per_leadtime.csv`` "
                        "(repeatable).")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--hours_per_step", type=int, default=6)
    return p.parse_args()


def _read(csv_path: Path) -> list[dict]:
    """Load one lead-time CSV."""

    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append({k: float(r[k]) if k != "leadtime" else int(r["leadtime"])
                         for k in r})
    rows.sort(key=lambda x: x["leadtime"])
    return rows


def main() -> None:
    """Plot overlaid lead-time curves."""

    args = _parse_args()
    root = Path(args.results_root).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    series_data: list[tuple[str, list[dict]]] = []
    for spec in args.series:
        label, rel = spec.split(",", 1)
        path = root / rel.strip()
        if not path.is_file():
            raise FileNotFoundError(path)
        series_data.append((label.strip(), _read(path)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)
    (ax_rmse, ax_crps), (ax_mae, ax_nll) = axes
    metrics = [
        (ax_rmse, "rmse", "RMSE (K)", "RMSE vs lead-time"),
        (ax_mae, "mae", "MAE (K)", "MAE vs lead-time"),
        (ax_crps, "crps", "CRPS (K)", "CRPS vs lead-time"),
    ]
    for ax, key, ylabel, title in metrics:
        for i, (label, rows) in enumerate(series_data):
            hours = [(r["leadtime"] + 1) * args.hours_per_step for r in rows]
            ax.plot(hours, [r[key] for r in rows], "o-", label=label,
                    color=f"C{i % 10}")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i, (label, rows) in enumerate(series_data):
        hours = [(r["leadtime"] + 1) * args.hours_per_step for r in rows]
        ax_nll.plot(hours, [r["nll"] for r in rows], "o-", label=f"NLL {label}",
                    color=f"C{i % 10}")
    ax_nll.set_ylabel("NLL")
    ax_nll.set_xlabel("Lead-time (hours)")
    ax_nll.set_title("NLL vs lead-time")
    ax_nll.grid(True, alpha=0.3)
    ax_nll.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
