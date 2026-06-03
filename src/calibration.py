"""Calibration diagnostics: ECE, reliability diagram, PIT histogram.

All inputs are Gaussian predictive parameters (mean, log_var) and ground truth.
Inputs should already be in denormalized physical units (Kelvin) for
plotted outputs to be meaningful, but the metrics themselves are scale-invariant.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
from torch import Tensor

_SQRT_2 = math.sqrt(2.0)


def _standard_normal_cdf(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z / _SQRT_2))


@torch.no_grad()
def compute_ece(mean: Tensor, log_var: Tensor, target: Tensor,
                n_bins: int = 10) -> float:
    """Expected Calibration Error over central-credible-interval coverage.

    For ``n_bins`` nominal coverage levels ``alpha in {1/n, 2/n, ..., 1}``,
    we form the symmetric central credible interval around the predicted
    mean and measure the empirical fraction of targets that fall inside it.
    ECE is the mean absolute gap between nominal and empirical coverage.

    Args:
        mean: Predicted means.
        log_var: Predicted log-variances.
        target: Ground-truth values.
        n_bins: Number of equally spaced nominal coverage levels in (0, 1].

    Returns:
        Scalar ECE in [0, 1].
    """
    sigma = (0.5 * log_var).exp()
    z = ((target - mean) / sigma).abs()
    nominal = torch.linspace(1.0 / n_bins, 1.0, n_bins, device=z.device)
    gaps = []
    for alpha in nominal:
        half_width = math.sqrt(2.0) * torch.erfinv(alpha)
        covered = (z <= half_width).float().mean()
        gaps.append((covered - alpha).abs().item())
    return float(np.mean(gaps))


@torch.no_grad()
def reliability_diagram(mean: Tensor, log_var: Tensor, target: Tensor,
                        save_path: Optional[str] = None) -> dict:
    """Reliability-diagram data and optional figure.

    Confidence levels: 10%, 20%, ..., 90%.

    Args:
        mean: Predicted means.
        log_var: Predicted log-variances.
        target: Ground-truth values.
        save_path: If provided, saves a 300 dpi PNG.

    Returns:
        ``{"confidence": [...], "coverage": [...], "ece": float}``.
    """
    sigma = (0.5 * log_var).exp()
    z = ((target - mean) / sigma).abs()
    confidences = np.linspace(0.1, 0.9, 9)
    coverages = []
    for alpha in confidences:
        half_width = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(alpha)).item())
        coverages.append((z <= half_width).float().mean().item())
    confidences = confidences.tolist()
    ece = float(np.mean(np.abs(np.array(confidences) - np.array(coverages))))

    if save_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="ideal")
        ax.plot(confidences, coverages, "o-", label="empirical")
        ax.set_xlabel("Nominal confidence")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"Reliability (ECE={ece:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    return {"confidence": confidences, "coverage": coverages, "ece": ece}


@torch.no_grad()
def pit_histogram(mean: Tensor, log_var: Tensor, target: Tensor,
                  n_bins: int = 20, save_path: Optional[str] = None) -> dict:
    """Probability Integral Transform histogram.

    ``PIT = Phi((target - mean) / sigma)``. Under perfect calibration,
    the PIT distribution is Uniform(0, 1).

    Args:
        mean: Predicted means.
        log_var: Predicted log-variances.
        target: Ground-truth values.
        n_bins: Number of histogram bins.
        save_path: If provided, saves a 300 dpi PNG.

    Returns:
        ``{"bin_edges": [...], "counts": [...]}``.
    """
    sigma = (0.5 * log_var).exp()
    pit = _standard_normal_cdf((target - mean) / sigma).cpu().numpy().reshape(-1)
    counts, bin_edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))

    if save_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(bin_edges[:-1], counts / counts.sum(),
               width=1.0 / n_bins, align="edge", edgecolor="black")
        ax.axhline(1.0 / n_bins, color="r", linestyle="--", label="uniform")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title("PIT histogram")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    return {"bin_edges": bin_edges.tolist(), "counts": counts.tolist()}
