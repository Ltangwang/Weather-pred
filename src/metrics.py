"""Evaluation metrics. All inputs MUST be denormalized to physical units (Kelvin)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_PI = math.sqrt(math.pi)
_LOG_2PI = math.log(2.0 * math.pi)


@torch.no_grad()
def rmse(pred: Tensor, target: Tensor) -> float:
    """Root mean squared error in Kelvin."""
    return torch.sqrt(((pred - target) ** 2).mean()).item()


@torch.no_grad()
def mae(pred: Tensor, target: Tensor) -> float:
    """Mean absolute error in Kelvin."""
    return (pred - target).abs().mean().item()


@torch.no_grad()
def crps_gaussian(mean: Tensor, log_var: Tensor, target: Tensor) -> float:
    """Mean Gaussian CRPS in Kelvin (denormalized inputs)."""
    sigma = (0.5 * log_var).exp()
    z = (target - mean) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / _SQRT_2))
    phi = torch.exp(-0.5 * z * z) / _SQRT_2PI
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / _SQRT_PI)
    return crps.mean().item()


@torch.no_grad()
def nll_gaussian(mean: Tensor, log_var: Tensor, target: Tensor) -> float:
    """Full Gaussian NLL (with log(2pi) constant), denormalized inputs.

    ``NLL = 0.5 * (log(2*pi) + log_var + (y-mean)^2 / exp(log_var))``.
    """
    val = 0.5 * (_LOG_2PI + log_var + (target - mean).pow(2) / log_var.exp())
    return val.mean().item()


def _batch_crps_sum(mean: Tensor, log_var: Tensor, target: Tensor) -> tuple[float, int]:
    sigma = (0.5 * log_var).exp()
    z = (target - mean) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / _SQRT_2))
    phi = torch.exp(-0.5 * z * z) / _SQRT_2PI
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / _SQRT_PI)
    return crps.sum().item(), crps.numel()


def _batch_nll_sum(mean: Tensor, log_var: Tensor, target: Tensor) -> tuple[float, int]:
    val = 0.5 * (_LOG_2PI + log_var + (target - mean).pow(2) / log_var.exp())
    return val.sum().item(), val.numel()


@dataclass
class MetricAccumulator:
    """Online sum-based accumulator for denormalized Gaussian metrics.

    Avoids storing the full validation/test tensors in RAM when evaluating
    on the complete WeatherBench splits.
    """

    n: int = 0
    mae_sum: float = 0.0
    mse_sum: float = 0.0
    crps_sum: float = 0.0
    nll_sum: float = 0.0
    ece_covered: list[float] = field(default_factory=list)
    ece_bins: int = 10

    def __post_init__(self) -> None:
        if not self.ece_covered:
            self.ece_covered = [0.0] * self.ece_bins

    @torch.no_grad()
    def update(self, mean: Tensor, log_var: Tensor, target: Tensor) -> None:
        """Incorporate one batch of equal-shaped tensors."""
        err = (mean - target).reshape(-1)
        n = err.numel()
        if n == 0:
            return
        self.n += n
        self.mae_sum += err.abs().sum().item()
        self.mse_sum += err.pow(2).sum().item()
        crps_s, _ = _batch_crps_sum(mean, log_var, target)
        nll_s, _ = _batch_nll_sum(mean, log_var, target)
        self.crps_sum += crps_s
        self.nll_sum += nll_s

        sigma = (0.5 * log_var).exp().reshape(-1)
        z = err.abs() / sigma
        nominal = torch.linspace(
            1.0 / self.ece_bins, 1.0, self.ece_bins, device=z.device)
        for i, alpha in enumerate(nominal):
            half_width = math.sqrt(2.0) * float(torch.erfinv(alpha).item())
            self.ece_covered[i] += float((z <= half_width).sum().item())

    def finalize(self) -> dict[str, float]:
        """Return headline scalar metrics."""
        if self.n == 0:
            raise ValueError("MetricAccumulator received no samples.")
        nominal = np.linspace(1.0 / self.ece_bins, 1.0, self.ece_bins)
        gaps = []
        for i, alpha in enumerate(nominal):
            covered = self.ece_covered[i] / self.n
            gaps.append(abs(covered - alpha))
        return {
            "rmse": math.sqrt(self.mse_sum / self.n),
            "mae": self.mae_sum / self.n,
            "crps": self.crps_sum / self.n,
            "nll": self.nll_sum / self.n,
            "ece": float(np.mean(gaps)),
        }
