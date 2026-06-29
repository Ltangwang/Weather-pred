"""Loss functions for probabilistic spatiotemporal forecasting.

All probabilistic losses operate on (mean, log_var, target) tensors of identical shape.
`mse_loss` is only intended for deterministic baselines.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_PI = math.sqrt(math.pi)


def gaussian_nll(mean: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """Gaussian Negative Log-Likelihood (constant terms dropped).

    Formula: ``0.5 * (log_var + (target - mean)^2 / exp(log_var))``.

    Args:
        mean: Predicted mean.
        log_var: Predicted log-variance (same shape as `mean`).
        target: Ground truth (same shape as `mean`).

    Returns:
        Scalar mean loss.
    """
    return 0.5 * (log_var + (target - mean).pow(2) / log_var.exp()).mean()


def beta_gaussian_nll(mean: Tensor, log_var: Tensor, target: Tensor,
                      beta: float = 0.5) -> Tensor:
    """β-NLL reweighting from Seitzer et al. (2022).

    Per-element NLL is multiplied by ``σ^(2β)`` with ``σ = exp(0.5 * log_var)``
    detached from the graph (stop-gradient on the variance scale). ``β=0.5`` is
    the recommended default in the original paper.

    Args:
        mean: Predicted mean.
        log_var: Predicted log-variance.
        target: Ground truth.
        beta: Reweighting exponent (typically 0.5).

    Returns:
        Scalar mean β-NLL loss.
    """
    per_elem = 0.5 * (log_var + (target - mean).pow(2) / log_var.exp())
    sigma = (0.5 * log_var).exp()
    weight = sigma.pow(2.0 * beta).detach()
    return (weight * per_elem).mean()


def gaussian_crps(mean: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """Closed-form CRPS for a Gaussian predictive distribution.

    ``sigma = exp(0.5 * log_var)``,
    ``z = (target - mean) / sigma``,
    ``CRPS = sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))``.

    Args:
        mean: Predicted mean.
        log_var: Predicted log-variance.
        target: Ground truth.

    Returns:
        Scalar mean CRPS.
    """
    sigma = (0.5 * log_var).exp()
    z = (target - mean) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / _SQRT_2))
    phi = torch.exp(-0.5 * z * z) / _SQRT_2PI
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / _SQRT_PI)
    return crps.mean()


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Standard mean squared error. Only for deterministic baselines."""
    return F.mse_loss(pred, target)
