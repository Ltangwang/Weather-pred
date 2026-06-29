"""Unit tests for src.losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses import beta_gaussian_nll, gaussian_crps, gaussian_nll, mse_loss


def test_gaussian_nll_scalar_and_grad():
    torch.manual_seed(0)
    mean = torch.randn(2, 1, 4, 4, requires_grad=True)
    log_var = torch.zeros(2, 1, 4, 4, requires_grad=True)
    target = torch.randn(2, 1, 4, 4)
    loss = gaussian_nll(mean, log_var, target)
    assert loss.dim() == 0
    loss.backward()
    assert mean.grad is not None and log_var.grad is not None


def test_gaussian_crps_lower_when_pred_matches_target():
    torch.manual_seed(0)
    target = torch.randn(4, 1, 8, 8)
    log_var = torch.full_like(target, -2.0)
    perfect = gaussian_crps(target.clone(), log_var, target)
    bad = gaussian_crps(target + 1.0, log_var, target)
    assert perfect.item() < bad.item()


def test_mse_matches_functional():
    torch.manual_seed(0)
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    assert torch.allclose(mse_loss(a, b), F.mse_loss(a, b))


def test_beta_nll_reweights_with_stop_gradient():
    torch.manual_seed(0)
    mean = torch.randn(2, 1, 4, 4, requires_grad=True)
    log_var = torch.full((2, 1, 4, 4), -1.0, requires_grad=True)
    target = torch.randn(2, 1, 4, 4)
    loss = beta_gaussian_nll(mean, log_var, target, beta=0.5)
    assert loss.dim() == 0
    loss.backward()
    assert mean.grad is not None and log_var.grad is not None
    plain = gaussian_nll(mean.detach(), log_var.detach(), target)
    assert not torch.allclose(loss.detach(), plain.detach())
