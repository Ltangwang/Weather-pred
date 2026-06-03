"""Unit tests for src.metrics."""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from src.metrics import crps_gaussian, mae, nll_gaussian, rmse


def test_rmse_zero_when_pred_equals_target():
    x = torch.randn(8, 1, 4, 4)
    assert rmse(x, x) < 1e-7


def test_rmse_matches_sklearn():
    torch.manual_seed(0)
    a = torch.randn(64)
    b = torch.randn(64)
    expected = math.sqrt(mean_squared_error(a.numpy(), b.numpy()))
    assert abs(rmse(a, b) - expected) < 1e-5


def test_mae_zero_when_pred_equals_target():
    x = torch.randn(4, 1, 8, 8)
    assert mae(x, x) < 1e-7


def test_crps_gaussian_lower_when_perfect_mean():
    torch.manual_seed(0)
    target = torch.randn(4, 1, 8, 8)
    log_var = torch.full_like(target, -2.0)
    assert crps_gaussian(target.clone(), log_var, target) < \
        crps_gaussian(target + 1.0, log_var, target)


def test_nll_gaussian_returns_float():
    target = torch.zeros(4, 1, 4, 4)
    mean = torch.zeros_like(target)
    log_var = torch.zeros_like(target)
    val = nll_gaussian(mean, log_var, target)
    assert isinstance(val, float)
    expected = 0.5 * math.log(2 * math.pi)
    assert abs(val - expected) < 1e-5
