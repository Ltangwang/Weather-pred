"""Unit tests for src.calibration."""
from __future__ import annotations

import os

import numpy as np
import torch

from src.calibration import compute_ece, pit_histogram, reliability_diagram


def test_ece_near_zero_when_perfectly_calibrated():
    torch.manual_seed(0)
    n = 20000
    mean = torch.zeros(n)
    log_var = torch.zeros(n)
    sigma = (0.5 * log_var).exp()
    target = mean + sigma * torch.randn(n)
    ece = compute_ece(mean, log_var, target, n_bins=10)
    assert ece < 0.05


def test_ece_large_when_overconfident():
    torch.manual_seed(0)
    n = 5000
    mean = torch.zeros(n)
    log_var = torch.full((n,), -6.0)
    target = torch.randn(n) * 2.0
    ece = compute_ece(mean, log_var, target, n_bins=10)
    assert ece > 0.2


def test_reliability_diagram_keys_and_save(tmp_path):
    torch.manual_seed(0)
    n = 5000
    mean = torch.zeros(n)
    log_var = torch.zeros(n)
    target = torch.randn(n)
    out_path = str(tmp_path / "rel.png")
    out = reliability_diagram(mean, log_var, target, save_path=out_path)
    assert set(out) == {"confidence", "coverage", "ece"}
    assert len(out["confidence"]) == len(out["coverage"]) == 9
    assert os.path.exists(out_path)


def test_pit_histogram_uniform_when_calibrated(tmp_path):
    torch.manual_seed(0)
    n = 20000
    mean = torch.zeros(n)
    log_var = torch.zeros(n)
    target = torch.randn(n)
    out = pit_histogram(mean, log_var, target, n_bins=20,
                        save_path=str(tmp_path / "pit.png"))
    counts = np.array(out["counts"], dtype=float)
    freq = counts / counts.sum()
    assert np.max(np.abs(freq - 1.0 / 20)) < 0.02
