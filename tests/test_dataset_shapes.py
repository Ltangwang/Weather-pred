"""Shape tests for the WeatherBench dataset (real + synthetic)."""
from __future__ import annotations

import os

import pytest
import torch

from src.dataset import (SyntheticWeatherBench, WeatherBenchDataset,
                         denormalize, load_norm_stats, save_norm_stats)


_DATA_ROOT = "/root/autodl-tmp/OpenSTL/data"


def _has_real_data() -> bool:
    for s in ("weather_5_625deg", "weather", "5_625deg"):
        p = os.path.join(_DATA_ROOT, s, "2m_temperature")
        if os.path.isdir(p) and any(f.endswith(".nc") for f in os.listdir(p)):
            return True
    return False


def test_synthetic_shapes():
    ds = SyntheticWeatherBench(length=4)
    x, y = ds[0]
    assert x.shape == (10, 1, 32, 64)
    assert y.shape == (10, 1, 32, 64)
    assert x.dtype == torch.float32
    assert torch.isnan(x).sum() == 0
    assert torch.isnan(y).sum() == 0


def test_denormalize_roundtrip():
    x = torch.randn(4, 1, 32, 64)
    mean, std = 280.0, 15.0
    y = denormalize(x, mean, std)
    z = (y - mean) / std
    assert torch.allclose(x, z, atol=1e-5)


def test_norm_stats_json_roundtrip(tmp_path):
    p = str(tmp_path / "stats.json")
    save_norm_stats(280.123, 15.456, p)
    m, s = load_norm_stats(p)
    assert abs(m - 280.123) < 1e-6 and abs(s - 15.456) < 1e-6


@pytest.mark.skipif(not _has_real_data(),
                    reason="WeatherBench .nc files not available locally")
def test_real_weatherbench_shapes():
    ds = WeatherBenchDataset(_DATA_ROOT, split="val", in_len=10, out_len=10)
    x, y = ds[0]
    assert x.shape == (10, 1, 32, 64)
    assert y.shape == (10, 1, 32, 64)
    assert x.dtype == torch.float32
    assert torch.isnan(x).sum() == 0
