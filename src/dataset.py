"""WeatherBench dataset wrapper around OpenSTL.

Provides a thin adapter over `openstl.datasets.dataloader_weather.WeatherBenchDataset`
that also persists denormalization statistics for downstream metric computation.
A `SyntheticWeatherBench` is included for unit testing when real data is absent.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_DEFAULT_IN_LEN = 10
_DEFAULT_OUT_LEN = 10
_DEFAULT_RES: Tuple[int, int] = (32, 64)


def denormalize(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Convert a z-score tensor back to physical units (Kelvin)."""
    return tensor * std + mean


def save_norm_stats(mean: float, std: float, path: str) -> None:
    """Persist ``{"mean": ..., "std": ...}`` to a JSON file."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"mean": float(mean), "std": float(std)}, f, indent=2)


def load_norm_stats(path: str) -> Tuple[float, float]:
    """Load ``(mean, std)`` from JSON."""
    with open(path) as f:
        d = json.load(f)
    return float(d["mean"]), float(d["std"])


class WeatherBenchDataset(Dataset):
    """Thin wrapper over OpenSTL's `WeatherBenchDataset` for t2m at 5.625deg.

    Args:
        data_root: Path containing the OpenSTL `weather_5_625deg/` (or
            `weather/`) directory.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        in_len: Number of input frames (default 10).
        out_len: Number of output frames to predict (default 10).
        variable: Variable short-name; default ``"t2m"``.
        data_split: Spatial resolution code; default ``"5_625"``.
        norm_stats: Optional ``(mean, std)`` override; if None and split is
            not "train", a previously saved JSON must be supplied via
            `norm_stats_path` in `train`-then-eval workflows.
        norm_stats_path: Optional JSON path to load/save mean/std.

    Returns per item:
        ``(x, y)`` where ``x.shape == (in_len, C, H, W)`` and
        ``y.shape == (out_len, C, H, W)``; both normalized float32.
    """

    SPLIT_TIMES = {
        "train": ["1979", "2015"],
        "val":   ["2016", "2016"],
        "test":  ["2017", "2018"],
    }

    def __init__(self, data_root: str, split: str = "train",
                 in_len: int = _DEFAULT_IN_LEN,
                 out_len: int = _DEFAULT_OUT_LEN,
                 variable: str = "t2m",
                 data_split: str = "5_625",
                 norm_stats: Optional[Tuple[float, float]] = None,
                 norm_stats_path: Optional[str] = None):
        if split not in self.SPLIT_TIMES:
            raise ValueError(f"split must be one of {list(self.SPLIT_TIMES)}")
        from openstl.datasets.dataloader_weather import WeatherBenchDataset as _OSDS

        # ASSUMPTION: OpenSTL data layout is `<data_root>/weather_<split>deg/<var>/`.
        for suffix in (f"weather_{data_split}deg", "weather", f"{data_split}deg"):
            cand = os.path.join(data_root, suffix)
            if os.path.exists(cand):
                weather_root = cand
                break
        else:
            raise FileNotFoundError(
                f"Could not find weather data under {data_root}")

        idx_in = list(range(-in_len + 1, 1))
        idx_out = list(range(1, out_len + 1))

        mean_std_kw = {}
        if norm_stats is not None:
            m, s = norm_stats
            mean_std_kw = {"mean": np.array(m).reshape(1, 1, 1, 1),
                           "std":  np.array(s).reshape(1, 1, 1, 1)}

        self._ds = _OSDS(
            data_root=weather_root, data_name=variable, data_split=data_split,
            training_time=self.SPLIT_TIMES[split],
            idx_in=idx_in, idx_out=idx_out, step=1, levels=["50"],
            use_augment=False, **mean_std_kw,
        )
        self.mean = float(np.array(self._ds.mean).reshape(-1)[0])
        self.std = float(np.array(self._ds.std).reshape(-1)[0])
        self.in_len = in_len
        self.out_len = out_len
        if norm_stats_path is not None and split == "train":
            save_norm_stats(self.mean, self.std, norm_stats_path)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int):
        x, y = self._ds[index]
        return x.float(), y.float()

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert a normalized tensor back to Kelvin."""
        return denormalize(tensor, self.mean, self.std)


class SyntheticWeatherBench(Dataset):
    """Synthetic WeatherBench-shaped dataset for testing without real data.

    Produces deterministic random tensors of shape ``(in_len, 1, H, W)`` and
    ``(out_len, 1, H, W)`` with configurable mean/std.
    """

    def __init__(self, length: int = 32, in_len: int = _DEFAULT_IN_LEN,
                 out_len: int = _DEFAULT_OUT_LEN,
                 resolution: Tuple[int, int] = _DEFAULT_RES,
                 mean: float = 280.0, std: float = 15.0, seed: int = 0):
        self.length = length
        self.in_len = in_len
        self.out_len = out_len
        self.H, self.W = resolution
        self.mean = mean
        self.std = std
        self._rng = np.random.default_rng(seed)
        # Pre-generate to ensure determinism per index.
        self._buf = self._rng.standard_normal(
            (length + in_len + out_len, 1, self.H, self.W)).astype(np.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        x = torch.from_numpy(self._buf[index:index + self.in_len])
        y = torch.from_numpy(
            self._buf[index + self.in_len:index + self.in_len + self.out_len])
        return x, y

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return denormalize(tensor, self.mean, self.std)
