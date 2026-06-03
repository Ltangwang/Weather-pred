"""Smoke tests for src.utils."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

from src.utils import (get_logger, load_checkpoint, save_checkpoint, set_seed)


def test_set_seed_repeatable():
    set_seed(42)
    a = (torch.randn(4).tolist(), np.random.rand(4).tolist(), random.random())
    set_seed(42)
    b = (torch.randn(4).tolist(), np.random.rand(4).tolist(), random.random())
    assert a == b


def test_get_logger(tmp_path):
    log_file = tmp_path / "x.log"
    logger = get_logger("test_logger", str(log_file))
    assert isinstance(logger, logging.Logger)
    logger.info("hello")
    assert log_file.exists()
    assert "hello" in log_file.read_text()


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = str(tmp_path / "last.pth")
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": 7,
        "best_crps": 0.123,
    }
    save_checkpoint(state, path, is_best=True)
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(tmp_path, "best.pth"))

    new_model = torch.nn.Linear(3, 2)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    meta = load_checkpoint(path, new_model, new_opt)
    assert meta["epoch"] == 7
    assert abs(meta["best_crps"] - 0.123) < 1e-9
