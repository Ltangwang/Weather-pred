"""Shared utilities: seeding, logging, checkpoint helpers."""
from __future__ import annotations

import logging
import os
import random
import shutil
import sys
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix random seeds for torch / numpy / random.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Return a logger that writes to both stdout and (optionally) a file.

    Args:
        name: Logger name.
        log_file: Optional path to log file. Parent dirs are created.

    Returns:
        Configured `logging.Logger` (idempotent across calls with same name).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def save_checkpoint(state: dict, path: str, is_best: bool = False) -> None:
    """Save a checkpoint dict. If `is_best`, also copy to `best.pth` in the same dir.

    Args:
        state: Dict containing at least `model_state_dict`; arbitrary meta allowed.
        path: Destination path (e.g. `results/run/last.pth`).
        is_best: If True, copies to sibling `best.pth`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path) or ".", "best.pth")
        shutil.copyfile(path, best_path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """Load a checkpoint into model (and optionally optimizer).

    Args:
        path: Checkpoint path.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.

    Returns:
        Meta dict (epoch, best metric, etc.) excluding the state dicts.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    meta = {k: v for k, v in ckpt.items()
            if k not in ("model_state_dict", "optimizer_state_dict")}
    return meta
