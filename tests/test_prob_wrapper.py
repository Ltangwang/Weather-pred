"""Unit tests for src.model.ProbWrapper."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model import ProbWrapper


class _DummyBackbone(nn.Module):
    """Tiny backbone: (B, T_in, C, H, W) -> (B, T_out, C, H, W)."""

    def __init__(self, C: int = 1, T_out: int = 10):
        super().__init__()
        self.T_out = T_out
        self.conv = nn.Conv2d(C, C, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, C, H, W = x.shape
        last = x[:, -1]
        out = self.conv(last)
        return out.unsqueeze(1).repeat(1, self.T_out, 1, 1, 1)


def test_prob_wrapper_shape_and_clamp():
    B, T_in, C, H, W = 2, 10, 1, 32, 64
    clamp = (-3.0, 3.0)
    model = ProbWrapper(_DummyBackbone(C=C, T_out=10), out_channels=C,
                        log_var_clamp=clamp)
    x = torch.randn(B, T_in, C, H, W)
    y = model(x)
    assert y.shape == (B, 2, C, H, W)
    log_var = y[:, 1]
    assert (log_var >= clamp[0]).all() and (log_var <= clamp[1]).all()


def test_prob_wrapper_log_var_bias_init_default():
    """Second half of 1x1 head bias uses ``log_var_bias_init`` (default 0)."""
    C = 3
    model = ProbWrapper(_DummyBackbone(C=C, T_out=10), out_channels=C)
    b = model.prob_head.bias
    assert b is not None
    assert torch.allclose(b[:C], torch.zeros(C))
    assert torch.allclose(b[C:], torch.zeros(C))


def test_prob_wrapper_grad_flows_to_backbone():
    model = ProbWrapper(_DummyBackbone(C=1, T_out=10), out_channels=1)
    x = torch.randn(2, 10, 1, 32, 64, requires_grad=False)
    target = torch.randn(2, 1, 32, 64)
    y = model(x)
    loss = (y[:, 0] - target).pow(2).mean() + y[:, 1].mean()
    loss.backward()
    grads = [p.grad for p in model.backbone.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
