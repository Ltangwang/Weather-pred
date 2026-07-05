"""ProbWrapper and lightweight ConvLSTM backbone."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class SimpleConvLSTM(nn.Module):
    """ConvLSTM backbone: (B, T_in, C, H, W) -> (B, T_out, C, H, W)."""

    def __init__(self, in_shape: Tuple[int, int, int, int],
                 hidden_channels: int = 64, num_layers: int = 3,
                 kernel_size: int = 3, out_len: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        T_in, C, H, W = in_shape
        self.in_len = T_in
        self.in_channels = C
        self.out_len = out_len if out_len is not None else T_in
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        cells = []
        for i in range(num_layers):
            cells.append(ConvLSTMCell(
                in_channels=C if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
            ))
        self.cells = nn.ModuleList(cells)
        self.head = nn.Conv2d(hidden_channels, C, kernel_size=1)

    def _init_state(self, x: torch.Tensor):
        B, _, _, H, W = x.shape
        states = []
        for _ in range(self.num_layers):
            h = x.new_zeros(B, self.hidden_channels, H, W)
            c = x.new_zeros(B, self.hidden_channels, H, W)
            states.append((h, c))
        return states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, C, H, W)
        states = self._init_state(x)

        # Encoder: consume input frames.
        for t in range(x.shape[1]):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                h, c = cell(inp, states[i])
                states[i] = (h, c)
                inp = self.dropout(h) if i < self.num_layers - 1 else h

        # Decoder: autoregressive rollout starting from the last predicted
        # frame (initialized as the last input frame).
        outs = []
        last_frame = x[:, -1]
        for _ in range(self.out_len):
            inp = last_frame
            for i, cell in enumerate(self.cells):
                h, c = cell(inp, states[i])
                states[i] = (h, c)
                inp = self.dropout(h) if i < self.num_layers - 1 else h
            y_t = self.head(inp)
            outs.append(y_t)
            last_frame = y_t

        return torch.stack(outs, dim=1)


class ProbWrapper(nn.Module):
    """Gaussian head on a deterministic backbone. Output: (B, 2, C, H, W)."""

    def __init__(self, backbone: nn.Module, out_channels: int,
                 log_var_clamp: Tuple[float, float] = (-7.0, 3.0),
                 log_var_bias_init: float = 0.0,
                 multi_frame: bool = False):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        self.log_var_clamp = log_var_clamp
        self.multi_frame = multi_frame
        self.prob_head = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1)
        if self.prob_head.bias is not None:
            with torch.no_grad():
                self.prob_head.bias.zero_()
                self.prob_head.bias[out_channels:].fill_(log_var_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if self.multi_frame:
            if feat.dim() != 5:
                raise ValueError("multi_frame=True requires a 5-D backbone "
                                 f"output (B,T,C,H,W); got dim={feat.dim()}.")
            B, T, C, H, W = feat.shape
            h = self.prob_head(feat.reshape(B * T, C, H, W))
            h = h.reshape(B, T, 2 * self.out_channels, H, W)
            mean, log_var = h.chunk(2, dim=2)
            log_var = log_var.clamp(min=self.log_var_clamp[0],
                                    max=self.log_var_clamp[1])
            # (B, 2, T, C, H, W) — index 0 = mean, 1 = log_var.
            return torch.stack([mean, log_var], dim=1)

        if feat.dim() == 5:
            last = feat[:, -1]
        elif feat.dim() == 4:
            last = feat
        else:
            raise ValueError(f"Unexpected backbone output dim={feat.dim()}; "
                             f"expected 4 or 5.")
        h = self.prob_head(last)
        mean, log_var = h.chunk(2, dim=1)
        log_var = log_var.clamp(min=self.log_var_clamp[0],
                                max=self.log_var_clamp[1])
        return torch.stack([mean, log_var], dim=1)
