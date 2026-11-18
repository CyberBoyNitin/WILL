"""Learned token-level halting gate.

The gate outputs a halting probability.  This is a learned decision, not a
calibrated answer-confidence estimate; the architecture never treats it as
one.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from backend.config import ModelConfig


class HaltingGate(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim
        self.norm = nn.LayerNorm(d)
        self.step_proj = nn.Linear(d, d)
        self.context_proj = nn.Linear(d, d)
        self.net = nn.Sequential(
            nn.Linear(3 * d, max(d // 2, 8)),
            nn.GELU(),
            nn.Linear(max(d // 2, 8), 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # A negative initial bias avoids halting everything at step one.
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.constant_(last.bias, -1.5)
        nn.init.normal_(last.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        step_emb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if step_emb.ndim == 1:
            step_emb = (
                step_emb.view(1, 1, -1) if hidden.ndim == 3 else step_emb
            )
        c = (
            self.context_proj(context)
            if context is not None
            else torch.zeros_like(hidden)
        )
        x = torch.cat([self.norm(hidden), self.step_proj(step_emb).expand_as(hidden), c], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)
