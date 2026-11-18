"""Shared recurrent reasoning block.

One instance of this module is called repeatedly.  The parameters are
therefore shared across all refinement steps; only the learned step embedding
changes.  The block is token-wise so that the adaptive controller can gather a
subset of active positions and run only those.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import ModelConfig


class RecurrentReasoningBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim
        f = cfg.ff_mult * d
        self.cfg = cfg
        self.in_norm = nn.LayerNorm(d)
        self.context_proj = nn.Linear(d, d)
        self.context_gate = nn.Linear(2 * d, d)
        self.step_embedding = nn.Embedding(cfg.max_steps, d)
        self.step_proj = nn.Linear(d, d)
        self.ff_norm = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, 2 * f)
        self.ff_out = nn.Linear(f, d)
        self.dropout = nn.Dropout(cfg.dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.step_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.context_gate.weight, mean=0.0, std=0.02)
        # Start with a weak context gate so the residual path dominates.
        nn.init.constant_(self.context_gate.bias, -1.0)

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        step_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Refine hidden states by one step.

        hidden: [N, D] (gathered active positions) or [B, T, D]
        context: same leading shape as hidden
        step_emb: [D] or [N, D]
        """
        if hidden.shape != context.shape:
            raise ValueError("hidden and context must have identical shapes")
        if step_emb.ndim == 1:
            step_emb = step_emb.view(1, 1, -1) if hidden.ndim == 3 else step_emb
        elif step_emb.ndim == 2:
            if hidden.ndim == 3:
                step_emb = step_emb.view(1, step_emb.shape[0], -1)
            elif step_emb.shape[0] != hidden.shape[0]:
                raise ValueError("per-token step embeddings must match batch size")

        h = self.in_norm(hidden)
        c = self.context_proj(context)
        gate = torch.sigmoid(self.context_gate(torch.cat([h, c], dim=-1)))
        x = h + gate * c
        x = x + self.step_proj(step_emb)
        u = self.ff_norm(x)
        value, ffn_gate = self.ff_in(u).chunk(2, dim=-1)
        update = self.ff_out(F.gelu(ffn_gate) * value)
        return hidden + self.dropout(update)

    def macs_per_token(self) -> int:
        total = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                total += module.in_features * module.out_features
        return int(total)
