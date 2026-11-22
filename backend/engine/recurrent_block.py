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
        context_proj: torch.Tensor | None = None,
        context_gate_c: torch.Tensor | None = None,
        step_proj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine hidden states by one step.

        hidden: [N, D] (gathered active positions) or [B, T, D]
        context: same leading shape as hidden
        step_emb: [D] or [N, D]

        ``context_proj`` and ``context_gate_c`` let a caller that reuses the
        same context across several steps cache the context projections.  This
        removes two small matrix multiplies per active token per step without
        changing the computation.
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
        c = self.context_proj(context) if context_proj is None else context_proj
        if context_gate_c is None:
            gate = torch.sigmoid(self.context_gate(torch.cat([h, c], dim=-1)))
        else:
            d = h.shape[-1]
            gate = torch.sigmoid(
                F.linear(h, self.context_gate.weight[:, :d]) + context_gate_c
            )
        x = h + gate * c
        x = x + (
            self.step_proj(step_emb) if step_proj is None else step_proj
        )
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
