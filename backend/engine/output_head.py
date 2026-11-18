"""Output projection and adaptive state aggregation."""
from __future__ import annotations

import torch
import torch.nn as nn

from backend.config import ModelConfig


class AdaptiveOutputAggregator(nn.Module):
    """Weighted aggregation of per-step hidden states.

    For soft ACT the weights are the halting probability masses; for hard
    inference the controller supplies a single final state with weight 1.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        states: list[torch.Tensor],
        weights: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not states:
            raise ValueError("at least one state is required")
        if weights is None:
            return self.norm(states[-1])
        if len(states) != len(weights):
            raise ValueError("states and weights must have the same length")
        out = torch.zeros_like(states[0])
        for state, weight in zip(states, weights):
            w = weight
            while w.ndim < state.ndim:
                w = w.unsqueeze(-1)
            out = out + w * state
        return self.norm(out)


class OutputProjection(nn.Module):
    def __init__(self, cfg: ModelConfig, token_embedding: nn.Embedding | None = None):
        super().__init__()
        self.cfg = cfg
        if cfg.tie_embeddings and token_embedding is not None:
            self.proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
            self.proj.weight = token_embedding.weight
        else:
            self.proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
