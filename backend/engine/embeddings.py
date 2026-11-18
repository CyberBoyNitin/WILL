"""Token + positional embeddings with normalisation and dropout."""
from __future__ import annotations

import torch
import torch.nn as nn

from backend.config import ModelConfig


class InputEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(
            cfg.vocab_size, cfg.hidden_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.dropout(self.norm(x))
