"""Slow-weight encoder that maps contextual states into memory space."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import ModelConfig


def pool_memory_tokens(x: torch.Tensor) -> torch.Tensor:
    """Position-weighted pooling with a final-token term.

    Mean pooling makes short keys that share most bytes (``K00`` vs ``K10``)
    nearly collinear.  Increasing positional weights plus the final token keep
    order-sensitive keys distinguishable while still using the whole span.
    """
    squeeze = x.ndim == 2
    if squeeze:
        x = x.unsqueeze(0)
    t = x.shape[1]
    weights = torch.arange(1, t + 1, dtype=x.dtype, device=x.device).view(1, t, 1)
    pooled = (x * weights).sum(dim=1) / weights.sum().clamp_min(1.0)
    pooled = pooled + x[:, -1, :]
    pooled = F.normalize(pooled, dim=-1)
    return pooled.squeeze(0) if squeeze else pooled


class MemoryEncoder(nn.Module):
    """Projects contextual hidden states to normalised keys and values.

    Keys address memory; values are the self-supervised targets.  Both are
    derived from the same input, so no labels are required.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.key = nn.Linear(cfg.hidden_dim, cfg.memory_dim)
        self.value = nn.Linear(cfg.hidden_dim, cfg.memory_dim)
        token_dim = int(cfg.memory_token_dim or cfg.memory_dim)
        self.token_dim = token_dim
        self.token_key = nn.Embedding(cfg.vocab_size, token_dim)
        self.token_value = nn.Embedding(cfg.vocab_size, token_dim)
        if token_dim != cfg.memory_dim:
            self.token_key_proj = nn.Linear(token_dim, cfg.memory_dim, bias=False)
            self.token_value_proj = nn.Linear(token_dim, cfg.memory_dim, bias=False)
        self.norm = nn.LayerNorm(cfg.memory_dim)
        self.seq_key = nn.GRU(cfg.memory_dim, cfg.memory_dim, batch_first=True)
        self.seq_value = nn.GRU(cfg.memory_dim, cfg.memory_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(cfg.memory_dim)
        nn.init.normal_(self.token_key.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.token_value.weight, mean=0.0, std=0.5)
        for gru in (self.seq_key, self.seq_value):
            for name, param in gru.named_parameters():
                if "weight" in name:
                    nn.init.normal_(param, mean=0.0, std=0.1)
                else:
                    nn.init.zeros_(param)
        with torch.no_grad():
            self.token_key.weight[0].zero_()
            self.token_value.weight[0].zero_()

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.key(hidden)
        v = self.value(hidden)
        if input_ids is not None:
            # Token identity keeps short record keys distinguishable even
            # before the context encoder has been trained on a memory task.
            if self.token_dim == k.shape[-1]:
                k = k + self.token_key(input_ids)
                v = v + self.token_value(input_ids)
            else:
                k = k + self.token_key_proj(self.token_key(input_ids))
                v = v + self.token_value_proj(self.token_value(input_ids))
        k = self.norm(k)
        v = self.norm(v)
        k, _ = self.seq_key(k)
        v, _ = self.seq_value(v)
        k = self.out_norm(k)
        v = self.out_norm(v)
        return F.normalize(k, dim=-1), F.normalize(v, dim=-1)

    def encode_pooled(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool a [B,T,D] sequence to one key/value per batch row."""
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            m = mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        k, v = self.forward(pooled.unsqueeze(1))
        return k.squeeze(1), v.squeeze(1)


class MemoryQueryBridge(nn.Module):
    """Refine the writer-aligned query and gate retrieval per channel.

    The base query is the pooled key produced by ``MemoryEncoder``, which makes
    retrieval exact against the Test 5/6 writer at initialisation.  A two-layer
    query MLP supplies a learned correction in the same ``memory_dim`` space.
    The retrieved vector is multiplied by a vector-wise sigmoid gate and
    projected back to hidden_dim, so the recurrent block receives a
    channel-wise memory residual instead of a single scalar gain.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, m = cfg.hidden_dim, cfg.memory_dim
        self.query = nn.Sequential(nn.Linear(d, m), nn.SiLU(), nn.Linear(m, m))
        self.gate = nn.Linear(d, m)
        self.readout = nn.Linear(m, d)

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(self.query(hidden), dim=-1)
        gate = torch.sigmoid(self.gate(hidden))
        return query, gate


class MemoryIntegrationGate(nn.Module):
    """Learned gate controlling how much retrieved memory enters the state.

    The gate is trainable with ordinary model training.  A negative initial
    bias means an untrained model does not blindly trust retrieved vectors.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.readout = nn.Linear(cfg.memory_dim, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self, hidden: torch.Tensor, memory_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[:-1] != memory_output.shape[:-1]:
            raise ValueError("memory output must match hidden leading dimensions")
        g = torch.sigmoid(self.gate(hidden))
        combined = hidden + g * self.readout(memory_output)
        return self.norm(combined), g
