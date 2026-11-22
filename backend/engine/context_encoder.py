"""Small causal transformer used to build a contextual representation once.

The recurrent reasoning block is *not* applied here.  Context attention is
computed once per forward pass and reused by every recurrent iteration, which
is both cheaper and keeps the adaptive controller's token-wise gather logic
simple.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import ModelConfig
from backend.engine.kv_cache import KVCache


def build_causal_mask(
    attention_mask: torch.Tensor | None,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an additive attention mask [B,1,T,T] (or [1,1,T,T])."""
    neg = torch.finfo(dtype).min
    causal = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask = mask.masked_fill(causal, neg)
    if attention_mask is None:
        return mask.unsqueeze(0).unsqueeze(0)
    key_valid = attention_mask.to(dtype=dtype).unsqueeze(1).unsqueeze(2)  # B,1,1,T
    mask = mask.unsqueeze(0).unsqueeze(0) + (1.0 - key_valid) * neg
    return mask


def build_incremental_mask(
    attention_mask: torch.Tensor | None,
    query_len: int,
    key_len: int,
    start_pos: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive mask for prefill or incremental decoding.

    ``key_len`` includes the cached prefix.  The causal condition is expressed
    in absolute index space, so a query at absolute position ``p`` may attend
    to keys ``0..p``.  Padding keys are removed by ``attention_mask``.
    """
    neg = torch.finfo(dtype).min
    query_positions = torch.arange(
        start_pos, start_pos + query_len, device=device
    ).unsqueeze(1)
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    allowed = key_positions <= query_positions
    mask = torch.zeros(
        query_len, key_len, dtype=dtype, device=device
    ).masked_fill(~allowed, neg)
    if attention_mask is None:
        return mask.unsqueeze(0).unsqueeze(0)
    key_valid = attention_mask[:, :key_len].to(dtype=dtype).unsqueeze(1).unsqueeze(2)
    return mask.unsqueeze(0).unsqueeze(0) + (1.0 - key_valid) * neg


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.resid_dropout(self.out(y))

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cache=None,
    ) -> torch.Tensor:
        """Attention over a cached prefix plus the current query tokens."""
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            cache.append(k, v)
            k = cache.key
            v = cache.value
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.resid_dropout(self.out(y))


class CausalTransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        hidden = cfg.context_ff_mult * cfg.hidden_dim
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cache=None,
    ) -> torch.Tensor:
        x = x + self.attn.forward_cached(self.norm1(x), attn_mask=attn_mask, cache=cache)
        x = x + self.ff(self.norm2(x))
        return x


class CausalContextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [CausalTransformerBlock(cfg) for _ in range(cfg.context_layers)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, d = x.shape
        mask = build_causal_mask(attention_mask, t, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        return self.norm(x)

    def forward_cached(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        start_pos: int | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Run prefill or decode while retaining per-layer K/V tensors.

        The returned cache is the same object when one is supplied, so callers
        can keep a single cache across a generation loop.
        """
        b, t, d = x.shape
        if cache is None:
            cache = KVCache.empty(len(self.layers))
            if start_pos is None:
                start_pos = 0
        if start_pos is None:
            start_pos = cache.length
        key_len = start_pos + t
        mask = build_incremental_mask(
            attention_mask, t, key_len, start_pos, x.device, x.dtype
        )
        for index, layer in enumerate(self.layers):
            x = layer.forward_cached(
                x, attn_mask=mask, cache=cache.layers[index]
            )
        return self.norm(x), cache


def attention_mac_estimate(cfg: ModelConfig, seq_len: int) -> int:
    """Per-layer estimate for QKV, attention scores/values and FFN."""
    d = cfg.hidden_dim
    f = cfg.context_ff_mult * d
    return int(4 * seq_len * d * d + 2 * seq_len * seq_len * d + 2 * seq_len * d * f)
