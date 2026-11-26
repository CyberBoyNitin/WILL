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


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings applied to SDPA queries and keys.

    Buffers are non-persistent so legacy checkpoints load without new keys.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        head_dim = cfg.hidden_dim // cfg.n_heads
        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [B, heads, T, head_dim], positions: [B, T]
        angles = torch.einsum(
            "bt,d->btd", positions.to(torch.float32), self.inv_freq.to(x.device)
        )
        emb = torch.cat((angles, angles), dim=-1).to(x.dtype)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


def repeat_kv(x: torch.Tensor, repetitions: int) -> torch.Tensor:
    """Expand grouped KV heads to the query head count for SDPA."""
    if repetitions == 1:
        return x
    return torch.repeat_interleave(x, repetitions, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.dropout = cfg.dropout
        self.use_rope = bool(cfg.use_rope)
        self.rope = RotaryEmbedding(cfg) if self.use_rope else None
        if self.n_kv_heads == self.n_heads:
            # Preserve the exact parameter names of v1-v3 checkpoints.
            self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim)
        else:
            self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.k_proj = nn.Linear(cfg.hidden_dim, self.n_kv_heads * self.head_dim)
            self.v_proj = nn.Linear(cfg.hidden_dim, self.n_kv_heads * self.head_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def _project(self, x: torch.Tensor):
        b, t, _ = x.shape
        if self.n_kv_heads == self.n_heads:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        b, _, t, _ = q.shape
        if self.rope is not None and positions is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        k = repeat_kv(k, self.n_heads // self.n_kv_heads)
        v = repeat_kv(v, self.n_heads // self.n_kv_heads)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )
        return (
            y.transpose(1, 2)
            .contiguous()
            .view(b, t, self.n_heads * self.head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self._project(x)
        y = self._attend(q, k, v, attn_mask, positions)
        return self.resid_dropout(self.out(y))

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cache=None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention over a cached prefix plus the current query tokens."""
        q, k, v = self._project(x)
        if cache is not None:
            cache.append(k, v)
            k, v = cache.key, cache.value
        # RoPE must use absolute positions, including the cached prefix.  If a
        # caller omits positions, derive them from the cache length.
        if self.rope is not None and positions is None:
            start = cache.length - x.shape[1] if cache is not None else 0
            positions = (
                torch.arange(start, start + x.shape[1], device=x.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )
        y = self._attend(q, k, v, attn_mask, positions)
        return self.resid_dropout(self.out(y))


class CausalTransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        hidden = int(cfg.context_ff_mult * cfg.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, positions=positions)
        x = x + self.ff(self.norm2(x))
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cache=None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn.forward_cached(
            self.norm1(x), attn_mask=attn_mask, cache=cache, positions=positions
        )
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
        positions = (
            torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
            if self.cfg.use_rope
            else None
        )
        for layer in self.layers:
            x = layer(x, attn_mask=mask, positions=positions)
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
        positions = (
            torch.arange(start_pos, start_pos + t, device=x.device)
            .unsqueeze(0)
            .expand(x.shape[0], t)
            if self.cfg.use_rope
            else None
        )
        for index, layer in enumerate(self.layers):
            x = layer.forward_cached(
                x,
                attn_mask=mask,
                cache=cache.layers[index],
                positions=positions,
            )
        return self.norm(x), cache


def attention_mac_estimate(cfg: ModelConfig, seq_len: int) -> int:
    """Per-layer estimate for QKV, attention scores/values and FFN."""
    d = cfg.hidden_dim
    f = int(cfg.context_ff_mult * d)
    return int(4 * seq_len * d * d + 2 * seq_len * seq_len * d + 2 * seq_len * d * f)
