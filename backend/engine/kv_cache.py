"""Incremental key/value cache for the causal context encoder.

The cache is deliberately explicit instead of relying on a framework-specific
``past_key_values`` object.  It stores one contiguous key and value tensor per
context layer, which makes the decode path a small number of predictable tensor
operations and keeps the public model API stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LayerKVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.key is None else int(self.key.shape[2])

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if self.key is None:
            # Detach from the fused QKV allocation so the cache only retains
            # the K/V storage it actually needs (and memory_bytes is honest).
            self.key = key.contiguous()
            self.value = value.contiguous()
            return
        self.key = torch.cat([self.key, key], dim=2)
        self.value = torch.cat([self.value, value], dim=2)

    def detach_(self) -> None:
        if self.key is not None:
            self.key = self.key.detach()
        if self.value is not None:
            self.value = self.value.detach()

    def select_rows(self, indices: torch.Tensor) -> "LayerKVCache":
        if self.key is None:
            return LayerKVCache()
        return LayerKVCache(
            key=self.key.index_select(0, indices),
            value=self.value.index_select(0, indices) if self.value is not None else None,
        )


@dataclass
class KVCache:
    layers: list[LayerKVCache] = field(default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> "KVCache":
        return cls(layers=[LayerKVCache() for _ in range(n_layers)])

    @property
    def length(self) -> int:
        return self.layers[0].length if self.layers else 0

    def detach_(self) -> "KVCache":
        for layer in self.layers:
            layer.detach_()
        return self

    def memory_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            if layer.key is not None:
                total += layer.key.numel() * layer.key.element_size()
            if layer.value is not None:
                total += layer.value.numel() * layer.value.element_size()
        return int(total)

    def select_rows(self, indices: torch.Tensor) -> "KVCache":
        return KVCache(
            layers=[layer.select_rows(indices) for layer in self.layers]
        )
