"""Typed results returned by :class:`will_engine.WILLForCausalLM`."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    text: str
    generated_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    latency_ms: float
    mean_depth: float
    token_evaluations: int
    estimated_flops: int
    mode: str
    checkpoint_id: str | None
    memory_session_id: str | None
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # ``raw`` is the full model payload; keep the public dict compact but
        # retain it under a namespaced key for callers that need the details.
        data["will"] = data.pop("raw")
        return data
