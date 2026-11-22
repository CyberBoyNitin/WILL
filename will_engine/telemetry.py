"""Process-wide telemetry for the WILL engine.

The counters are deliberately additive and honest: they measure what the
engine did, not what it might save.  No energy figure is reported because no
power measurement is taken.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _empty_bucket() -> dict[str, float]:
    return {
        "requests": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "token_evaluations": 0.0,
        "estimated_flops": 0.0,
        "generation_seconds": 0.0,
    }


@dataclass
class Telemetry:
    started_at: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    requests: int = 0
    streaming_requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_evaluations: int = 0
    estimated_flops: int = 0
    generation_seconds: float = 0.0
    absorb_calls: int = 0
    absorbed_chars: int = 0
    absorbed_chunks: int = 0
    memory_updates: int = 0
    last_request_at: float | None = None
    by_mode: dict[str, dict[str, float]] = field(default_factory=dict)
    by_checkpoint: dict[str, dict[str, float]] = field(default_factory=dict)
    last_error: str | None = None

    def record_generation(
        self,
        *,
        checkpoint_id: str | None,
        mode: str,
        prompt_tokens: int,
        completion_tokens: int,
        token_evaluations: int,
        estimated_flops: int,
        seconds: float,
        streamed: bool = False,
    ) -> None:
        with self.lock:
            self.requests += 1
            if streamed:
                self.streaming_requests += 1
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)
            self.token_evaluations += int(token_evaluations)
            self.estimated_flops += int(estimated_flops)
            self.generation_seconds += float(seconds)
            self.last_request_at = time.time()
            for table, key in (
                (self.by_mode, mode),
                (self.by_checkpoint, checkpoint_id or "fresh"),
            ):
                bucket = table.setdefault(key, _empty_bucket())
                bucket["requests"] += 1
                bucket["prompt_tokens"] += int(prompt_tokens)
                bucket["completion_tokens"] += int(completion_tokens)
                bucket["token_evaluations"] += int(token_evaluations)
                bucket["estimated_flops"] += int(estimated_flops)
                bucket["generation_seconds"] += float(seconds)

    def record_absorb(
        self, *, chars: int, chunks: int, updates: int
    ) -> None:
        with self.lock:
            self.absorb_calls += 1
            self.absorbed_chars += int(chars)
            self.absorbed_chunks += int(chunks)
            self.memory_updates += int(updates)

    def record_error(self, message: str) -> None:
        with self.lock:
            self.errors += 1
            self.last_error = message[:500]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            uptime = max(1e-9, time.time() - self.started_at)
            total_tokens = self.prompt_tokens + self.completion_tokens
            return {
                "format": "will-telemetry-v1",
                "uptime_seconds": round(uptime, 3),
                "generation": {
                    "requests": self.requests,
                    "streaming_requests": self.streaming_requests,
                    "errors": self.errors,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": total_tokens,
                    "token_evaluations": self.token_evaluations,
                    "estimated_flops": self.estimated_flops,
                    "generation_seconds": round(self.generation_seconds, 4),
                    "tokens_per_second": round(
                        total_tokens / max(1e-9, self.generation_seconds), 3
                    ),
                    "by_mode": {
                        key: dict(value) for key, value in self.by_mode.items()
                    },
                    "by_checkpoint": {
                        key: dict(value)
                        for key, value in self.by_checkpoint.items()
                    },
                },
                "memory": {
                    "absorb_calls": self.absorb_calls,
                    "absorbed_chars": self.absorbed_chars,
                    "absorbed_chunks": self.absorbed_chunks,
                    "fast_state_updates": self.memory_updates,
                },
                "last_request_at": self.last_request_at,
                "last_error": self.last_error,
                "energy_measured": False,
                "notes": [
                    "Counters cover this process since start; they reset on restart.",
                    "No energy or efficiency saving is claimed without a power measurement.",
                ],
            }


GLOBAL_TELEMETRY = Telemetry()
