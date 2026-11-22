"""Mixed-precision execution plans.

The trainer and benchmarks share one resolver so that a precision name has the
same meaning everywhere.  A plan is explicit about what actually ran: asking
for ``fp16`` never silently turns into ``fp32``; only ``auto`` may fall back,
and the reason is recorded.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Iterator

import torch

PRECISIONS = ("fp32", "bf16", "fp16", "auto")


@dataclass(frozen=True)
class PrecisionPlan:
    requested: str
    effective: str
    enabled: bool
    device_type: str
    dtype: torch.dtype | None
    grad_scaler: bool
    reason: str | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["dtype"] = None if self.dtype is None else str(self.dtype).replace("torch.", "")
        return data


def _cuda_bf16_supported() -> bool:
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def resolve_precision(name: str, device: torch.device) -> PrecisionPlan:
    """Resolve a precision request for ``device``.

    ``auto`` selects bf16 on CUDA when the GPU reports support, otherwise
    fp32.  Explicit requests are honoured or rejected; they are never silently
    downgraded, so a reported run cannot claim a precision it did not use.
    """
    if name not in PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; expected one of {PRECISIONS}")
    device_type = device.type

    if name == "auto":
        if device_type == "cuda" and _cuda_bf16_supported():
            return PrecisionPlan(
                requested=name,
                effective="bf16",
                enabled=True,
                device_type=device_type,
                dtype=torch.bfloat16,
                grad_scaler=False,
                reason="auto selected bf16 (CUDA bf16 supported)",
            )
        reason = (
            "auto selected fp32 (CUDA bf16 unsupported)"
            if device_type == "cuda"
            else "auto selected fp32 (CPU default)"
        )
        return PrecisionPlan(
            requested=name,
            effective="fp32",
            enabled=False,
            device_type=device_type,
            dtype=None,
            grad_scaler=False,
            reason=reason,
        )

    if name == "fp32":
        return PrecisionPlan(
            requested=name,
            effective="fp32",
            enabled=False,
            device_type=device_type,
            dtype=None,
            grad_scaler=False,
        )

    if device_type not in ("cpu", "cuda"):
        raise ValueError(f"autocast precision {name!r} is not supported on {device_type!r}")

    dtype = torch.bfloat16 if name == "bf16" else torch.float16
    if device_type == "cuda" and name == "bf16" and not _cuda_bf16_supported():
        raise ValueError("bf16 was requested but this CUDA device does not support it")
    if device_type == "cpu" and name == "fp16":
        reason = "CPU fp16 autocast; ops run in half precision with fp32 accumulation"
    elif device_type == "cpu":
        reason = "CPU bf16 autocast; ops run in bfloat16 with fp32 accumulation"
    else:
        reason = f"CUDA {name} autocast"

    return PrecisionPlan(
        requested=name,
        effective=name,
        enabled=True,
        device_type=device_type,
        dtype=dtype,
        grad_scaler=bool(device_type == "cuda" and name == "fp16"),
        reason=reason,
    )


@contextmanager
def autocast_context(plan: PrecisionPlan) -> Iterator[None]:
    if not plan.enabled or plan.dtype is None:
        yield
        return
    with torch.autocast(device_type=plan.device_type, dtype=plan.dtype):
        yield


def build_grad_scaler(plan: PrecisionPlan):
    """Return a CUDA GradScaler for fp16, or ``None``.

    CPU autocast keeps master weights in fp32 and does not need scaling.
    """
    if not plan.grad_scaler:
        return None
    return torch.amp.GradScaler("cuda", enabled=True)
