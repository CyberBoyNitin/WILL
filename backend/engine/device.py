"""Unified device/AMP selection for WILL v4.

The same helper is used by training, evaluation and the FastAPI backend so a
T4 notebook, a local CPU box and a modern A100/H100 all take the same code path.
No GPU-specific numbers are ever inferred: if CUDA is unavailable the model is
moved to CPU and AMP is disabled.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when usable, otherwise CPU."""
    if isinstance(requested, torch.device):
        device = requested
    else:
        name = (requested or "auto").lower()
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def amp_dtype(device: torch.device | str, prefer_bf16: bool = True) -> torch.dtype | None:
    """Return the low-precision dtype for ``device`` or ``None`` for fp32 CPU."""
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type != "cuda":
        return None
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # T4 has no bf16 tensor cores; fp16 with GradScaler is the safe path.
    return torch.float16


@contextmanager
def autocast_for(
    device: torch.device | str,
    *,
    enabled: bool = True,
    prefer_bf16: bool = True,
) -> Iterator[torch.dtype | None]:
    """Enter CUDA autocast when appropriate, otherwise a no-op context."""
    dtype = amp_dtype(device, prefer_bf16=prefer_bf16) if enabled else None
    if dtype is not None:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield dtype
    else:
        yield None


def device_summary(device: torch.device | str) -> dict[str, object]:
    """Serialisable diagnostics used by training reports and /v1/telemetry."""
    dev = torch.device(device) if isinstance(device, str) else device
    return {
        "device": str(dev),
        "cuda": dev.type == "cuda",
        "amp_dtype": str(amp_dtype(dev)) if amp_dtype(dev) else None,
        "torch_version": torch.__version__,
    }
