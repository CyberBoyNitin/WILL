"""Parameter, MAC and process metrics.

MAC/FLOP values are analytic estimates from tensor shapes.  They are labelled
as estimates everywhere and are never presented as measured energy.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Any

import torch
import torch.nn as nn

from backend.config import ModelConfig
from backend.engine.context_encoder import attention_mac_estimate


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in module.parameters()))


def linear_macs(module: nn.Module) -> int:
    total = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            total += int(m.in_features * m.out_features)
    return total


def estimate_forward_macs(
    cfg: ModelConfig,
    seq_len: int,
    token_evaluations: int,
    context_layers: int | None = None,
) -> int:
    """Estimate MACs for one forward pass.

    ``token_evaluations`` is the number of times the recurrent block was
    applied to a token (hard mode can be smaller than B*T*steps).
    """
    d = cfg.hidden_dim
    layers = cfg.context_layers if context_layers is None else context_layers
    context = layers * attention_mac_estimate(cfg, seq_len)
    # Recurrent block linear MACs per token, derived from the same formula as
    # the actual block: context projection + context gate + step projection +
    # gated FFN.
    f = cfg.ff_mult * d
    block_per_token = 4 * d * d + 2 * d * f + 2 * d * d
    recurrent = int(token_evaluations * block_per_token)
    head = int(seq_len * d * cfg.vocab_size)
    memory = int(seq_len * d * cfg.memory_dim * 3)
    return int(context + recurrent + head + memory)


def flops_from_macs(macs: int) -> int:
    """Conventional 2 FLOPs per multiply-accumulate."""
    return int(2 * macs)


def module_breakdown(model: nn.Module) -> list[dict[str, Any]]:
    """Return direct-parameter counts per named module (no double counting)."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, module in model.named_modules():
        if name == "" or name in seen:
            continue
        direct = list(module.named_parameters(recurse=False))
        buffers = list(module.named_buffers(recurse=False))
        if not direct and not buffers:
            continue
        seen.add(name)
        params = sum(p.numel() for _, p in direct)
        trainable = sum(p.numel() for _, p in direct if p.requires_grad)
        shapes = {
            pname: list(p.shape) for pname, p in direct if p.numel() > 0
        }
        rows.append(
            {
                "name": name,
                "type": type(module).__name__,
                "parameters": int(params),
                "trainable": int(trainable),
                "frozen": int(params - trainable),
                "shapes": shapes,
            }
        )
    return rows


def process_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return float("nan")


def cpu_percent() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).cpu_percent(interval=None))
    except Exception:
        return float("nan")


def gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "total_memory_mb": None,
            "allocated_mb": None,
            "reserved_mb": None,
        }
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "name": props.name,
        "total_memory_mb": round(props.total_memory / (1024 * 1024), 1),
        "allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 * 1024), 1),
        "reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 * 1024), 1),
    }


def hardware_info() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "gpu": gpu_info(),
        "cpu_count": os.cpu_count(),
        "cpu_threads": torch.get_num_threads(),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "process_rss_mb": round(process_rss_mb(), 1),
        "timestamp": time.time(),
    }


class Stopwatch:
    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = device

    def __enter__(self) -> "Stopwatch":
        self._sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self._sync()
        self.elapsed = time.perf_counter() - self.start

    def _sync(self) -> None:
        if self.device is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
