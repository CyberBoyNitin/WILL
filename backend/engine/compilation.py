"""Optional ``torch.compile`` with an explicit eager fallback.

Compilation is an optimisation, never a correctness requirement.  The wrapper
tries the requested mode, runs a numerical probe against the eager model, and
returns the eager model with a machine-readable reason when the compiler is
missing, unsupported, or produces different logits.

The incremental KV-cache generation path stays eager: it has data-dependent
loop bounds and a cache that is row-compacted every step, so compiling the
whole module would either graph-break heavily or freeze assumptions that the
generation path intentionally changes.  Callers can still compile the
full-sequence forward used for training and scoring.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

Probe = Callable[[nn.Module], torch.Tensor]


@dataclass
class CompileStatus:
    requested: bool
    compiled: bool
    used: str
    mode: str
    checked: bool
    max_abs_diff: float | None = None
    error: str | None = None
    reason: str | None = None
    generation_path: str = "eager"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def maybe_compile(
    model: nn.Module,
    *,
    enabled: bool = True,
    mode: str = "reduce-overhead",
    probe: Probe | None = None,
    tolerance: float = 1e-3,
) -> tuple[nn.Module, dict[str, Any]]:
    """Return ``(module_to_use, status)``.

    When compilation or the numerical probe fails, the original model is
    returned unchanged and ``status["used"] == "eager"``.  The caller should
    record ``status`` rather than assuming compilation happened.
    """
    status = CompileStatus(
        requested=bool(enabled),
        compiled=False,
        used="eager",
        mode=mode,
        checked=False,
    )
    if not enabled:
        status.reason = "compilation disabled by configuration"
        return model, status.as_dict()
    if not hasattr(torch, "compile"):
        status.error = "torch.compile is unavailable in this PyTorch build"
        status.reason = "eager fallback"
        return model, status.as_dict()

    try:
        compiled = torch.compile(model, mode=mode)
    except Exception as exc:  # pragma: no cover - depends on local toolchain
        status.error = f"{type(exc).__name__}: {exc}"
        status.reason = "torch.compile raised during setup; eager fallback"
        return model, status.as_dict()

    if probe is None:
        status.compiled = True
        status.used = "compiled"
        status.reason = "compiled without a numerical probe"
        return compiled, status.as_dict()

    try:
        with torch.no_grad():
            eager_out = probe(model)
            compiled_out = probe(compiled)
        if eager_out.shape != compiled_out.shape:
            raise RuntimeError(
                f"probe shape mismatch: {tuple(eager_out.shape)} vs "
                f"{tuple(compiled_out.shape)}"
            )
        if not bool(torch.isfinite(compiled_out).all()):
            raise RuntimeError("compiled probe produced non-finite values")
        diff = float((eager_out.float() - compiled_out.float()).abs().max().item())
        status.checked = True
        status.max_abs_diff = diff
        if diff > tolerance:
            status.error = (
                f"compiled logits differ from eager by {diff:.6g} "
                f"(tolerance {tolerance:g})"
            )
            status.reason = "numerical mismatch; eager fallback"
            return model, status.as_dict()
    except Exception as exc:
        status.error = f"{type(exc).__name__}: {exc}"
        status.reason = "compilation probe failed; eager fallback"
        return model, status.as_dict()

    status.compiled = True
    status.used = "compiled"
    status.reason = f"compiled and verified (max |diff| {status.max_abs_diff:.3g})"
    return compiled, status.as_dict()
