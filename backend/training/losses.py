"""Training objectives.

The primary objective is next-token cross entropy.  The only additional term
enabled by default is the ACT ponder cost, which penalises expected recurrent
depth and therefore asks the model to buy accuracy with as little computation
as it can.  Optional memory and calibration terms are explicitly opt-in; they
are not silently combined.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def language_model_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    stats: dict[str, Any],
    ponder_lambda: float = 0.0,
    memory_lambda: float = 0.0,
    memory_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    lm = language_model_loss(logits, labels)
    ponder = stats.get("ponder_cost")
    if not isinstance(ponder, torch.Tensor):
        ponder = torch.tensor(float(ponder or 0.0), device=logits.device)
    total = lm + float(ponder_lambda) * ponder
    if memory_loss is not None and memory_lambda:
        total = total + float(memory_lambda) * memory_loss
    return total, {
        "lm_loss": float(lm.detach().item()),
        "ponder_cost": float(ponder.detach().item()),
        "memory_loss": float(memory_loss.detach().item()) if memory_loss is not None else None,
        "total_loss": float(total.detach().item()),
    }
