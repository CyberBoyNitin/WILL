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
    accuracy_gated: bool = False,
    depth_target_lambda: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Next-token loss plus an optional accuracy-gated ponder cost.

    The early-exit reward applies only where the model is actually correct and
    confident.  Measuring the first curriculum showed that hard examples are
    often *confidently wrong* (median CE 0.004), so a pure ``exp(-CE)`` weight
    still penalises them.  We therefore use sequence-level correctness for the
    exit reward and add a per-token depth target: fully-correct sequences
    target depth 1, failed sequences target ``max_steps``.  The depth-target
    term is what forces the gate open on hard inputs while the exit reward
    keeps easy inputs shallow.
    """
    lm = language_model_loss(logits, labels)
    ponder = stats.get("ponder_cost")
    if not isinstance(ponder, torch.Tensor):
        ponder = torch.tensor(float(ponder or 0.0), device=logits.device)
    gated_weight = None
    depth_target_loss = torch.zeros((), device=logits.device)
    if accuracy_gated:
        answer = (labels != -100).float()
        per_token_ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        correct = (logits.argmax(dim=-1) == labels) & (labels != -100)
        confidence = torch.exp(-per_token_ce.detach()).clamp(0.0, 1.0)
        depth = stats.get("token_depth")
        if not isinstance(depth, torch.Tensor):
            depth = ponder.expand_as(labels)
        # Sequence-level correctness: a single wrong answer token means the
        # whole problem failed, so depth is expanded for the whole sequence.
        seq_correct = (correct | (labels == -100)).all(dim=1, keepdim=True)
        # Exit reward only on fully-correct sequences; failed problems get no
        # early-exit pressure at all.
        gated_weight = (
            confidence * seq_correct.float() * depth * answer
        ).sum() / answer.sum().clamp_min(1.0)
        ponder = gated_weight
        # Explicit depth target: solved problem -> 1, failed -> max steps.
        max_steps = float(stats.get("steps_configured", 4))
        target = torch.where(
            seq_correct,
            torch.ones_like(depth),
            torch.full_like(depth, max_steps),
        )
        depth_target_loss = (
            ((depth - target) ** 2) * answer
        ).sum() / answer.sum().clamp_min(1.0)
    total = lm + float(ponder_lambda) * ponder
    if accuracy_gated and float(depth_target_lambda) > 0.0:
        total = total + float(depth_target_lambda) * depth_target_loss
    if memory_loss is not None and memory_lambda:
        total = total + float(memory_lambda) * memory_loss
    return total, {
        "lm_loss": float(lm.detach().item()),
        "ponder_cost": float(ponder.detach().item()),
        "ponder_weight": (
            float(gated_weight.detach().item()) if gated_weight is not None else 1.0
        ),
        "depth_target_loss": float(depth_target_loss.detach().item()),
        "memory_loss": float(memory_loss.detach().item()) if memory_loss is not None else None,
        "total_loss": float(total.detach().item()),
    }
