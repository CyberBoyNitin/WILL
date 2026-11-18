"""Adaptive computation controller.

Three execution modes share exactly the same recurrent block and halting gate:

* ``fixed_depth``   -- run a fixed number of refinement steps.
* ``adaptive_soft`` -- differentiable ACT-style training / inference.
* ``adaptive_hard`` -- inference-only hard early exit that actually removes
  halted positions from subsequent recurrent block evaluations.

The hard path is deliberately wrapped in ``torch.no_grad``; it is an inference
optimisation, not a training path.  Soft ACT trains all configured steps and
does **not** claim training-time compute savings.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from backend.config import ModelConfig
from backend.engine.halting_gate import HaltingGate
from backend.engine.output_head import AdaptiveOutputAggregator
from backend.engine.recurrent_block import RecurrentReasoningBlock

VALID_MODES = ("fixed_depth", "adaptive_soft", "adaptive_hard")


class AdaptiveComputationController(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        block: RecurrentReasoningBlock,
        halting_gate: HaltingGate,
        aggregator: AdaptiveOutputAggregator,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.block = block
        self.halting_gate = halting_gate
        self.aggregator = aggregator

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mode: str = "adaptive_soft",
        max_steps: int | None = None,
        halt_epsilon: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if hidden.shape != context.shape:
            raise ValueError("hidden and context must have the same shape")
        b, t, d = hidden.shape
        steps = int(max_steps if max_steps is not None else self.cfg.max_steps)
        if steps < 1:
            raise ValueError("max_steps must be >= 1")
        steps = min(steps, self.cfg.max_steps)
        valid = (
            torch.ones(b, t, device=hidden.device, dtype=torch.float32)
            if attention_mask is None
            else attention_mask.to(device=hidden.device, dtype=torch.float32)
        )
        if mode == "fixed_depth":
            return self._fixed(hidden, context, valid, steps)
        if mode == "adaptive_soft":
            return self._soft(hidden, context, valid, steps)
        return self._hard(
            hidden,
            context,
            valid,
            steps,
            self.cfg.halt_epsilon if halt_epsilon is None else halt_epsilon,
        )

    # ------------------------------------------------------------------
    def _step_emb(self, step: int, device: torch.device) -> torch.Tensor:
        idx = torch.tensor(step, device=device, dtype=torch.long)
        return self.block.step_embedding(idx)

    def _fixed(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        valid: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        b, t, _ = hidden.shape
        h = hidden
        for s in range(steps):
            h = self.block(h, context, self._step_emb(s, hidden.device))
        out = self.aggregator([h], None)
        valid_count = int(valid.sum().item())
        depth = valid * float(steps)
        active = [valid_count] * steps
        active += [0] * (self.cfg.max_steps - steps)
        histogram = [0] * (self.cfg.max_steps + 1)
        if steps <= self.cfg.max_steps:
            histogram[steps] = valid_count
        stats = {
            "mode": "fixed_depth",
            "steps_configured": steps,
            "token_depth": depth,
            "mean_depth": float(steps),
            "median_depth": float(steps),
            "min_depth": float(steps) if valid_count else 0.0,
            "max_depth": float(steps) if valid_count else 0.0,
            "depth_histogram": histogram,
            "step_active_counts": active,
            "step_halt_counts": [0] * (steps - 1) + [valid_count]
            + [0] * (self.cfg.max_steps - steps),
            "step_active_mass": [float(valid_count)] * steps
            + [0.0] * (self.cfg.max_steps - steps),
            "ponder_cost": torch.tensor(float(steps), device=hidden.device),
            "expected_steps": float(steps),
            "adaptive_fraction": 0.0,
            "total_token_evaluations": valid_count * steps,
            "expected_inference_evaluations": float(valid_count * steps),
            "valid_tokens": valid_count,
            "max_steps_used": steps,
            "halt_prob_at_exit": torch.zeros_like(valid),
        }
        return out, stats

    # ------------------------------------------------------------------
    def _soft(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        valid: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        b, t, _ = hidden.shape
        min_steps = min(self.cfg.min_steps, max_steps)
        h = hidden
        cum = torch.zeros(b, t, device=hidden.device, dtype=hidden.dtype)
        states: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        for s in range(max_steps):
            step_emb = self._step_emb(s, hidden.device)
            h = self.block(h, context, step_emb)
            p = self.halting_gate(h, step_emb, context)
            if s < min_steps - 1:
                p = torch.zeros_like(p)
            if s == max_steps - 1:
                w = 1.0 - cum
            else:
                remaining = (1.0 - cum).clamp_min(0.0)
                w = torch.minimum(p, remaining)
            w = w.clamp(0.0, 1.0)
            w = w * valid
            weights.append(w)
            states.append(h)
            probs.append(p)
            cum = (cum + w).clamp(max=1.0)

        out = self.aggregator(states, weights)
        step_index = torch.arange(1, max_steps + 1, device=hidden.device, dtype=hidden.dtype)
        expected = torch.stack(weights, dim=-1) * step_index
        token_depth = expected.sum(dim=-1)
        ponder = (token_depth * valid).sum() / valid.sum().clamp_min(1.0)

        weight_stack = torch.stack(weights, dim=-1)  # B,T,S
        halt_mass = weight_stack.sum(dim=(0, 1))
        active_mass = torch.flip(torch.cumsum(torch.flip(weight_stack, dims=[-1]), dim=-1), dims=[-1])
        active_mass = active_mass.sum(dim=(0, 1))
        valid_count = int(valid.sum().item())
        depth_values = token_depth[valid.bool()].detach()
        hist = [0.0] * (self.cfg.max_steps + 1)
        for i in range(max_steps):
            hist[i + 1] = float(halt_mass[i].detach().item())
        # Round expected histogram counts for display while keeping raw masses.
        stats = {
            "mode": "adaptive_soft",
            "steps_configured": max_steps,
            "token_depth": token_depth,
            "mean_depth": float(depth_values.mean().item()) if valid_count else 0.0,
            "median_depth": float(depth_values.median().item()) if valid_count else 0.0,
            "min_depth": float(depth_values.min().item()) if valid_count else 0.0,
            "max_depth": float(depth_values.max().item()) if valid_count else 0.0,
            "depth_histogram": hist,
            "step_active_counts": [float(x) for x in active_mass.detach().tolist()],
            "step_halt_counts": [float(x) for x in halt_mass.detach().tolist()],
            "step_active_mass": [float(x) for x in active_mass.detach().tolist()],
            "ponder_cost": ponder,
            "expected_steps": float(ponder.detach().item()),
            "adaptive_fraction": (
                float((depth_values < max_steps - 1e-6).float().mean().item())
                if valid_count
                else 0.0
            ),
            "total_token_evaluations": valid_count * max_steps,
            "expected_inference_evaluations": float(active_mass.sum().detach().item()),
            "valid_tokens": valid_count,
            "max_steps_used": max_steps,
            "halt_prob_at_exit": torch.stack(probs, dim=-1).mul(weight_stack).sum(-1).detach(),
        }
        return out, stats

    # ------------------------------------------------------------------
    def _hard(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        valid: torch.Tensor,
        max_steps: int,
        halt_epsilon: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not 0.0 < halt_epsilon < 1.0:
            raise ValueError("halt_epsilon must be in (0,1)")
        b, t, d = hidden.shape
        min_steps = min(self.cfg.min_steps, max_steps)
        with torch.no_grad():
            flat_h = hidden.detach().reshape(b * t, d).clone()
            flat_ctx = context.detach().reshape(b * t, d)
            flat_valid = valid.reshape(-1).bool()
            out = torch.zeros_like(flat_h)
            depth = torch.zeros(b * t, device=hidden.device, dtype=torch.long)
            cum = torch.zeros(b * t, device=hidden.device, dtype=hidden.dtype)
            exit_prob = torch.zeros(b * t, device=hidden.device, dtype=hidden.dtype)
            active = flat_valid.clone()
            step_active: list[int] = []
            step_halt: list[int] = []
            for s in range(max_steps):
                idx = active.nonzero(as_tuple=False).squeeze(1)
                step_active.append(int(idx.numel()))
                if idx.numel() == 0:
                    step_halt.append(0)
                    continue
                step_emb = self._step_emb(s, hidden.device)
                h_act = flat_h.index_select(0, idx)
                c_act = flat_ctx.index_select(0, idx)
                h_new = self.block(h_act, c_act, step_emb)
                p = self.halting_gate(h_new, step_emb, c_act)
                if s < min_steps - 1:
                    p = torch.zeros_like(p)
                flat_h.index_copy_(0, idx, h_new)
                cum.index_add_(0, idx, p)
                depth.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
                exit_prob.index_copy_(0, idx, p)

                if s == max_steps - 1:
                    newly = active.clone()
                else:
                    newly = active & (cum >= 1.0 - halt_epsilon)
                if newly.any():
                    nidx = newly.nonzero(as_tuple=False).squeeze(1)
                    out.index_copy_(0, nidx, flat_h.index_select(0, nidx))
                    active = active & ~newly
                step_halt.append(int(newly.sum().item()))
                if not active.any():
                    break

            # Any position that was somehow left active (defensive) gets the
            # last computed state.
            if active.any():
                aidx = active.nonzero(as_tuple=False).squeeze(1)
                out.index_copy_(0, aidx, flat_h.index_select(0, aidx))

            out = out.view(b, t, d)
            token_depth = depth.view(b, t).float()
            valid_count = int(flat_valid.sum().item())
            depth_valid = token_depth.reshape(-1)[flat_valid].detach()
            active_counts = step_active + [0] * (max_steps - len(step_active))
            halt_counts = step_halt + [0] * (max_steps - len(step_halt))
            hist = [0] * (self.cfg.max_steps + 1)
            if valid_count:
                counts = torch.bincount(
                    depth_valid.long(), minlength=self.cfg.max_steps + 1
                )
                hist = [int(x) for x in counts[: self.cfg.max_steps + 1].tolist()]
            stats = {
                "mode": "adaptive_hard",
                "steps_configured": max_steps,
                "token_depth": token_depth,
                "mean_depth": float(depth_valid.mean().item()) if valid_count else 0.0,
                "median_depth": float(depth_valid.median().item()) if valid_count else 0.0,
                "min_depth": float(depth_valid.min().item()) if valid_count else 0.0,
                "max_depth": float(depth_valid.max().item()) if valid_count else 0.0,
                "depth_histogram": hist,
                "step_active_counts": active_counts,
                "step_halt_counts": halt_counts,
                "step_active_mass": [float(x) for x in active_counts],
                "ponder_cost": torch.tensor(
                    float(depth_valid.mean().item()) if valid_count else 0.0,
                    device=hidden.device,
                ),
                "expected_steps": float(depth_valid.mean().item()) if valid_count else 0.0,
                "adaptive_fraction": (
                    float((depth_valid < max_steps).float().mean().item())
                    if valid_count
                    else 0.0
                ),
                "total_token_evaluations": int(sum(active_counts)),
                "expected_inference_evaluations": float(sum(active_counts)),
                "valid_tokens": valid_count,
                "max_steps_used": int(depth_valid.max().item()) if valid_count else 0,
                "halt_prob_at_exit": exit_prob.view(b, t),
                "halt_epsilon": halt_epsilon,
            }
        out = self.aggregator([out], None)
        return out, stats


def serializable_stats(stats: dict[str, Any], tokenizer=None) -> dict[str, Any]:
    """Convert tensors and per-token depths to JSON-safe structures."""
    out: dict[str, Any] = {}
    token_depth = stats.get("token_depth")
    for key, value in stats.items():
        if key in ("token_depth", "halt_prob_at_exit"):
            continue
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().tolist()
        elif isinstance(value, float):
            out[key] = round(value, 6) if math.isfinite(value) else None
        else:
            out[key] = value
    if isinstance(token_depth, torch.Tensor):
        # Keep batch=1 flattened for the API; batch>1 is nested.
        td = token_depth.detach().cpu()
        out["token_depth"] = td.tolist() if td.ndim > 1 else td.tolist()
    return out
