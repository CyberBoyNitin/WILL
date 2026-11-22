"""Adaptive-computation controller for WILL."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        batch, seq, _ = hidden.shape
        steps = (
            int(max_steps)
            if max_steps is not None
            else self.cfg.max_steps if mode == "fixed_depth" else self.cfg.resolved_adaptive_steps()
        )
        if steps < 1:
            raise ValueError("max_steps must be >= 1")
        steps = min(steps, self.cfg.max_steps)
        valid = (
            torch.ones(batch, seq, device=hidden.device, dtype=torch.float32)
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

    def _step_emb(self, step: int, device: torch.device) -> torch.Tensor:
        return self.block.step_embedding(torch.tensor(step, device=device))

    def _step_projection_tables(
        self, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.block.step_embedding.weight[:steps]
        return (
            embeddings,
            self.block.step_proj(embeddings),
            self.halting_gate.step_proj(embeddings),
        )

    def _fixed(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        valid: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        h = hidden
        for step in range(steps):
            h = self.block(h, context, self._step_emb(step, hidden.device))
        valid_count = int(valid.sum().item())
        active = [valid_count] * steps + [0] * (self.cfg.max_steps - steps)
        histogram = [0] * (self.cfg.max_steps + 1)
        histogram[steps] = valid_count
        return self.aggregator([h], None), {
            "mode": "fixed_depth",
            "steps_configured": steps,
            "token_depth": valid * float(steps),
            "mean_depth": float(steps),
            "median_depth": float(steps),
            "min_depth": float(steps) if valid_count else 0.0,
            "max_depth": float(steps) if valid_count else 0.0,
            "depth_histogram": histogram,
            "step_active_counts": active,
            "step_halt_counts": [0] * (steps - 1) + [valid_count] + [0] * (self.cfg.max_steps - steps),
            "step_active_mass": [float(x) for x in active],
            "ponder_cost": torch.tensor(float(steps), device=hidden.device),
            "expected_steps": float(steps),
            "adaptive_fraction": 0.0,
            "total_token_evaluations": valid_count * steps,
            "expected_inference_evaluations": float(valid_count * steps),
            "valid_tokens": valid_count,
            "max_steps_used": steps,
            "halt_prob_at_exit": torch.zeros_like(valid),
        }

    def _soft(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        valid: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        min_steps = min(self.cfg.min_steps, max_steps)
        h = hidden
        cumulative = torch.zeros_like(valid, dtype=hidden.dtype)
        states: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        for step in range(max_steps):
            step_emb = self._step_emb(step, hidden.device)
            h = self.block(h, context, step_emb)
            probability = self.halting_gate(h, step_emb, context)
            if step < min_steps - 1:
                probability = torch.zeros_like(probability)
            weight = 1.0 - cumulative if step == max_steps - 1 else torch.minimum(probability, (1.0 - cumulative).clamp_min(0.0))
            weight = weight.clamp(0.0, 1.0) * valid
            states.append(h)
            weights.append(weight)
            probabilities.append(probability)
            cumulative = (cumulative + weight).clamp(max=1.0)
        weight_stack = torch.stack(weights, dim=-1)
        token_depth = (weight_stack * torch.arange(1, max_steps + 1, device=hidden.device, dtype=hidden.dtype)).sum(dim=-1)
        depth_values = token_depth[valid.bool()].detach()
        valid_count = int(valid.sum().item())
        halt_mass = weight_stack.sum(dim=(0, 1))
        active_mass = torch.flip(torch.cumsum(torch.flip(weight_stack, [-1]), -1), [-1]).sum(dim=(0, 1))
        histogram = [0.0] * (self.cfg.max_steps + 1)
        for step in range(max_steps):
            histogram[step + 1] = float(halt_mass[step].detach().item())
        ponder = (token_depth * valid).sum() / valid.sum().clamp_min(1.0)
        return self.aggregator(states, weights), {
            "mode": "adaptive_soft",
            "steps_configured": max_steps,
            "token_depth": token_depth,
            "mean_depth": float(depth_values.mean().item()) if valid_count else 0.0,
            "median_depth": float(depth_values.median().item()) if valid_count else 0.0,
            "min_depth": float(depth_values.min().item()) if valid_count else 0.0,
            "max_depth": float(depth_values.max().item()) if valid_count else 0.0,
            "depth_histogram": histogram,
            "step_active_counts": [float(x) for x in active_mass.detach().tolist()],
            "step_halt_counts": [float(x) for x in halt_mass.detach().tolist()],
            "step_active_mass": [float(x) for x in active_mass.detach().tolist()],
            "ponder_cost": ponder,
            "expected_steps": float(ponder.detach().item()),
            "adaptive_fraction": float((depth_values < max_steps - 1e-6).float().mean().item()) if valid_count else 0.0,
            "total_token_evaluations": valid_count * max_steps,
            "expected_inference_evaluations": float(active_mass.sum().detach().item()),
            "valid_tokens": valid_count,
            "max_steps_used": max_steps,
            "halt_prob_at_exit": torch.stack(probabilities, dim=-1).mul(weight_stack).sum(-1).detach(),
        }

    def _hard(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        valid: torch.Tensor,
        max_steps: int,
        halt_epsilon: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Hard early-exit using a compact active-token working set.

        The initial valid-token gather happens once.  Thereafter ``h``, context
        projections, cumulative halt mass, and original-token indices all stay
        compacted together.  A token is scattered back only when it halts (or
        at the final forced-exit step).  This preserves real token-level
        skipping while avoiding the full-size gather/scatter cycle that used to
        surround every recurrent step.
        """
        if not 0.0 < halt_epsilon < 1.0:
            raise ValueError("halt_epsilon must be in (0,1)")
        batch, seq, dim = hidden.shape
        min_steps = min(self.cfg.min_steps, max_steps)
        with torch.no_grad():
            flat_h = hidden.detach().reshape(-1, dim).clone()
            flat_context = context.detach().reshape(-1, dim)
            valid_flat = valid.reshape(-1).bool()
            active = valid_flat.clone()
            output = torch.zeros_like(flat_h)
            depth = torch.zeros(flat_h.shape[0], dtype=torch.long, device=hidden.device)
            cumulative = torch.zeros(flat_h.shape[0], dtype=hidden.dtype, device=hidden.device)
            exit_probability = torch.zeros_like(cumulative)
            step_active: list[int] = []
            step_halt: list[int] = []
            context_projection = self.block.context_proj(flat_context)
            context_gate_constant = F.linear(
                context_projection,
                self.block.context_gate.weight[:, dim:],
                self.block.context_gate.bias,
            )
            # A gate projection is needed whenever at least one step can halt
            # before the final forced-exit step.  With min_steps=2 and
            # max_steps=3 that is step 1, so the test is ``> 0``, not ``>= 2``.
            gate_context = (
                self.halting_gate.context_proj(flat_context)
                if max_steps > min_steps
                else None
            )
            step_embeddings, block_steps, gate_steps = self._step_projection_tables(max_steps)

            for step in range(max_steps):
                indices = active.nonzero(as_tuple=False).squeeze(1)
                active_count = int(indices.numel())
                step_active.append(active_count)
                if active_count == 0:
                    step_halt.append(0)
                    continue
                h_active = flat_h.index_select(0, indices)
                context_active = flat_context.index_select(0, indices)
                h_new = self.block(
                    h_active,
                    context_active,
                    step_embeddings[step],
                    context_proj=context_projection.index_select(0, indices),
                    context_gate_c=context_gate_constant.index_select(0, indices),
                    step_proj=block_steps[step],
                )
                flat_h.index_copy_(0, indices, h_new)
                depth.index_add_(0, indices, torch.ones_like(indices))
                forced_continue = step < min_steps - 1
                forced_exit = step == max_steps - 1
                if forced_exit:
                    output.index_copy_(0, indices, h_new)
                    active[indices] = False
                    step_halt.append(active_count)
                    continue
                if forced_continue:
                    step_halt.append(0)
                    continue
                assert gate_context is not None
                probability = self.halting_gate(
                    h_new,
                    step_embeddings[step],
                    context_active,
                    context_proj=gate_context.index_select(0, indices),
                    step_proj=gate_steps[step].expand_as(h_new),
                )
                cumulative.index_add_(0, indices, probability)
                halted_local = cumulative.index_select(0, indices) >= (1.0 - halt_epsilon)
                halted_count = int(halted_local.sum().item())
                if halted_count:
                    halted_indices = indices.index_select(0, halted_local.nonzero(as_tuple=False).squeeze(1))
                    output.index_copy_(0, halted_indices, flat_h.index_select(0, halted_indices))
                    exit_probability.index_copy_(0, halted_indices, probability[halted_local])
                    active[halted_indices] = False
                step_halt.append(halted_count)

            if active.any():
                indices = active.nonzero(as_tuple=False).squeeze(1)
                output.index_copy_(0, indices, flat_h.index_select(0, indices))
            token_depth = depth.view(batch, seq).float()
            valid_count = int(valid_flat.sum().item())
            depth_values = token_depth.reshape(-1)[valid_flat].detach()
            active_counts = step_active + [0] * (self.cfg.max_steps - len(step_active))
            halt_counts = step_halt + [0] * (self.cfg.max_steps - len(step_halt))
            histogram = [0] * (self.cfg.max_steps + 1)
            if valid_count:
                counts = torch.bincount(depth_values.long(), minlength=self.cfg.max_steps + 1)
                histogram = [int(x) for x in counts[: self.cfg.max_steps + 1].tolist()]
            stats = {
                "mode": "adaptive_hard",
                "steps_configured": max_steps,
                "token_depth": token_depth,
                "mean_depth": float(depth_values.mean().item()) if valid_count else 0.0,
                "median_depth": float(depth_values.median().item()) if valid_count else 0.0,
                "min_depth": float(depth_values.min().item()) if valid_count else 0.0,
                "max_depth": float(depth_values.max().item()) if valid_count else 0.0,
                "depth_histogram": histogram,
                "step_active_counts": active_counts,
                "step_halt_counts": halt_counts,
                "step_active_mass": [float(x) for x in active_counts],
                "ponder_cost": torch.tensor(float(depth_values.mean().item()) if valid_count else 0.0, device=hidden.device),
                "expected_steps": float(depth_values.mean().item()) if valid_count else 0.0,
                "adaptive_fraction": float((depth_values < max_steps).float().mean().item()) if valid_count else 0.0,
                "total_token_evaluations": int(sum(active_counts)),
                "expected_inference_evaluations": float(sum(active_counts)),
                "valid_tokens": valid_count,
                "max_steps_used": int(depth_values.max().item()) if valid_count else 0,
                "halt_prob_at_exit": exit_probability.view(batch, seq),
                "halt_epsilon": halt_epsilon,
            }
        return self.aggregator([output.view(batch, seq, dim)], None), stats


def serializable_stats(stats: dict[str, Any], tokenizer=None) -> dict[str, Any]:
    """Convert tensors and per-token depths to JSON-safe structures."""
    output: dict[str, Any] = {}
    for key, value in stats.items():
        if key in ("token_depth", "halt_prob_at_exit"):
            continue
        if isinstance(value, torch.Tensor):
            output[key] = value.detach().cpu().tolist()
        elif isinstance(value, float):
            output[key] = round(value, 6) if math.isfinite(value) else None
        else:
            output[key] = value
    depth = stats.get("token_depth")
    if isinstance(depth, torch.Tensor):
        output["token_depth"] = depth.detach().cpu().tolist()
    return output
