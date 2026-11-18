"""The unified WILL model.

WILL combines a causal context encoder, a shared recurrent reasoning block
with learned halting, and an optional session-local fast-weight memory.  The
memory subsystem is optional: the model runs correctly with no memory state.
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from backend.config import ModelConfig
from backend.engine.adaptive_controller import (
    VALID_MODES,
    AdaptiveComputationController,
    serializable_stats,
)
from backend.engine.context_encoder import CausalContextEncoder
from backend.engine.embeddings import InputEmbedding
from backend.engine.halting_gate import HaltingGate
from backend.engine.metrics import (
    count_parameters,
    estimate_forward_macs,
    flops_from_macs,
)
from backend.engine.output_head import AdaptiveOutputAggregator, OutputProjection
from backend.engine.recurrent_block import RecurrentReasoningBlock
from backend.memory.fast_memory import FastMemoryState, FastWeightMemory
from backend.memory.memory_encoder import MemoryEncoder, MemoryIntegrationGate


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class WILLModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = InputEmbedding(cfg)
        self.context_encoder = CausalContextEncoder(cfg)
        self.recurrent_block = RecurrentReasoningBlock(cfg)
        self.halting_gate = HaltingGate(cfg)
        self.aggregator = AdaptiveOutputAggregator(cfg)
        self.controller = AdaptiveComputationController(
            cfg, self.recurrent_block, self.halting_gate, self.aggregator
        )
        self.memory_encoder = MemoryEncoder(cfg)
        self.memory = FastWeightMemory(cfg)
        self.memory_gate = MemoryIntegrationGate(cfg)
        self.output_projection = OutputProjection(
            cfg, token_embedding=self.embedding.token_embedding
        )
        self.apply(_init_weights)
        with torch.no_grad():
            self.embedding.token_embedding.weight[0].zero_()
            self.recurrent_block.context_gate.bias.fill_(-1.0)
            self.halting_gate.net[-1].bias.fill_(-1.5)
            self.memory_gate.gate.bias.fill_(-2.0)
        nn.init.normal_(self.memory_encoder.token_key.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.memory_encoder.token_value.weight, mean=0.0, std=0.5)
        with torch.no_grad():
            self.memory_encoder.token_key.weight[0].zero_()
            self.memory_encoder.token_value.weight[0].zero_()

    # ------------------------------------------------------------------
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Contextual representation without recurrent refinement.

        Used by document ingestion to avoid paying for the recurrent loop when
        only a memory representation is needed.
        """
        hidden = self.embedding(input_ids)
        return self.context_encoder(hidden, attention_mask)

    def encode_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_text(input_ids, attention_mask)
        keys, values = self.memory_encoder(context, input_ids)
        return context, keys, values

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        memory_state: FastMemoryState | None = None,
        mode: str = "adaptive_soft",
        max_steps: int | None = None,
        halt_epsilon: float | None = None,
        memory_enabled: bool | None = None,
        return_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown execution mode {mode!r}")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        b, t = input_ids.shape
        hidden = self.embedding(input_ids)
        context = self.context_encoder(hidden, attention_mask)

        memory_stats: dict[str, Any] | None = None
        use_memory = memory_state is not None and (
            True if memory_enabled is None else memory_enabled
        )
        if use_memory:
            keys, _ = self.memory_encoder(context, input_ids)
            retrieved = self.memory.read(memory_state, keys)
            context, gate = self.memory_gate(context, retrieved)
            memory_stats = {
                "enabled": True,
                "kind": memory_state.kind,
                "updates": memory_state.updates,
                "gate_mean": float(gate.mean().detach().item()),
                "gate_max": float(gate.max().detach().item()),
                "retrieved_norm": float(
                    retrieved.detach().norm(dim=-1).mean().item()
                ),
                "state_bytes": memory_state.state_bytes(),
            }
        elif memory_state is not None:
            memory_stats = {
                "enabled": False,
                "kind": memory_state.kind,
                "reason": "memory disabled for this execution mode",
            }
        else:
            memory_stats = {"enabled": False, "reason": "no memory state"}

        output, stats = self.controller(
            context,
            context,
            attention_mask=attention_mask,
            mode=mode,
            max_steps=max_steps,
            halt_epsilon=halt_epsilon,
        )
        logits = self.output_projection(output)

        token_evaluations = int(stats.get("total_token_evaluations", b * t))
        macs = estimate_forward_macs(self.cfg, t, token_evaluations)
        stats.update(
            {
                "batch_size": b,
                "seq_len": t,
                "estimated_macs": macs,
                "estimated_flops": flops_from_macs(macs),
                "memory": memory_stats,
                "params_total": count_parameters(self),
                "params_trainable": count_parameters(self, trainable_only=True),
            }
        )
        return logits, stats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        attention_mask: torch.Tensor | None = None,
        memory_state: FastMemoryState | None = None,
        mode: str = "adaptive_hard",
        max_steps: int | None = None,
        halt_epsilon: float | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        max_new_tokens: int = 48,
        stop_at_eos: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("generate expects a single [1,T] sequence")
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        start = time.perf_counter()
        prompt_len = int(input_ids.shape[1])
        prompt_logits, prompt_stats = self.forward(
            input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
            mode=mode,
            max_steps=max_steps,
            halt_epsilon=halt_epsilon,
        )
        depths: list[float] = list(
            prompt_stats["token_depth"][0].detach().cpu().tolist()
        )
        total_evaluations = int(prompt_stats["total_token_evaluations"])
        generated: list[int] = []
        gen_depths: list[float] = []
        logits = prompt_logits
        for _ in range(max_new_tokens):
            if input_ids.shape[1] >= self.cfg.max_seq_len:
                break
            next_logits = logits[:, -1, :].float()
            if temperature is None or temperature <= 0:
                next_id = int(torch.argmax(next_logits, dim=-1).item())
            else:
                scaled = next_logits / max(temperature, 1e-4)
                if top_k and top_k > 0:
                    k = min(top_k, scaled.shape[-1])
                    values, indices = torch.topk(scaled, k)
                    probs = torch.softmax(values, dim=-1)
                    choice = torch.multinomial(probs, num_samples=1)
                    next_id = int(indices.gather(-1, choice).item())
                else:
                    probs = torch.softmax(scaled, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            token = torch.tensor([[next_id]], device=device, dtype=torch.long)
            input_ids = torch.cat([input_ids, token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=-1,
            )
            if stop_at_eos and next_id == tokenizer.eos_id:
                gen_depths.append(float(depths[-1]) if depths else 1.0)
                break
            logits, stats = self.forward(
                input_ids,
                attention_mask=attention_mask,
                memory_state=memory_state,
                mode=mode,
                max_steps=max_steps,
                halt_epsilon=halt_epsilon,
            )
            depth = float(stats["token_depth"][0, -1].detach().cpu().item())
            gen_depths.append(depth)
            total_evaluations += int(stats["total_token_evaluations"])
        elapsed = time.perf_counter() - start
        all_depths = depths + gen_depths
        prompt_text = tokenizer.decode(input_ids[0, :prompt_len].tolist(), skip_special=False)
        generated_text = tokenizer.decode(generated, skip_special=True)
        per_token = []
        for i, tid in enumerate(input_ids[0].tolist()):
            kind = "prompt" if i < prompt_len else "generated"
            depth = all_depths[i] if i < len(all_depths) else 0.0
            per_token.append(
                {
                    "index": i,
                    "id": int(tid),
                    "text": tokenizer.token_text(int(tid)),
                    "depth": round(float(depth), 4),
                    "kind": kind,
                }
            )
        return {
            "prompt": prompt_text,
            "text": generated_text,
            "tokens": per_token,
            "prompt_tokens": prompt_len,
            "generated_tokens": len(generated),
            "latency_ms": round(elapsed * 1000, 3),
            "tokens_per_second": round(len(generated) / elapsed, 3) if elapsed > 0 else None,
            "mean_depth": round(sum(all_depths) / max(1, len(all_depths)), 4),
            "max_depth": round(max(all_depths), 4) if all_depths else 0.0,
            "token_evaluations": total_evaluations,
            "stats": serializable_stats(prompt_stats),
            "mode": mode,
            "memory": prompt_stats.get("memory"),
        }

    # ------------------------------------------------------------------
    def model_summary(self) -> dict[str, Any]:
        return {
            "config": self.cfg.model_dump(),
            "parameters_total": count_parameters(self),
            "parameters_trainable": count_parameters(self, trainable_only=True),
            "parameters_frozen": count_parameters(self) - count_parameters(self, trainable_only=True),
            "memory_initial_parameters": self.memory.init_state().parameter_count(),
        }
