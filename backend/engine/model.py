"""The unified WILL model.

WILL combines a causal context encoder, a shared recurrent reasoning block
with learned halting, and an optional session-local fast-weight memory.  The
memory subsystem is optional: the model runs correctly with no memory state.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import ModelConfig
from backend.engine.adaptive_controller import (
    VALID_MODES,
    AdaptiveComputationController,
    serializable_stats,
)
from backend.engine.context_encoder import CausalContextEncoder
from backend.engine.embeddings import InputEmbedding
from backend.engine.halting_gate import HaltingGate
from backend.engine.kv_cache import KVCache
from backend.engine.metrics import (
    count_parameters,
    estimate_forward_macs,
    flops_from_macs,
)
from backend.engine.output_head import AdaptiveOutputAggregator, OutputProjection
from backend.engine.recurrent_block import RecurrentReasoningBlock
from backend.memory.fast_memory import FastMemoryState, FastWeightMemory
from backend.memory.memory_encoder import (
    MemoryEncoder,
    MemoryIntegrationGate,
    MemoryQueryBridge,
)


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
        # Retained so legacy checkpoints with ``memory_gate.*`` keys still load;
        # the v4 path below does not use it.
        self.memory_gate = MemoryIntegrationGate(cfg)
        # v4: learned query bridge into fast-weight key space with a
        # vector-wise sigmoid gate, replacing the scalar mem_gate.
        self.memory_bridge = MemoryQueryBridge(cfg)
        # Starts at zero so retrieval initially uses the writer's exact key
        # vector; training can learn a small query-space correction.
        self.query_scale = nn.Parameter(torch.tensor(0.0))
        self.memory_scale = nn.Parameter(torch.tensor(1.0))
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
        self._param_counts: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    def parameter_counts(self) -> tuple[int, int]:
        """Cached (total, trainable) parameter counts.

        Parameter traversal is surprisingly expensive relative to a small
        decode step.  The counts are invariant across generation; call
        :meth:`invalidate_parameter_cache` after freezing or replacing
        parameters.
        """
        if self._param_counts is None:
            self._param_counts = (
                count_parameters(self),
                count_parameters(self, trainable_only=True),
            )
        return self._param_counts

    def invalidate_parameter_cache(self) -> None:
        self._param_counts = None

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
    def _apply_memory(
        self,
        context: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        memory_state: FastMemoryState | None,
        memory_enabled: bool | None,
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None]:
        use_memory = memory_state is not None and (
            True if memory_enabled is None else memory_enabled
        )
        if use_memory:
            assert memory_state is not None
            # The Test 5 writer stores a *pooled* key for the whole prompt.
            # Query the fast weights with a matching mask-aware pooled state,
            # then broadcast the retrieved residual over the sequence.  Using a
            # per-token query here made the learned key space unreachable.
            mask = attention_mask.to(context.dtype).unsqueeze(-1)
            lengths = attention_mask.sum(dim=1).clamp_min(1)
            last_idx = (lengths - 1).long()
            row = torch.arange(context.shape[0], device=context.device)
            mean = (context * mask).sum(dim=1) / lengths.to(context.dtype).unsqueeze(1)
            pooled_hidden = F.normalize(mean + context[row, last_idx], dim=-1)
            # Writer-aligned key: the Test 5/6 writer pools the per-token keys
            # from `memory_encoder`; reproduce that exact pooling here.
            keys, _ = self.memory_encoder(context, input_ids)
            key_mask = attention_mask.unsqueeze(-1)
            pos = torch.arange(1, context.shape[1] + 1, device=context.device)
            pos = pos.view(1, -1, 1).to(keys.dtype)
            denom = (pos * key_mask).sum(dim=1).clamp_min(1.0)
            key_mean = (keys * pos * key_mask).sum(dim=1) / denom
            key_last = keys[row, last_idx]
            key_query = F.normalize(key_mean + key_last, dim=-1)
            refine, gain = self.memory_bridge(pooled_hidden)
            query = F.normalize(key_query + self.query_scale * refine, dim=-1)
            retrieved = self.memory.read(memory_state, query)
            mem_out = self.memory_bridge.readout(gain * retrieved)
            # The raw readout is O(0.1) while the context is O(10), so a plain
            # add is invisible.  Normalise the residual and learn its scale.
            mem_out = F.layer_norm(
                mem_out, (mem_out.shape[-1],)
            ) * self.memory_scale
            context = context + mem_out.unsqueeze(1)
            memory_stats = {
                "enabled": True,
                "kind": memory_state.kind,
                "updates": memory_state.updates,
                "gate_mean": float(gain.detach().mean().item()),
                "gate_max": float(gain.detach().max().item()),
                "retrieved_norm": float(
                    retrieved.detach().norm(dim=-1).mean().item()
                ),
                "state_bytes": memory_state.state_bytes(),
            }
        elif memory_state is not None:
            query = None
            memory_stats = {
                "enabled": False,
                "kind": memory_state.kind,
                "reason": "memory disabled for this execution mode",
            }
        else:
            query = None
            memory_stats = {"enabled": False, "reason": "no memory state"}
        return context, memory_stats, query

    def _forward_context(
        self,
        context: torch.Tensor,
        input_ids: torch.Tensor,
        controller_mask: torch.Tensor,
        memory_state: FastMemoryState | None,
        memory_enabled: bool | None,
        mode: str,
        max_steps: int | None,
        halt_epsilon: float | None,
        mac_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        context, memory_stats, memory_query = self._apply_memory(
            context, input_ids, controller_mask, memory_state, memory_enabled
        )
        output, stats = self.controller(
            context,
            context,
            attention_mask=controller_mask,
            mode=mode,
            max_steps=max_steps,
            halt_epsilon=halt_epsilon,
        )
        logits = self.output_projection(output)
        b, t = input_ids.shape
        token_evaluations = int(stats.get("total_token_evaluations", b * t))
        macs = estimate_forward_macs(
            self.cfg,
            int(mac_seq_len or t),
            token_evaluations,
        )
        params_total, params_trainable = self.parameter_counts()
        stats.update(
            {
                "batch_size": b,
                "seq_len": t,
                "estimated_macs": macs,
                "estimated_flops": flops_from_macs(macs),
                "memory": memory_stats,
                "params_total": params_total,
                "params_trainable": params_trainable,
            }
        )
        if memory_query is not None:
            # Exposed for the Phase C query-alignment objective; it is a small
            # [batch, memory_dim] tensor and is detached by the API serializer.
            stats["memory_query"] = memory_query
        return logits, stats

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
        hidden = self.embedding(input_ids)
        context = self.context_encoder(hidden, attention_mask)
        return self._forward_context(
            context,
            input_ids,
            attention_mask,
            memory_state,
            memory_enabled,
            mode,
            max_steps,
            halt_epsilon,
        )

    # ------------------------------------------------------------------
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        memory_state: FastMemoryState | None = None,
        mode: str = "adaptive_hard",
        max_steps: int | None = None,
        halt_epsilon: float | None = None,
        memory_enabled: bool | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any], KVCache]:
        """Process a prompt once and return logits plus a populated KV cache."""
        if mode not in VALID_MODES:
            raise ValueError(f"unknown execution mode {mode!r}")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        hidden = self.embedding(input_ids, positions=positions)
        context, cache = self.context_encoder.forward_cached(
            hidden, attention_mask, cache=None, start_pos=0
        )
        logits, stats = self._forward_context(
            context,
            input_ids,
            attention_mask,
            memory_state,
            memory_enabled,
            mode,
            max_steps,
            halt_epsilon,
        )
        return logits, stats, cache

    def decode_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: KVCache,
        memory_state: FastMemoryState | None = None,
        mode: str = "adaptive_hard",
        max_steps: int | None = None,
        halt_epsilon: float | None = None,
        memory_enabled: bool | None = None,
        positions: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any], KVCache]:
        """Decode one or more new tokens against an existing KV cache."""
        if input_ids.ndim != 2:
            raise ValueError("decode_step expects [batch, new_tokens]")
        if mode not in VALID_MODES:
            raise ValueError(f"unknown execution mode {mode!r}")
        start_pos = cache.length
        hidden = self.embedding(input_ids, positions=positions, start_pos=start_pos)
        context, cache = self.context_encoder.forward_cached(
            hidden,
            attention_mask,
            cache=cache,
            start_pos=start_pos,
        )
        controller_mask = (
            torch.ones_like(input_ids, dtype=torch.long)
            if valid_mask is None
            else valid_mask.to(device=input_ids.device, dtype=torch.long)
        )
        logits, stats = self._forward_context(
            context,
            input_ids,
            controller_mask,
            memory_state,
            memory_enabled,
            mode,
            max_steps,
            halt_epsilon,
            mac_seq_len=int(attention_mask.shape[1]),
        )
        return logits, stats, cache

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
        stop_strings: list[str] | None = None,
        token_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event=None,
    ) -> dict[str, Any]:
        """Autoregressive generation with an incremental context KV cache.

        The prompt is evaluated exactly once.  Every subsequent token runs a
        single decode step, so latency no longer grows with the full prompt
        being recomputed at every position.

        ``token_callback`` receives one dict per generated token with the
        incremental UTF-8 text, which lets an API stream without re-decoding
        the whole prefix.  ``stop_strings`` ends generation after the first
        generated token that completes one of the strings.
        """
        if self.training:
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
        prompt_logits, prompt_stats, cache = self.prefill(
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
        total_macs = int(prompt_stats.get("estimated_macs", 0))
        generated: list[int] = []
        gen_depths: list[float] = []
        decode_stats: list[dict[str, Any]] = []
        finish_reason = "length"
        streamed_text = ""
        logits = prompt_logits
        for _ in range(max_new_tokens):
            if cancel_event is not None and cancel_event.is_set():
                finish_reason = "cancelled"
                break
            if input_ids.shape[1] >= self.cfg.max_seq_len:
                finish_reason = "length"
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
                [
                    attention_mask,
                    torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
                ],
                dim=-1,
            )
            current_text = tokenizer.decode(generated, skip_special=True)
            delta = current_text[len(streamed_text) :]
            streamed_text = current_text
            if token_callback is not None:
                try:
                    token_callback(
                        {
                            "id": next_id,
                            "text": delta,
                            "index": len(generated) - 1,
                            "eos": bool(
                                stop_at_eos and next_id == tokenizer.eos_id
                            ),
                        }
                    )
                except Exception:
                    # A disconnected stream must not abort generation.
                    pass
            if stop_at_eos and next_id == tokenizer.eos_id:
                finish_reason = "stop"
                # The EOS token is emitted from the previous step's logits, so
                # it inherits that step's depth rather than the prompt's.
                gen_depths.append(
                    gen_depths[-1] if gen_depths else (depths[-1] if depths else 1.0)
                )
                break
            if stop_strings and any(
                marker and marker in current_text for marker in stop_strings
            ):
                finish_reason = "stop"
                gen_depths.append(
                    gen_depths[-1] if gen_depths else (depths[-1] if depths else 1.0)
                )
                break
            logits, stats, cache = self.decode_step(
                token,
                attention_mask,
                cache,
                memory_state=memory_state,
                mode=mode,
                max_steps=max_steps,
                halt_epsilon=halt_epsilon,
            )
            depth = float(stats["token_depth"][0, 0].detach().cpu().item())
            gen_depths.append(depth)
            total_evaluations += int(stats["total_token_evaluations"])
            total_macs += int(stats.get("estimated_macs", 0))
            decode_stats.append(
                {
                    "depth": round(depth, 4),
                    "token_evaluations": int(
                        stats.get("total_token_evaluations", 0)
                    ),
                }
            )
        elapsed = time.perf_counter() - start
        all_depths = depths + gen_depths
        prompt_text = tokenizer.decode(
            input_ids[0, :prompt_len].tolist(), skip_special=False
        )
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
            "generated_ids": generated,
            "tokens": per_token,
            "prompt_tokens": prompt_len,
            "generated_tokens": len(generated),
            "finish_reason": finish_reason,
            "latency_ms": round(elapsed * 1000, 3),
            "tokens_per_second": (
                round(len(generated) / elapsed, 3) if elapsed > 0 else None
            ),
            "mean_depth": round(sum(all_depths) / max(1, len(all_depths)), 4),
            "max_depth": round(max(all_depths), 4) if all_depths else 0.0,
            "token_evaluations": total_evaluations,
            "estimated_macs": total_macs,
            "estimated_flops": flops_from_macs(total_macs),
            "stats": serializable_stats(prompt_stats),
            "generation_stats": {
                "decode_steps": decode_stats,
                "cache_bytes": cache.memory_bytes(),
                "prompt_token_evaluations": int(
                    prompt_stats["total_token_evaluations"]
                ),
                "prompt_estimated_macs": int(prompt_stats.get("estimated_macs", 0)),
            },
            "mode": mode,
            "memory": prompt_stats.get("memory"),
        }

    # ------------------------------------------------------------------
    def _select_next_ids(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        if temperature is None or temperature <= 0:
            return torch.argmax(logits.float(), dim=-1)
        ids = []
        for row in range(logits.shape[0]):
            scaled = logits[row].float() / max(temperature, 1e-4)
            if top_k and top_k > 0:
                k = min(top_k, scaled.shape[-1])
                values, indices = torch.topk(scaled, k)
                probs = torch.softmax(values, dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                ids.append(int(indices.gather(-1, choice).item()))
            else:
                probs = torch.softmax(scaled, dim=-1)
                ids.append(int(torch.multinomial(probs, num_samples=1).item()))
        return torch.tensor(ids, device=logits.device, dtype=torch.long)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[list[int]],
        tokenizer,
        memory_state: FastMemoryState | None = None,
        mode: str = "adaptive_hard",
        max_steps: int | None = None,
        halt_epsilon: float | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        max_new_tokens: int = 48,
        stop_at_eos: bool = True,
    ) -> dict[str, Any]:
        """Generate a batch with left-padded prompts and row-compacted caches.

        Left padding puts every row's last prompt token at the same column, so
        the prefill logits are directly usable without a per-row gather.  Rows
        that finish are removed from the cache, which avoids computing decode
        steps for padding.
        """
        if not prompts:
            return {
                "generated_ids": [],
                "prompt_tokens": [],
                "generated_tokens": [],
                "token_evaluations": 0,
                "estimated_macs": 0,
                "mean_depth": 0.0,
                "latency_ms": 0.0,
                "batch_size": 0,
            }
        if self.training:
            self.eval()
        device = next(self.parameters()).device
        start = time.perf_counter()
        b = len(prompts)
        pad_id = int(getattr(tokenizer, "pad_id", 0))
        max_prompt = max(1, max(len(p) for p in prompts))
        max_prompt = min(max_prompt, self.cfg.max_seq_len)
        input_ids = torch.full(
            (b, max_prompt), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (b, max_prompt), dtype=torch.long, device=device
        )
        positions = torch.zeros(
            (b, max_prompt), dtype=torch.long, device=device
        )
        prompt_lengths: list[int] = []
        for row, prompt in enumerate(prompts):
            prompt = list(prompt)[: self.cfg.max_seq_len]
            if not prompt:
                prompt = [int(getattr(tokenizer, "bos_id", 1))]
            n = len(prompt)
            prompt_lengths.append(n)
            input_ids[row, max_prompt - n :] = torch.tensor(
                prompt, dtype=torch.long, device=device
            )
            attention_mask[row, max_prompt - n :] = 1
            positions[row, max_prompt - n :] = torch.arange(
                n, dtype=torch.long, device=device
            )

        prompt_logits, prompt_stats, cache = self.prefill(
            input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
            mode=mode,
            max_steps=max_steps,
            halt_epsilon=halt_epsilon,
            positions=positions,
        )
        prompt_depth = prompt_stats["token_depth"].detach()
        prompt_valid = attention_mask.bool()
        depth_sum = float(prompt_depth[prompt_valid].sum().item())
        depth_count = int(prompt_valid.sum().item())
        prompt_depths_by_row = [
            [float(x) for x in prompt_depth[row][prompt_valid[row]].tolist()]
            for row in range(b)
        ]
        gen_depths_by_row: list[list[float]] = [[] for _ in range(b)]
        last_depth_by_row = [
            values[-1] if values else 1.0 for values in prompt_depths_by_row
        ]
        total_evaluations = int(prompt_stats["total_token_evaluations"])
        total_macs = int(prompt_stats.get("estimated_macs", 0))

        generated: list[list[int]] = [[] for _ in range(b)]
        finished = [n >= self.cfg.max_seq_len for n in prompt_lengths]
        row_indices = list(range(b))
        logits = prompt_logits
        current_mask = attention_mask
        for _ in range(max(0, int(max_new_tokens))):
            if not row_indices:
                break
            next_ids = self._select_next_ids(
                logits[:, -1, :], temperature, top_k
            )
            decode_rows: list[int] = []
            for local, original in enumerate(row_indices):
                if finished[original]:
                    continue
                token_id = int(next_ids[local].item())
                generated[original].append(token_id)
                if stop_at_eos and token_id == tokenizer.eos_id:
                    finished[original] = True
                    gen_depths_by_row[original].append(
                        last_depth_by_row[original]
                    )
                    depth_sum += last_depth_by_row[original]
                    depth_count += 1
                    continue
                if (
                    prompt_lengths[original] + len(generated[original])
                    >= self.cfg.max_seq_len
                ):
                    finished[original] = True
                    gen_depths_by_row[original].append(
                        last_depth_by_row[original]
                    )
                    depth_sum += last_depth_by_row[original]
                    depth_count += 1
                    continue
                decode_rows.append(local)
            if not decode_rows:
                break
            keep = torch.tensor(decode_rows, dtype=torch.long, device=device)
            cache = cache.select_rows(keep)
            current_mask = current_mask.index_select(0, keep)
            next_ids = next_ids.index_select(0, keep)
            row_indices = [row_indices[i] for i in decode_rows]
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        (len(row_indices), 1),
                        dtype=current_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            decode_positions = torch.tensor(
                [
                    [
                        prompt_lengths[original]
                        + len(generated[original])
                        - 1
                    ]
                    for original in row_indices
                ],
                dtype=torch.long,
                device=device,
            )
            logits, stats, cache = self.decode_step(
                next_ids.unsqueeze(1),
                current_mask,
                cache,
                memory_state=memory_state,
                mode=mode,
                max_steps=max_steps,
                halt_epsilon=halt_epsilon,
                positions=decode_positions,
            )
            total_evaluations += int(stats.get("total_token_evaluations", 0))
            total_macs += int(stats.get("estimated_macs", 0))
            step_depth = stats["token_depth"].detach()
            depth_sum += float(step_depth.sum().item())
            depth_count += int(step_depth.numel())
            for local, original in enumerate(row_indices):
                value = float(step_depth[local, 0].item())
                gen_depths_by_row[original].append(value)
                last_depth_by_row[original] = value
        row_mean_depths = [
            (
                sum(prompt_depths_by_row[row] + gen_depths_by_row[row])
                / max(
                    1,
                    len(prompt_depths_by_row[row]) + len(gen_depths_by_row[row]),
                )
            )
            for row in range(b)
        ]
        elapsed = time.perf_counter() - start
        return {
            "generated_ids": generated,
            "prompt_tokens": prompt_lengths,
            "generated_tokens": [len(g) for g in generated],
            "token_evaluations": total_evaluations,
            "estimated_macs": total_macs,
            "estimated_flops": flops_from_macs(total_macs),
            "mean_depth": round(depth_sum / max(1, depth_count), 6),
            "row_mean_depths": row_mean_depths,
            "depth_count": depth_count,
            "latency_ms": round(elapsed * 1000, 3),
            "batch_size": b,
            "cache_bytes": cache.memory_bytes(),
            "stats": serializable_stats(prompt_stats),
            "mode": mode,
            "memory": prompt_stats.get("memory"),
        }

    # ------------------------------------------------------------------
    def model_summary(self) -> dict[str, Any]:
        total, trainable = self.parameter_counts()
        return {
            "config": self.cfg.model_dump(),
            "parameters_total": total,
            "parameters_trainable": trainable,
            "parameters_frozen": total - trainable,
            "memory_initial_parameters": self.memory.init_state().parameter_count(),
        }
