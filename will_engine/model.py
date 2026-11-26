"""Installable inference engine and the ``WILLForCausalLM`` facade."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import torch

from backend.config import AppConfig, load_config
from backend.engine.metrics import (
    count_parameters,
    flops_from_macs,
    hardware_info,
    process_rss_mb,
)
from backend.engine.model import WILLModel
from backend.engine.tokenizer import BOS_ID, SEP_ID, ByteTokenizer, build_tokenizer
from backend.memory.document_ingestion import DocumentIngestion
from backend.memory.fast_memory import FastWeightMemory
from backend.memory.memory_manager import MemoryManager
from backend.memory.memory_retriever import MemoryRetriever
from backend.training.checkpoints import (
    list_checkpoints,
    load_model_from_checkpoint,
    resolve_checkpoint,
)
from backend.training.trainer import resolve_device
from will_engine.chat import format_chat_prompt
from will_engine.telemetry import GLOBAL_TELEMETRY
from will_engine.types import GenerationResult


class WILLForCausalLM:
    """A checkpoint-backed causal LM with session-local memory and telemetry.

    Example::

        from will_engine import WILLForCausalLM

        engine = WILLForCausalLM("add-single-act-final")
        result = engine.generate("calc 2 + 3 =", max_new_tokens=4)
        print(result.text)
    """

    def __init__(
        self,
        checkpoint_id: str | None = None,
        *,
        checkpoint_dir: str | Path | None = None,
        device: str = "auto",
        config: AppConfig | None = None,
        memory_dir: str | Path | None = None,
        model: WILLModel | None = None,
        tokenizer: ByteTokenizer | None = None,
        payload: dict[str, Any] | None = None,
        memory_manager: MemoryManager | None = None,
        retriever: MemoryRetriever | None = None,
    ) -> None:
        cfg = (config or load_config()).model_copy(deep=True)
        if checkpoint_dir is not None:
            cfg.server.checkpoint_dir = str(checkpoint_dir)
        if memory_dir is not None:
            cfg.server.memory_dir = str(memory_dir)
        self.cfg = cfg
        self.tokenizer = tokenizer or build_tokenizer(cfg.model)
        self.payload = payload or {}
        self._lock = threading.RLock()

        if model is None:
            resolved = checkpoint_id or self._default_checkpoint_id()
            if resolved:
                path = resolve_checkpoint(self.cfg.server.checkpoint_dir, resolved)
                model, _model_cfg, loaded_payload = load_model_from_checkpoint(
                    path, device=resolve_device(device)
                )
                self.payload = loaded_payload
                if tokenizer is None:
                    from backend.training.checkpoints import (
                        load_tokenizer_for_checkpoint,
                    )

                    self.tokenizer = load_tokenizer_for_checkpoint(loaded_payload)
                self.checkpoint_id = resolved
            else:
                model = WILLModel(self.cfg.model)
                model.eval()
                self.checkpoint_id = None
        else:
            self.checkpoint_id = checkpoint_id

        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.memory_manager = memory_manager or MemoryManager(
            FastWeightMemory(self.model.cfg),
            self.cfg.memory,
            self.cfg.server.memory_dir,
            device=self.device,
        )
        memory_cfg = getattr(self.memory_manager, "cfg", self.cfg.memory)
        self.ingestion = DocumentIngestion(
            self.model, self.tokenizer, memory_cfg, device=self.device
        )
        self.retriever = retriever or MemoryRetriever(
            self.model, self.tokenizer, memory_cfg, device=self.device
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_model(
        cls,
        model: WILLModel,
        tokenizer: ByteTokenizer,
        *,
        payload: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
        config: AppConfig | None = None,
        memory_manager: MemoryManager | None = None,
        retriever: MemoryRetriever | None = None,
    ) -> "WILLForCausalLM":
        """Wrap an already-loaded model (used by the HTTP service cache)."""
        return cls(
            checkpoint_id=checkpoint_id,
            config=config,
            model=model,
            tokenizer=tokenizer,
            payload=payload,
            memory_manager=memory_manager,
            retriever=retriever,
        )

    def _default_checkpoint_id(self) -> str | None:
        checkpoints = list_checkpoints(self.cfg.server.checkpoint_dir)
        return checkpoints[0]["checkpoint_id"] if checkpoints else None

    def _prompt_ids(self, prompt: str, append_sep: bool = True) -> list[int]:
        body = self.tokenizer.encode(prompt)
        budget = max(1, self.model.cfg.max_seq_len - 1 - (1 if append_sep else 0))
        body = body[-budget:]
        ids = [BOS_ID] + body
        if append_sep:
            ids.append(SEP_ID)
        return ids

    def _assert_session_compatible(self, session) -> None:
        stored = session.metadata.get("checkpoint_id") or session.metadata.get(
            "model_checkpoint"
        )
        if stored and self.checkpoint_id and stored != self.checkpoint_id:
            raise ValueError(
                f"memory session belongs to checkpoint {stored!r}, not "
                f"{self.checkpoint_id!r}"
            )

    def _memory_state(self, session_id: str | None):
        if not session_id:
            return None
        session = self.memory_manager.get_session(session_id)
        self._assert_session_compatible(session)
        state = session.state
        widths = {
            int(tensor.shape[-1])
            for tensor in state.params.values()
            if tensor.ndim >= 1
        }
        expected = int(self.model.cfg.memory_dim)
        if widths and expected not in widths:
            raise ValueError(
                "memory session width does not match this checkpoint "
                f"(session {sorted(widths)}, model {expected})"
            )
        return state

    @staticmethod
    def _normalise_stop(stop: str | list[str] | None) -> list[str]:
        if stop is None:
            return []
        values = [stop] if isinstance(stop, str) else list(stop)
        return [value for value in values if value]

    @staticmethod
    def _trim_at_stop(text: str, stops: list[str]) -> str:
        cut = len(text)
        for marker in stops:
            index = text.find(marker)
            if index >= 0:
                cut = min(cut, index)
        return text[:cut]

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        mode: str = "adaptive_hard",
        max_steps: int | None = None,
        halt_epsilon: float | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        memory_session_id: str | None = None,
        stop: str | list[str] | None = None,
        append_sep: bool = True,
        token_callback=None,
        _streamed: bool = False,
        _cancel_event=None,
    ) -> GenerationResult:
        if not prompt:
            raise ValueError("prompt must not be empty")
        stops = self._normalise_stop(stop)
        prompt_ids = self._prompt_ids(prompt, append_sep=append_sep)
        memory_state = self._memory_state(memory_session_id)
        tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        start = time.perf_counter()
        try:
            out = self.model.generate(
                tensor,
                self.tokenizer,
                memory_state=memory_state,
                mode=mode,
                max_steps=max_steps,
                halt_epsilon=halt_epsilon,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                stop_at_eos=True,
                stop_strings=stops or None,
                token_callback=token_callback,
                cancel_event=_cancel_event,
            )
        except Exception as exc:
            GLOBAL_TELEMETRY.record_error(f"{type(exc).__name__}: {exc}")
            raise
        elapsed = time.perf_counter() - start
        text = self._trim_at_stop(str(out.get("text") or ""), stops)
        finish_reason = str(out.get("finish_reason") or "stop")
        if text != out.get("text"):
            finish_reason = "stop"
        usage = {
            "prompt_tokens": int(out.get("prompt_tokens", len(prompt_ids))),
            "completion_tokens": int(out.get("generated_tokens", 0)),
            "total_tokens": int(out.get("prompt_tokens", len(prompt_ids)))
            + int(out.get("generated_tokens", 0)),
        }
        GLOBAL_TELEMETRY.record_generation(
            checkpoint_id=self.checkpoint_id,
            mode=mode,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            token_evaluations=int(out.get("token_evaluations", 0)),
            estimated_flops=int(out.get("estimated_flops", 0)),
            seconds=elapsed,
            streamed=_streamed,
        )
        return GenerationResult(
            text=text,
            generated_ids=list(out.get("generated_ids") or []),
            prompt_tokens=usage["prompt_tokens"],
            generated_tokens=usage["completion_tokens"],
            finish_reason=finish_reason,
            latency_ms=round(elapsed * 1000, 3),
            mean_depth=float(out.get("mean_depth", 0.0)),
            token_evaluations=int(out.get("token_evaluations", 0)),
            estimated_flops=int(out.get("estimated_flops", 0)),
            mode=mode,
            checkpoint_id=self.checkpoint_id,
            memory_session_id=memory_session_id,
            usage=usage,
            raw=out,
        )

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield start/token/done/error events as tokens are generated."""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        cancel_event = threading.Event()
        prompt_tokens = len(
            self._prompt_ids(prompt, append_sep=kwargs.get("append_sep", True))
        )
        mode = kwargs.get("mode", "adaptive_hard")

        def on_token(token: dict[str, Any]) -> None:
            events.put({"type": "token", **token})

        def run() -> None:
            try:
                result = self.generate(
                    prompt,
                    token_callback=on_token,
                    _streamed=True,
                    _cancel_event=cancel_event,
                    **kwargs,
                )
                events.put({"type": "done", "result": result.to_dict()})
            except Exception as exc:
                events.put(
                    {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
                )
            finally:
                events.put(None)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        yield {
            "type": "start",
            "checkpoint_id": self.checkpoint_id,
            "mode": mode,
            "prompt_tokens": prompt_tokens,
        }
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                yield event
        finally:
            # Closing the SSE response early must stop the worker instead of
            # letting it generate the full sequence into an abandoned queue.
            cancel_event.set()
            thread.join(timeout=1.0)

    def chat(
        self,
        messages: list[Any],
        *,
        template: str = "plain",
        **kwargs: Any,
    ) -> GenerationResult:
        prompt = format_chat_prompt(messages, template=template)
        return self.generate(prompt, **kwargs)

    def stream_chat(
        self,
        messages: list[Any],
        *,
        template: str = "plain",
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        prompt = format_chat_prompt(messages, template=template)
        yield from self.stream(prompt, **kwargs)

    # ------------------------------------------------------------------
    def absorb(
        self,
        text: str,
        *,
        session_id: str | None = None,
        name: str = "api-memory",
        filename: str = "inline.txt",
        consolidate: bool = False,
    ) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        created = False
        if session_id is None:
            session = self.memory_manager.create_session(
                name, metadata={"checkpoint_id": self.checkpoint_id}
            )
            session_id = session.session_id
            created = True
        else:
            self._assert_session_compatible(
                self.memory_manager.get_session(session_id)
            )
        result = self.ingestion.ingest_text(
            session_id, text, self.memory_manager, filename=filename
        )
        consolidation = None
        unstructured = None
        if getattr(self.model.cfg, "tokenizer_type", "char") == "bpe":
            # v4: absorb raw prose directly into bounded sequential fast
            # weights, in addition to the legacy chunk/BM25 ingestion above.
            from backend.memory.slot_memory import absorb_documents

            session = self.memory_manager.get_session(session_id)
            session.state = absorb_documents(
                self.model,
                self.tokenizer,
                [text],
                state=session.state,
            )
            self.memory_manager.save_session(session_id)
            unstructured = {
                "state_bytes": int(session.state.state_bytes()),
                "slots": int(self.model.cfg.num_slots or self.model.cfg.memory_slots),
            }
        if consolidate:
            consolidation = self.memory_manager.consolidate_session(session_id)
        session = self.memory_manager.get_session(session_id)
        GLOBAL_TELEMETRY.record_absorb(
            chars=len(text),
            chunks=int(result.get("chunks_added", 0)),
            updates=int(result.get("updates", 0)),
        )
        return {
            "object": "will.memory.absorb",
            "session_id": session_id,
            "session_created": created,
            "name": session.name,
            "checkpoint_id": self.checkpoint_id,
            "chunks_added": int(result.get("chunks_added", 0)),
            "total_chunks": int(result.get("total_chunks", 0)),
            "pairs_seen": int(result.get("pairs_seen", 0)),
            "updates": int(result.get("updates", 0)),
            "total_updates": int(result.get("total_updates", 0)),
            "state_bytes": int(result.get("state_bytes", 0)),
            "adapt_seconds": result.get("adapt_seconds"),
            "consolidation": consolidation,
            "unstructured": unstructured,
        }

    def query(
        self,
        session_id: str,
        question: str,
        *,
        mode: str = "hybrid",
        top_k: int | None = None,
    ) -> dict[str, Any]:
        session = self.memory_manager.get_session(session_id)
        self._assert_session_compatible(session)
        result = self.retriever.answer(
            session, question, mode=mode, top_k=top_k
        )
        self.memory_manager.record_query(session_id, result)
        return result

    # ------------------------------------------------------------------
    def model_summary(self) -> dict[str, Any]:
        training = self.payload.get("training_config") or {}
        try:
            fast_parameters = self.model.memory.init_state(
                torch.device("cpu")
            ).parameter_count()
        except Exception:
            fast_parameters = None
        return {
            "checkpoint_id": self.checkpoint_id,
            "task": self.payload.get("task") or training.get("task"),
            "parameters": count_parameters(self.model),
            "device": str(self.device),
            "max_seq_len": self.model.cfg.max_seq_len,
            "max_steps": self.model.cfg.max_steps,
            "min_steps": self.model.cfg.min_steps,
            "adaptive_inference_steps": self.model.cfg.resolved_adaptive_steps(),
            "memory_dim": self.model.cfg.memory_dim,
            "memory_kind": self.model.cfg.memory_kind,
            "fast_parameters": fast_parameters,
        }

    def telemetry(self, include_hardware: bool = True) -> dict[str, Any]:
        snapshot = GLOBAL_TELEMETRY.snapshot()
        snapshot["model"] = self.model_summary()
        snapshot["process_rss_mb"] = round(process_rss_mb(), 1)
        snapshot["memory_sessions"] = len(self.memory_manager.list_sessions())
        if include_hardware:
            snapshot["hardware"] = hardware_info()
        return snapshot


def estimate_flops(macs: int) -> int:
    """Convenience wrapper kept public for SDK users."""
    return flops_from_macs(int(macs))
