"""Shared application services: model cache, memory registry, job manager."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from backend.config import AppConfig, ModelConfig, TrainingConfig
from backend.engine.metrics import count_parameters, module_breakdown
from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer
from backend.memory.document_ingestion import DocumentIngestion, Document, load_document
from backend.memory.fast_memory import FastWeightMemory
from backend.memory.memory_manager import MemoryManager, MemorySession
from backend.memory.memory_retriever import MemoryRetriever
from backend.server.db import Database
from backend.server.jobs import JobManager
from backend.training.checkpoints import (
    list_checkpoints,
    load_model_from_checkpoint,
    resolve_checkpoint,
)
from backend.training.trainer import resolve_device


class ModelService:
    """Small LRU cache of loaded checkpoints.

    A checkpoint is a complete model definition, so switching checkpoints is a
    real model switch, not a cosmetic label.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.server.device)
        self.tokenizer = ByteTokenizer()
        self._cache: OrderedDict[str, tuple[WILLModel, dict[str, Any]]] = OrderedDict()
        self._lock = threading.RLock()
        metas = list_checkpoints(cfg.server.checkpoint_dir)
        self.default_checkpoint_id: str | None = (
            metas[0]["checkpoint_id"] if metas else None
        )

    def _fresh(self) -> tuple[WILLModel, dict[str, Any]]:
        model = WILLModel(self.cfg.model)
        model.to(self.device)
        model.eval()
        payload = {
            "format": "will-fresh-model",
            "model_config": self.cfg.model.model_dump(),
            "training_config": TrainingConfig().model_dump(),
            "metrics": {},
            "task": "arith_add",
            "checkpoint_id": None,
        }
        return model, payload

    def get(
        self,
        checkpoint_id: str | None = None,
        allow_fresh: bool = True,
    ) -> tuple[WILLModel, dict[str, Any], str | None]:
        cid = checkpoint_id or self.default_checkpoint_id
        if cid is None:
            if not allow_fresh:
                raise FileNotFoundError("no checkpoint is available")
            model, payload = self._fresh()
            return model, payload, None
        with self._lock:
            if cid in self._cache:
                self._cache.move_to_end(cid)
                model, payload = self._cache[cid]
                return model, payload, cid
        path = resolve_checkpoint(self.cfg.server.checkpoint_dir, cid)
        model, _model_cfg, payload = load_model_from_checkpoint(path, device=self.device)
        with self._lock:
            self._cache[cid] = (model, payload)
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        return model, payload, cid

    def set_default(self, checkpoint_id: str) -> None:
        resolve_checkpoint(self.cfg.server.checkpoint_dir, checkpoint_id)
        self.default_checkpoint_id = checkpoint_id

    def unload(self, checkpoint_id: str | None = None) -> None:
        with self._lock:
            if checkpoint_id is None:
                self._cache.clear()
            else:
                self._cache.pop(checkpoint_id, None)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        return list_checkpoints(self.cfg.server.checkpoint_dir)

    def summary(self, checkpoint_id: str | None = None) -> dict[str, Any]:
        model, payload, cid = self.get(checkpoint_id, allow_fresh=True)
        breakdown = module_breakdown(model)
        # Keep the payload small: direct module rows only.
        top = sorted(breakdown, key=lambda r: r["parameters"], reverse=True)[:12]
        return {
            "checkpoint_id": cid,
            "default_checkpoint_id": self.default_checkpoint_id,
            "device": str(self.device),
            "model_config": model.cfg.model_dump(),
            "training_config": payload.get("training_config"),
            "task": payload.get("task"),
            "metrics": payload.get("metrics", {}),
            "parameters_total": count_parameters(model),
            "parameters_trainable": count_parameters(model, trainable_only=True),
            "parameters_frozen": count_parameters(model)
            - count_parameters(model, trainable_only=True),
            "memory_initial_parameters": model.memory.init_state().parameter_count(),
            "module_breakdown": top,
        }


class MemoryService:
    """Session registry bound to the currently selected checkpoint.

    Session state lives on disk.  Switching to a checkpoint with a different
    memory width rebuilds the in-process registry; incompatible sessions remain
    on disk and are rejected with a clear error rather than silently decoded.
    """

    def __init__(self, cfg: AppConfig, models: ModelService) -> None:
        self.cfg = cfg
        self.models = models
        self._lock = threading.RLock()
        self._cid: str | None = None
        self._manager: MemoryManager | None = None
        self._ingestion: DocumentIngestion | None = None
        self._retriever: MemoryRetriever | None = None

    def _ensure(self) -> None:
        cid = self.models.default_checkpoint_id
        with self._lock:
            if self._manager is not None and self._cid == cid:
                return
            model, _payload, _cid = self.models.get(cid, allow_fresh=True)
            device = self.models.device
            self._manager = MemoryManager(
                FastWeightMemory(model.cfg),
                self.cfg.memory,
                self.cfg.server.memory_dir,
                device=device,
            )
            self._ingestion = DocumentIngestion(
                model, self.models.tokenizer, self.cfg.memory, device=device
            )
            self._retriever = MemoryRetriever(
                model, self.models.tokenizer, self.cfg.memory, device=device
            )
            self._cid = cid

    @property
    def manager(self) -> MemoryManager:
        self._ensure()
        assert self._manager is not None
        return self._manager

    @property
    def ingestion(self) -> DocumentIngestion:
        self._ensure()
        assert self._ingestion is not None
        return self._ingestion

    @property
    def retriever(self) -> MemoryRetriever:
        self._ensure()
        assert self._retriever is not None
        return self._retriever

    def refresh(self) -> None:
        with self._lock:
            self._manager = None
            self._ingestion = None
            self._retriever = None
            self._cid = None

    # ------------------------------------------------------------------
    def create_session(self, name: str, metadata: dict | None = None) -> dict[str, Any]:
        meta = dict(metadata or {})
        meta.setdefault("model_checkpoint", self.models.default_checkpoint_id)
        session = self.manager.create_session(name, meta)
        return self.manager.summary(session)

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.manager.list_sessions()

    def get_session(self, session_id: str) -> MemorySession:
        return self.manager.get_session(session_id)

    def summary(self, session_id: str) -> dict[str, Any]:
        return self.manager.summary(self.get_session(session_id))

    def delete_session(self, session_id: str) -> None:
        self.manager.delete_session(session_id)

    def reset_session(self, session_id: str) -> dict[str, Any]:
        return self.manager.summary(self.manager.reset_session(session_id))

    def ingest_text(
        self, session_id: str, text: str, filename: str = "inline.txt"
    ) -> dict[str, Any]:
        result = self.ingestion.ingest_text(
            session_id, text, self.manager, filename=filename
        )
        return result

    def ingest_document(self, session_id: str, document: Document) -> dict[str, Any]:
        return self.ingestion.ingest(session_id, document, self.manager)

    def load_document(self, path: str | Path, filename: str | None = None) -> Document:
        return load_document(path, filename)

    def query(
        self,
        session_id: str,
        question: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        use_memory_in_generation: bool = True,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        result = self.retriever.answer(
            session,
            question,
            mode=mode,
            top_k=top_k,
            use_memory_in_generation=use_memory_in_generation,
        )
        self.manager.record_query(session_id, result)
        return result

    def search(
        self, session_id: str, question: str, mode: str = "hybrid", top_k: int | None = None
    ) -> list[dict[str, Any]]:
        session = self.get_session(session_id)
        rows = self.retriever.search(session, question, mode=mode, top_k=top_k)
        return [self._clean_row(row) for row in rows]

    @staticmethod
    def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in row.items()
            if key not in ("value_vec", "key_vec")
        }

    def adapt_rounds(self, session_id: str, rounds: int = 1) -> dict[str, Any]:
        session = self.get_session(session_id)
        keys = []
        values = []
        for chunk in session.chunks:
            for rec in chunk.get("records", []):
                key_vec = rec.get("key_vec")
                value_vec = rec.get("value_vec")
                if isinstance(key_vec, torch.Tensor) and isinstance(value_vec, torch.Tensor):
                    keys.append(key_vec.float())
                    values.append(value_vec.float())
        if not keys:
            return {
                "session_id": session_id,
                "rounds": 0,
                "updates": 0,
                "message": "no record pairs are available in this session",
            }
        k = torch.stack(keys).to(self.models.device)
        v = torch.stack(values).to(self.models.device)
        before = session.state.updates
        for _ in range(max(1, min(int(rounds), 200))):
            self.manager.adapt_pairs(session_id, k, v, persist=False)
        self.manager.save_session(session_id)
        session = self.get_session(session_id)
        return {
            "session_id": session_id,
            "rounds": rounds,
            "updates": session.state.updates - before,
            "total_updates": session.state.updates,
            "last_loss": session.state.last_loss,
            "adapt_seconds": round(session.adapt_seconds, 4),
            "state_bytes": session.state_bytes(),
        }

    def delete_document(self, session_id: str, doc_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        before = len(session.chunks)
        session.documents = [d for d in session.documents if d.get("doc_id") != doc_id]
        session.chunks = [c for c in session.chunks if c.get("doc_id") != doc_id]
        session.rebuild_index()
        self.manager.save_session(session_id)
        summary = self.manager.summary(session)
        summary["removed_chunks"] = before - len(session.chunks)
        summary["note"] = (
            "Retrieval chunks were removed. Fast weights are not unlearned by "
            "deleting chunks; reset the session to clear the adapted state."
        )
        return summary


@dataclass
class Services:
    cfg: AppConfig
    db: Database
    jobs: JobManager
    models: ModelService
    memory: MemoryService

    @classmethod
    def create(cls, cfg: AppConfig, max_workers: int = 1) -> "Services":
        db = Database(cfg.server.db_path)
        db.fail_stale_jobs()
        jobs = JobManager(db, max_workers=max_workers)
        models = ModelService(cfg)
        memory = MemoryService(cfg, models)
        return cls(cfg=cfg, db=db, jobs=jobs, models=models, memory=memory)

    def shutdown(self) -> None:
        self.jobs.shutdown()
