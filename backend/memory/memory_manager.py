"""Session-isolated memory registry, persistence and inspection."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from backend.config import MemoryConfig
from backend.memory.bm25 import BM25Index
from backend.memory.fast_memory import (
    STATE_VERSION,
    FastMemoryState,
    FastWeightMemory,
)

SESSION_RE = re.compile(r"^[a-f0-9]{32}$")
SESSION_FORMAT_VERSION = 2


def validate_session_id(session_id: str) -> str:
    if not SESSION_RE.match(session_id):
        raise ValueError("invalid memory session id")
    return session_id


@dataclass
class MemorySession:
    session_id: str
    name: str
    created_at: float
    state: FastMemoryState
    documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    updates: int = 0
    adapt_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    last_query: dict[str, Any] | None = None
    retrieval_counts: dict[str, int] = field(default_factory=lambda: {"lexical": 0, "neural": 0, "hybrid": 0})

    def rebuild_index(self) -> None:
        self.metadata["_bm25"] = BM25Index().fit([c["text"] for c in self.chunks])

    @property
    def bm25(self) -> BM25Index | None:
        return self.metadata.get("_bm25")

    def chunk_state_bytes(self) -> int:
        total = 0
        for c in self.chunks:
            for key in ("value_vec", "key_vec"):
                v = c.get(key)
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
            for r in c.get("records", []):
                for key in ("value_vec", "key_vec"):
                    v = r.get(key)
                    if isinstance(v, torch.Tensor):
                        total += v.numel() * v.element_size()
        return total

    def record_count(self) -> int:
        return int(sum(len(c.get("records", [])) for c in self.chunks))

    def state_bytes(self) -> int:
        return self.state.state_bytes() + self.chunk_state_bytes()

    def parameter_count(self) -> int:
        return self.state.parameter_count()

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": SESSION_FORMAT_VERSION,
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "state": {
                "version": self.state.version,
                "kind": self.state.kind,
                "params": {k: v.detach().cpu() for k, v in self.state.params.items()},
                "updates": self.state.updates,
                "adapt_seconds": self.state.adapt_seconds,
                "last_loss": self.state.last_loss,
                "metadata": self.state.metadata,
            },
            "documents": self.documents,
            "chunks": self.chunks,
            "updates": self.updates,
            "adapt_seconds": self.adapt_seconds,
            "metadata": {k: v for k, v in self.metadata.items() if k != "_bm25"},
            "last_query": self.last_query,
            "retrieval_counts": self.retrieval_counts,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemorySession":
        format_version = int(payload.get("format_version", 1))
        if format_version > SESSION_FORMAT_VERSION:
            raise ValueError(
                f"memory session format {format_version} is newer than supported "
                f"format {SESSION_FORMAT_VERSION}"
            )
        st = payload["state"]
        state_version = int(st.get("version", 1))
        if state_version > STATE_VERSION:
            raise ValueError(
                f"fast-memory state version {state_version} is newer than "
                f"supported version {STATE_VERSION}"
            )
        state = FastMemoryState(
            kind=st["kind"],
            params={k: v for k, v in st["params"].items()},
            updates=int(st.get("updates", 0)),
            adapt_seconds=float(st.get("adapt_seconds", 0.0)),
            last_loss=st.get("last_loss"),
            metadata=st.get("metadata", {}),
            version=STATE_VERSION,
        )
        session = cls(
            session_id=payload["session_id"],
            name=payload.get("name", "session"),
            created_at=float(payload.get("created_at", time.time())),
            state=state,
            documents=payload.get("documents", []),
            chunks=payload.get("chunks", []),
            updates=int(payload.get("updates", 0)),
            adapt_seconds=float(payload.get("adapt_seconds", 0.0)),
            metadata=payload.get("metadata", {}),
            last_query=payload.get("last_query"),
            retrieval_counts=payload.get("retrieval_counts", {"lexical": 0, "neural": 0, "hybrid": 0}),
        )
        session.rebuild_index()
        return session


class MemoryManager:
    def __init__(
        self,
        fast_memory: FastWeightMemory,
        cfg: MemoryConfig,
        storage_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.fast_memory = fast_memory
        self.cfg = cfg
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self._sessions: dict[str, MemorySession] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def _meta_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{validate_session_id(session_id)}.json"

    def _state_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{validate_session_id(session_id)}.pt"

    def create_session(self, name: str, metadata: dict | None = None) -> MemorySession:
        with self._lock:
            sid = uuid.uuid4().hex
            state = self.fast_memory.init_state(self.device)
            session = MemorySession(
                session_id=sid,
                name=(name or "untitled")[:120],
                created_at=time.time(),
                state=state,
                metadata=metadata or {},
            )
            self._sessions[sid] = session
            self.save_session(sid)
            return session

    def get_session(self, session_id: str) -> MemorySession:
        validate_session_id(session_id)
        with self._lock:
            if session_id not in self._sessions:
                self._restore(session_id)
            return self._sessions[session_id]

    def _restore(self, session_id: str) -> None:
        path = self._state_path(session_id)
        if not path.exists():
            raise KeyError("memory session not found")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        session = MemorySession.from_payload(payload)
        session.state.to(self.device)
        self._sessions[session_id] = session

    def reload_session(self, session_id: str) -> MemorySession:
        """Drop the in-process copy and restore it from disk."""
        validate_session_id(session_id)
        with self._lock:
            self._sessions.pop(session_id, None)
            return self.get_session(session_id)

    def save_session(self, session_id: str) -> None:
        validate_session_id(session_id)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("memory session not loaded")
            torch.save(session.to_payload(), self._state_path(session_id))
            self._meta_path(session_id).write_text(
                json.dumps(self.summary(session), indent=2), encoding="utf-8"
            )

    def delete_session(self, session_id: str) -> None:
        validate_session_id(session_id)
        with self._lock:
            self._sessions.pop(session_id, None)
            for p in (self._state_path(session_id), self._meta_path(session_id)):
                if p.exists():
                    p.unlink()

    def reset_session(self, session_id: str) -> MemorySession:
        with self._lock:
            session = self.get_session(session_id)
            session.state = self.fast_memory.init_state(self.device)
            session.documents = []
            session.chunks = []
            session.updates = 0
            session.adapt_seconds = 0.0
            session.last_query = None
            session.retrieval_counts = {"lexical": 0, "neural": 0, "hybrid": 0}
            session.rebuild_index()
            self.save_session(session_id)
            return session

    # ------------------------------------------------------------------
    def adapt_pairs(
        self,
        session_id: str,
        keys: torch.Tensor,
        values: torch.Tensor,
        persist: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            session = self.get_session(session_id)
            if keys.numel() == 0:
                return {"updates": 0, "loss": None}
            keys = keys.to(self.device).detach()
            values = values.to(self.device).detach()
            before = session.state.updates
            start = time.perf_counter()
            self.fast_memory.adapt(session.state, keys, values)
            elapsed = time.perf_counter() - start
            session.updates += 1
            session.adapt_seconds += elapsed
            if persist:
                self.save_session(session_id)
            return {
                "updates": session.state.updates - before,
                "total_updates": session.state.updates,
                "loss": session.state.last_loss,
                "adapt_seconds": elapsed,
            }

    def _pair_vectors(
        self, session: MemorySession
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        seen: set[tuple[str, str]] = set()
        for chunk in session.chunks:
            for rec in chunk.get("records", []):
                marker = (
                    str(rec.get("key_text", "")),
                    str(rec.get("value_text", "")),
                )
                if marker in seen:
                    continue
                key_vec = rec.get("key_vec")
                value_vec = rec.get("value_vec")
                if isinstance(key_vec, torch.Tensor) and isinstance(
                    value_vec, torch.Tensor
                ):
                    seen.add(marker)
                    rows.append((key_vec.float(), value_vec.float()))
        if not rows:
            return None
        return (
            torch.stack([row[0] for row in rows]).to(self.device),
            torch.stack([row[1] for row in rows]).to(self.device),
        )

    def consolidate_session(
        self,
        session_id: str,
        ridge: float | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Replay stored key/value pairs into the bounded fast state.

        For TTT-Linear this is a closed-form ridge solve, which is the
        minimum-norm least-squares state for the replay set.  The session
        source records remain available for citations; only the bounded fast
        state is rewritten.
        """
        with self._lock:
            session = self.get_session(session_id)
            pairs = self._pair_vectors(session)
            if pairs is None:
                return {
                    "session_id": session_id,
                    "pairs": 0,
                    "consolidations": int(
                        session.state.metadata.get("consolidations", 0)
                    ),
                    "weight_norm": session.state.weight_norm(),
                    "loss": None,
                    "seconds": 0.0,
                }
            keys, values = pairs
            before_norm = session.state.weight_norm()
            start = time.perf_counter()
            self.fast_memory.consolidate(
                session.state, keys, values, ridge=ridge
            )
            elapsed = time.perf_counter() - start
            after_norm = session.state.weight_norm()
            session.metadata["last_consolidation"] = {
                "pairs": int(keys.shape[0]),
                "ridge": (
                    float(ridge)
                    if ridge is not None
                    else float(getattr(self.fast_memory.cfg, "memory_ridge", 0.0))
                ),
                "weight_norm_before": round(before_norm, 6),
                "weight_norm_after": round(after_norm, 6),
                "loss": session.state.last_loss,
                "seconds": round(elapsed, 4),
                "timestamp": time.time(),
            }
            if persist:
                self.save_session(session_id)
            return {
                "session_id": session_id,
                "pairs": int(keys.shape[0]),
                "consolidations": int(
                    session.state.metadata.get("consolidations", 0)
                ),
                "weight_norm_before": round(before_norm, 6),
                "weight_norm_after": round(after_norm, 6),
                "loss": session.state.last_loss,
                "seconds": round(elapsed, 4),
            }

    def ingest_document(
        self,
        session_id: str,
        document: dict[str, Any],
        batches: Iterable[tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]],
    ) -> dict[str, Any]:
        """Stream encoded chunks into a session.

        ``batches`` yields (chunk_records, keys, values).  Only one batch is
        held in memory at a time; the source document is never fully loaded
        into GPU memory by this function.
        """
        with self._lock:
            session = self.get_session(session_id)
            original_docs = len(session.documents)
            original_chunks = len(session.chunks)
            original_state = session.state.clone()
            original_updates = session.updates
            original_adapt = session.adapt_seconds
            start = time.perf_counter()
            added = 0
            updates = 0
            pairs_seen = 0
            try:
                session.documents.append(document)
                for chunk_records, keys, values in batches:
                    if len(session.chunks) + len(chunk_records) > self.cfg.max_chunks:
                        raise ValueError(
                            f"session chunk capacity exceeded (max {self.cfg.max_chunks})"
                        )
                    session.chunks.extend(chunk_records)
                    added += len(chunk_records)
                    if keys.numel() == 0:
                        continue
                    keys = keys.to(self.device).detach()
                    values = values.to(self.device).detach()
                    pairs_seen += int(keys.shape[0])
                    # Bound each inner update to a small batch.
                    for i in range(0, keys.shape[0], 16):
                        k = keys[i : i + 16]
                        v = values[i : i + 16]
                        self.fast_memory.adapt(session.state, k, v)
                        session.updates += 1
                        updates += 1
                session.adapt_seconds += time.perf_counter() - start
                session.rebuild_index()
                self.save_session(session_id)
            except Exception:
                session.documents = session.documents[:original_docs]
                session.chunks = session.chunks[:original_chunks]
                session.state = original_state
                session.updates = original_updates
                session.adapt_seconds = original_adapt
                session.rebuild_index()
                raise
            return {
                "session_id": session_id,
                "document_id": document["doc_id"],
                "chunks_added": added,
                "total_chunks": len(session.chunks),
                "pairs_seen": pairs_seen,
                "updates": updates,
                "total_updates": session.updates,
                "adapt_seconds": round(time.perf_counter() - start, 4),
                "state_bytes": session.state_bytes(),
                "memory_state_bytes": session.state.state_bytes(),
            }

    # ------------------------------------------------------------------
    def summary(self, session: MemorySession) -> dict[str, Any]:
        docs = []
        for d in session.documents:
            docs.append(
                {
                    "doc_id": d.get("doc_id"),
                    "filename": d.get("filename"),
                    "kind": d.get("kind"),
                    "pages": d.get("pages"),
                    "size_bytes": d.get("size_bytes"),
                    "chunk_count": d.get("chunk_count"),
                    "created_at": d.get("created_at"),
                }
            )
        chunks = []
        for c in session.chunks[: self.cfg.max_chunks]:
            chunks.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_id": c.get("doc_id"),
                    "filename": c.get("filename"),
                    "page": c.get("page"),
                    "token_count": c.get("token_count"),
                    "records": len(c.get("records", [])),
                    "preview": (c.get("text") or "")[:280],
                }
            )
        return {
            "session_id": session.session_id,
            "name": session.name,
            "created_at": session.created_at,
            "format_version": SESSION_FORMAT_VERSION,
            "state_version": session.state.version,
            "weight_norm": round(session.state.weight_norm(), 6),
            "consolidations": int(
                session.state.metadata.get("consolidations", 0)
            ),
            "documents": docs,
            "chunks": chunks,
            "chunk_count": len(session.chunks),
            "record_count": session.record_count(),
            "document_count": len(session.documents),
            "updates": session.updates,
            "state_updates": session.state.updates,
            "adapt_seconds": round(session.adapt_seconds, 4),
            "memory_state_bytes": session.state.state_bytes(),
            "chunk_state_bytes": session.chunk_state_bytes(),
            "total_state_bytes": session.state_bytes(),
            "fast_parameters": session.parameter_count(),
            "memory_kind": session.state.kind,
            "memory_dim": (
                int(next(iter(session.state.params.values())).shape[-1])
                if session.state.params
                else None
            ),
            "last_query": session.last_query,
            "retrieval_counts": session.retrieval_counts,
            "capacity_chunks": self.cfg.max_chunks,
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            seen = set()
            for sid, session in self._sessions.items():
                out.append(self.summary(session))
                seen.add(sid)
            for meta_path in self.storage_dir.glob("*.json"):
                sid = meta_path.stem
                if sid in seen:
                    continue
                try:
                    out.append(json.loads(meta_path.read_text(encoding="utf-8")))
                except Exception:
                    continue
            out.sort(key=lambda s: s.get("created_at", 0), reverse=True)
            return out

    def record_query(self, session_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            session = self.get_session(session_id)
            mode = result.get("mode", "hybrid")
            session.retrieval_counts[mode] = session.retrieval_counts.get(mode, 0) + 1
            session.last_query = {
                "question": result.get("question", "")[:500],
                "mode": mode,
                "answer_type": result.get("answer_type"),
                "insufficient": result.get("insufficient", False),
                "top_score": result.get("confidence"),
                "timestamp": time.time(),
            }
            self.save_session(session_id)
