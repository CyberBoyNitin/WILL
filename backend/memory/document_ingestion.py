"""PDF/TXT/Markdown ingestion, chunking and self-supervised memory pairs.

Only one encoded batch is held at a time.  The original document remains on
disk; the session stores source references and bounded vector records, never a
lossless copy of the document inside the neural state.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from backend.config import MemoryConfig
from backend.engine.model import WILLModel
from backend.engine.tokenizer import BOS_ID, ByteTokenizer
from backend.memory.memory_manager import MemoryManager
from backend.memory.memory_encoder import pool_memory_tokens

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
RECORD_RE = re.compile(
    r"(?i)\b((?:r|k|record_?)\d{1,4})\b\s*(?:[:=]|\s)\s*([a-z0-9_]{1,32})\b"
)
ACCESS_RE = re.compile(r"(?i)\baccess\s+code\b\s*[:=]?\s*([a-z]+_\d+)")
GENERIC_RE = re.compile(r"(?i)\b([a-z]{2,16})\s*[:=]\s*([a-z0-9_]{1,32})\b")
STOP_KEYS = {
    "the", "and", "for", "with", "from", "this", "that", "note", "page",
    "chapter", "section", "http", "https",
}


@dataclass
class Document:
    doc_id: str
    filename: str
    kind: str
    pages: list[dict[str, Any]]
    size_bytes: int
    sha256: str
    source_path: str
    created_at: float = field(default_factory=time.time)

    def meta(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "kind": self.kind,
            "pages": len(self.pages),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "source_path": self.source_path,
        }


def load_document(path: str | Path, filename: str | None = None) -> Document:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("document not found")
    name = Path(filename or p.name).name
    suffix = p.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported document type: {suffix}")
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyMuPDF is required for PDF ingestion") from exc
        try:
            pdf = pymupdf.open(p)
        except Exception as exc:
            raise ValueError(f"could not open PDF: {exc}") from exc
        if getattr(pdf, "needs_pass", False):
            raise ValueError("password-protected PDFs are not supported")
        pages = []
        for i, page in enumerate(pdf):
            if i >= 400:
                break
            text = page.get_text("text") or ""
            if text.strip():
                pages.append({"page_number": i + 1, "text": text})
        pdf.close()
        if not pages:
            raise ValueError("PDF contains no extractable text")
        return Document(
            doc_id=uuid.uuid4().hex,
            filename=name,
            kind="pdf",
            pages=pages,
            size_bytes=len(raw),
            sha256=sha,
            source_path=str(p),
        )
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        raise ValueError("document is empty")
    return Document(
        doc_id=uuid.uuid4().hex,
        filename=name,
        kind="markdown" if suffix in (".md", ".markdown") else "text",
        pages=[{"page_number": 1, "text": text}],
        size_bytes=len(raw),
        sha256=sha,
        source_path=str(p),
    )


def chunk_pages(
    pages: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    chunk_size: int = 192,
    overlap: int = 32,
) -> list[dict[str, Any]]:
    step = max(1, chunk_size - overlap)
    chunks: list[dict[str, Any]] = []
    idx = 0
    for page in pages:
        ids = tokenizer.encode(page["text"])
        if not ids:
            continue
        starts = list(range(0, len(ids), step))
        for start in starts:
            piece = ids[start : start + chunk_size]
            if len(piece) < 8 and chunks:
                break
            text = tokenizer.decode(piece, skip_special=True)
            chunks.append(
                {
                    "chunk_id": f"c{idx:05d}",
                    "page": int(page["page_number"]),
                    "text": text,
                    "token_count": len(piece),
                    "char_start": start,
                }
            )
            idx += 1
            if start + chunk_size >= len(ids):
                break
    return chunks


class DocumentIngestion:
    def __init__(
        self,
        model: WILLModel,
        tokenizer: ByteTokenizer,
        cfg: MemoryConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    def _char_to_token(self, text: str, char_idx: int) -> int:
        # Chunks are encoded as [BOS] + bytes, so token position is 1 + prefix.
        return 1 + len(self.tokenizer.encode(text[:char_idx]))

    def _pool_span(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        text: str,
        start_char: int,
        end_char: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        n = keys.shape[0]
        a = max(1, min(n, self._char_to_token(text, start_char)))
        b = max(a + 1, min(n + 1, self._char_to_token(text, end_char)))
        if a >= n or b <= a:
            return None
        span_k = keys[a - 1 : b - 1]
        span_v = values[a - 1 : b - 1]
        # Position-weighted pooling keeps short record keys/values
        # distinguishable while retaining context.
        k = pool_memory_tokens(span_k)
        v = pool_memory_tokens(span_v)
        return k, v

    def _record_pairs(
        self,
        text: str,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        matches: list[tuple[int, int, int, int, str, str]] = []
        for m in ACCESS_RE.finditer(text):
            key_start = m.start()
            key_end = key_start + len("access code")
            matches.append((key_start, key_end, m.start(1), m.end(1), "access code", m.group(1)))
        for m in RECORD_RE.finditer(text):
            matches.append((m.start(1), m.end(1), m.start(2), m.end(2), m.group(1), m.group(2)))
        for m in GENERIC_RE.finditer(text):
            key = m.group(1)
            if key.lower() in STOP_KEYS:
                continue
            matches.append((m.start(1), m.end(1), m.start(2), m.end(2), key, m.group(2)))
        for ks, ke, vs, ve, key_text, value_text in matches:
            if (ks, vs) in seen:
                continue
            seen.add((ks, vs))
            pooled = self._pool_span(keys, values, text, ks, ke)
            pooled_v = self._pool_span(keys, values, text, vs, ve)
            if pooled is None or pooled_v is None:
                continue
            records.append(
                {
                    "key_text": key_text[:64],
                    "value_text": value_text[:64],
                    "key_vec": pooled[0].detach().cpu(),
                    "value_vec": pooled_v[1].detach().cpu(),
                }
            )
            if len(records) >= 24:
                break
        if records:
            # Also encode each key in isolation and use that as the adaptation
            # key.  Queries arrive as isolated phrases, not inside the source
            # context, so this aligns the adaptation and query distributions.
            key_texts = [r["key_text"] for r in records]
            seqs = [[BOS_ID] + self.tokenizer.encode(k, max_len=64) for k in key_texts]
            max_len = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
            mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s, device=self.device)
                mask[i, : len(s)] = 1
            _c, kk, _v = self.model.encode_memory(ids, mask)
            for i, rec in enumerate(records):
                vec = kk[i, 1 : int(mask[i].sum().item())]
                if vec.shape[0] == 0:
                    continue
                qk = pool_memory_tokens(vec)
                rec["key_vec"] = qk.detach().cpu()
        return records

    def _window_pairs(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = keys.shape[0]
        w = max(4, min(self.cfg.window, n // 2 if n >= 8 else 4))
        stride = max(1, self.cfg.stride)
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for start in range(0, max(1, n - 2 * w + 1), stride):
            k = keys[start : start + w].mean(dim=0)
            v = values[start + w : start + 2 * w].mean(dim=0)
            if k.numel() == 0 or v.numel() == 0:
                continue
            ks.append(torch.nn.functional.normalize(k, dim=-1))
            vs.append(torch.nn.functional.normalize(v, dim=-1))
            if len(ks) >= 16:
                break
        if not ks:
            return torch.empty((0, keys.shape[-1])), torch.empty((0, values.shape[-1]))
        return torch.stack(ks), torch.stack(vs)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_chunk(
        self, text: str, doc_meta: dict[str, Any], chunk: dict[str, Any]
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        ids = self.tokenizer.encode(text)
        ids = [BOS_ID] + ids[: self.model.cfg.max_seq_len - 1]
        tensor = torch.tensor([ids], device=self.device, dtype=torch.long)
        mask = torch.ones_like(tensor)
        context, keys, values = self.model.encode_memory(tensor, mask)
        keys = keys[0, 1:]  # drop BOS
        values = values[0, 1:]
        if keys.shape[0] == 0:
            return (
                {
                    **chunk,
                    **doc_meta,
                    "value_vec": torch.zeros(self.model.cfg.memory_dim),
                    "key_vec": torch.zeros(self.model.cfg.memory_dim),
                    "records": [],
                },
                torch.empty((0, self.model.cfg.memory_dim)),
                torch.empty((0, self.model.cfg.memory_dim)),
            )
        key_vec = pool_memory_tokens(keys)
        value_vec = pool_memory_tokens(values)
        records = self._record_pairs(text, keys, values)
        for r in records:
            r["chunk_id"] = chunk["chunk_id"]
        chunk_record = {
            **chunk,
            "doc_id": doc_meta["doc_id"],
            "filename": doc_meta["filename"],
            "value_vec": value_vec.detach().cpu(),
            "key_vec": key_vec.detach().cpu(),
            "records": records,
        }
        wk, wv = self._window_pairs(keys, values)
        pair_k = [wk] + [r["key_vec"].unsqueeze(0) for r in records]
        pair_v = [wv] + [r["value_vec"].unsqueeze(0) for r in records]
        return chunk_record, torch.cat(pair_k, dim=0), torch.cat(pair_v, dim=0)

    def iter_batches(
        self, document: Document, chunks_per_batch: int = 4
    ) -> Iterator[tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]]:
        doc_meta = document.meta()
        chunks = chunk_pages(
            document.pages,
            self.tokenizer,
            self.cfg.chunk_size,
            self.cfg.chunk_overlap,
        )
        if not chunks:
            raise ValueError("document produced no chunks")
        batch_chunks: list[dict[str, Any]] = []
        batch_k: list[torch.Tensor] = []
        batch_v: list[torch.Tensor] = []
        for chunk in chunks:
            rec, k, v = self._encode_chunk(chunk["text"], doc_meta, chunk)
            batch_chunks.append(rec)
            if k.numel():
                batch_k.append(k)
                batch_v.append(v)
            if len(batch_chunks) >= chunks_per_batch:
                yield (
                    batch_chunks,
                    torch.cat(batch_k, dim=0) if batch_k else torch.empty((0, self.model.cfg.memory_dim)),
                    torch.cat(batch_v, dim=0) if batch_v else torch.empty((0, self.model.cfg.memory_dim)),
                )
                batch_chunks, batch_k, batch_v = [], [], []
        if batch_chunks:
            yield (
                batch_chunks,
                torch.cat(batch_k, dim=0) if batch_k else torch.empty((0, self.model.cfg.memory_dim)),
                torch.cat(batch_v, dim=0) if batch_v else torch.empty((0, self.model.cfg.memory_dim)),
            )

    def ingest(
        self, session_id: str, document: Document, manager: MemoryManager
    ) -> dict[str, Any]:
        chunks = chunk_pages(
            document.pages, self.tokenizer, self.cfg.chunk_size, self.cfg.chunk_overlap
        )
        doc_meta = document.meta()
        doc_meta["chunk_count"] = len(chunks)
        return manager.ingest_document(
            session_id, doc_meta, self.iter_batches(document)
        )

    def ingest_text(
        self,
        session_id: str,
        text: str,
        manager: MemoryManager,
        filename: str = "inline.txt",
    ) -> dict[str, Any]:
        doc = Document(
            doc_id=uuid.uuid4().hex,
            filename=filename,
            kind="text",
            pages=[{"page_number": 1, "text": text}],
            size_bytes=len(text.encode("utf-8")),
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            source_path="",
        )
        return self.ingest(session_id, doc, manager)
