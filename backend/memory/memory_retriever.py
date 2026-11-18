"""Lexical, fast-weight and hybrid retrieval over a memory session."""
from __future__ import annotations

import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from backend.config import MemoryConfig
from backend.engine.model import WILLModel
from backend.engine.tokenizer import BOS_ID, ByteTokenizer
from backend.memory.memory_manager import MemorySession
from backend.memory.memory_encoder import pool_memory_tokens

KEY_QUERY_RE = re.compile(r"(?i)\b(?:r|k|record_?)\d{1,4}\b")
VALUE_OF_RE = re.compile(r"(?i)(?:value|code)\s+(?:of|for)\s+([a-z0-9_]+)")


def _normalise_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        # A single candidate (or identical scores) should not be reported as
        # zero confidence when it is a real positive match.
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def extract_query_key(question: str) -> str:
    q = question.strip()
    lower = q.lower()
    if "access code" in lower:
        return "access code"
    m = KEY_QUERY_RE.search(q)
    if m:
        return m.group(0)
    m = VALUE_OF_RE.search(q)
    if m:
        return m.group(1)
    return q[:400]


class MemoryRetriever:
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
    @torch.no_grad()
    def query_key_vector(self, question: str) -> torch.Tensor:
        phrase = extract_query_key(question)
        ids = self.tokenizer.encode(phrase, max_len=128)
        if not ids:
            ids = self.tokenizer.encode(question[:128], max_len=128)
        ids = [BOS_ID] + ids
        tensor = torch.tensor([ids], device=self.device, dtype=torch.long)
        mask = torch.ones_like(tensor)
        _context, keys, _values = self.model.encode_memory(tensor, mask)
        vec = pool_memory_tokens(keys[0, 1:])
        return vec

    @torch.no_grad()
    def neural_prediction(self, session: MemorySession, question: str) -> torch.Tensor:
        q = self.query_key_vector(question)
        pred = self.model.memory.read(session.state, q.unsqueeze(0))[0]
        return F.normalize(pred.float(), dim=-1)

    def _record_rows(self, session: MemorySession) -> list[dict[str, Any]]:
        rows = []
        for chunk in session.chunks:
            for rec in chunk.get("records", []):
                vec = rec.get("value_vec")
                if isinstance(vec, torch.Tensor):
                    rows.append(
                        {
                            "chunk_id": chunk["chunk_id"],
                            "doc_id": chunk.get("doc_id"),
                            "filename": chunk.get("filename"),
                            "page": chunk.get("page"),
                            "text": chunk.get("text", ""),
                            "key_text": rec.get("key_text"),
                            "value_text": rec.get("value_text"),
                            "value_vec": vec,
                        }
                    )
        return rows

    def _chunk_rows(self, session: MemorySession) -> list[dict[str, Any]]:
        rows = []
        for chunk in session.chunks:
            vec = chunk.get("value_vec")
            if isinstance(vec, torch.Tensor):
                rows.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk.get("doc_id"),
                        "filename": chunk.get("filename"),
                        "page": chunk.get("page"),
                        "text": chunk.get("text", ""),
                        "value_vec": vec,
                    }
                )
        return rows

    @torch.no_grad()
    def neural_scores(self, session: MemorySession, question: str) -> list[dict[str, Any]]:
        if not session.chunks:
            return []
        pred = self.neural_prediction(session, question).to("cpu")
        rows = self._record_rows(session)
        if rows:
            scored = []
            for row in rows:
                vec = F.normalize(row["value_vec"].float(), dim=-1)
                score = float((pred * vec).sum().item())
                scored.append({**row, "score": score})
            # Keep the best record per chunk for citation-level results.
            best: dict[str, dict[str, Any]] = {}
            for row in scored:
                cid = row["chunk_id"]
                if cid not in best or row["score"] > best[cid]["score"]:
                    best[cid] = row
            return sorted(best.values(), key=lambda r: r["score"], reverse=True)
        scored = []
        for row in self._chunk_rows(session):
            vec = F.normalize(row["value_vec"].float(), dim=-1)
            score = float((pred * vec).sum().item())
            scored.append({**row, "score": score})
        return sorted(scored, key=lambda r: r["score"], reverse=True)

    def lexical_scores(self, session: MemorySession, question: str) -> list[dict[str, Any]]:
        if not session.chunks or session.bm25 is None:
            return []
        scores = session.bm25.scores(question)
        rows = []
        for chunk, score in zip(session.chunks, scores):
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk.get("doc_id"),
                    "filename": chunk.get("filename"),
                    "page": chunk.get("page"),
                    "text": chunk.get("text", ""),
                    "score": float(score),
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows

    # ------------------------------------------------------------------
    def search(
        self,
        session: MemorySession,
        question: str,
        mode: str = "hybrid",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in ("lexical", "neural", "hybrid"):
            raise ValueError("mode must be lexical, neural or hybrid")
        k = top_k or self.cfg.top_k
        lexical = self.lexical_scores(session, question)
        neural = self.neural_scores(session, question)
        if mode == "lexical":
            return lexical[:k]
        if mode == "neural":
            return neural[:k]

        lex_by_chunk = {r["chunk_id"]: r for r in lexical}
        neu_by_chunk = {r["chunk_id"]: r for r in neural}
        lex_norm = dict(zip([r["chunk_id"] for r in lexical], _normalise_scores([r["score"] for r in lexical])))
        neu_norm = dict(zip([r["chunk_id"] for r in neural], _normalise_scores([r["score"] for r in neural])))
        combined = []
        for cid in set(lex_by_chunk) | set(neu_by_chunk):
            base = neu_by_chunk.get(cid) or lex_by_chunk.get(cid)
            if base is None:
                continue
            lex_s = lex_norm.get(cid, 0.0)
            neu_s = neu_norm.get(cid, 0.0)
            score = self.cfg.lexical_weight * lex_s + self.cfg.neural_weight * neu_s
            combined.append(
                {
                    **base,
                    "score": float(score),
                    "lexical_score": float(lex_by_chunk.get(cid, {}).get("score", 0.0)),
                    "neural_score": float(neu_by_chunk.get(cid, {}).get("score", 0.0)),
                }
            )
        combined.sort(key=lambda r: r["score"], reverse=True)
        return combined[:k]

    # ------------------------------------------------------------------
    def _extract_answer(
        self, question: str, results: list[dict[str, Any]]
    ) -> tuple[str | None, dict[str, Any] | None]:
        if not results:
            return None, None
        lower = question.lower()
        patterns: list[re.Pattern] = []
        if "access code" in lower:
            patterns.append(re.compile(r"(?i)\baccess\s+code\b\s*[:=]?\s*([a-z]+_\d+)"))
        key = KEY_QUERY_RE.search(question)
        if key:
            patterns.append(
                re.compile(
                    rf"(?i)\b{re.escape(key.group(0))}\b\s*(?:[:=]|\s)\s*([a-z0-9_]{{1,32}})"
                )
            )
        m = VALUE_OF_RE.search(question)
        if m:
            patterns.append(
                re.compile(
                    rf"(?i)\b{re.escape(m.group(1))}\b\s*(?:[:=]|\s)\s*([a-z0-9_]{{1,32}})"
                )
            )
        for row in results:
            text = row.get("text") or ""
            for pattern in patterns:
                found = pattern.search(text)
                if found:
                    return found.group(1), row
        return None, None

    def _build_prompt(self, question: str, results: list[dict[str, Any]]) -> str:
        context_parts = []
        for row in results[:3]:
            context_parts.append(row.get("text", ""))
        context = " ".join(context_parts)
        max_tokens = self.model.cfg.max_seq_len - self.cfg.max_answer_tokens - 8
        ctx_ids = self.tokenizer.encode(context)
        if len(ctx_ids) > max_tokens:
            ctx_ids = ctx_ids[-max_tokens:]
        context = self.tokenizer.decode(ctx_ids, skip_special=True)
        return f"context: {context} question: {question} answer:"

    def answer(
        self,
        session: MemorySession,
        question: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        use_memory_in_generation: bool = True,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        results = self.search(session, question, mode=mode, top_k=top_k)
        top_score = float(results[0]["score"]) if results else 0.0
        sources = []
        for row in results:
            sources.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "doc_id": row.get("doc_id"),
                    "filename": row.get("filename"),
                    "page": row.get("page"),
                    "score": round(float(row.get("score", 0.0)), 6),
                    "lexical_score": (
                        round(float(row["lexical_score"]), 6)
                        if "lexical_score" in row
                        else None
                    ),
                    "neural_score": (
                        round(float(row["neural_score"]), 6)
                        if "neural_score" in row
                        else None
                    ),
                    "record_key": row.get("key_text"),
                    "record_value": row.get("value_text"),
                    "preview": (row.get("text") or "")[:360],
                }
            )
        if not session.chunks:
            return {
                "question": question,
                "answer": "insufficient information: this memory session contains no documents.",
                "answer_type": "insufficient",
                "grounded": False,
                "insufficient": True,
                "confidence": 0.0,
                "sources": [],
                "mode": mode,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "notes": "Upload and adapt a document first.",
            }
        extractive, source = self._extract_answer(question, results)
        generated = ""
        generation_error = None
        memory_state = session.state if (use_memory_in_generation and mode != "lexical") else None
        if extractive is None and top_score >= self.cfg.min_retrieval_score:
            prompt = self._build_prompt(question, results)
            ids = self.tokenizer.encode(prompt, add_bos=True)
            ids = ids[-self.model.cfg.max_seq_len :]
            try:
                out = self.model.generate(
                    torch.tensor([ids], device=self.device, dtype=torch.long),
                    self.tokenizer,
                    memory_state=memory_state,
                    mode="adaptive_hard",
                    max_steps=self.model.cfg.max_steps,
                    max_new_tokens=self.cfg.max_answer_tokens,
                )
                generated = (out.get("text") or "").strip()
            except Exception as exc:  # pragma: no cover - defensive
                generation_error = str(exc)
        if extractive is not None:
            answer = extractive
            answer_type = "extractive_from_source"
            grounded = True
            notes = "Answer extracted directly from the cited source chunk."
        elif generated:
            answer = generated
            answer_type = "generated"
            grounded = False
            notes = (
                "Generated by the local model from retrieved context. "
                "It is not a verified quote; check the cited sources."
            )
        else:
            answer = "insufficient information: no reliable answer was retrieved from this session."
            answer_type = "insufficient"
            grounded = False
            notes = (
                f"Top retrieval score {top_score:.4f} is below the threshold "
                f"{self.cfg.min_retrieval_score:.2f}, or the model produced no answer."
            )
            if generation_error:
                notes += f" Generation error: {generation_error}"
        return {
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "grounded": grounded,
            "insufficient": answer_type == "insufficient",
            "confidence": round(top_score, 6),
            "sources": sources,
            "mode": mode,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "notes": notes,
            "memory_used_in_generation": memory_state is not None,
        }

    def compare_modes(
        self, session: MemorySession, question: str, top_k: int | None = None
    ) -> dict[str, Any]:
        out = {}
        for mode in ("lexical", "neural", "hybrid"):
            results = self.search(session, question, mode=mode, top_k=top_k)
            out[mode] = [
                {
                    "chunk_id": r.get("chunk_id"),
                    "filename": r.get("filename"),
                    "page": r.get("page"),
                    "score": round(float(r.get("score", 0.0)), 6),
                    "record_key": r.get("key_text"),
                    "record_value": r.get("value_text"),
                    "preview": (r.get("text") or "")[:220],
                }
                for r in results
            ]
        return {"question": question, "results": out}
