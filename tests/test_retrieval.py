"""Retrieval, ingestion and grounded-answer tests."""
from __future__ import annotations

import torch

from backend.config import MemoryConfig
from backend.memory.bm25 import BM25Index
from backend.memory.document_ingestion import (
    DocumentIngestion,
    chunk_pages,
    load_document,
)
from backend.memory.fast_memory import FastWeightMemory
from backend.memory.memory_manager import MemoryManager
from backend.memory.memory_retriever import MemoryRetriever


def test_bm25_ranks_exact_key_first():
    index = BM25Index().fit(
        [
            "K00=V100. K01=V101.",
            "K02=V102. K03=V103.",
            "unrelated note about weather",
        ]
    )
    scores = index.scores("what is the value of K00?")
    assert len(scores) == 3
    assert scores[0] == max(scores)
    assert scores[0] > 0


def test_chunk_pages_overlap(tokenizer):
    text = " ".join(f"word{i}" for i in range(120))
    chunks = chunk_pages(
        [{"page_number": 1, "text": text}],
        tokenizer,
        chunk_size=48,
        overlap=8,
    )
    assert len(chunks) > 1
    assert chunks[0]["char_start"] == 0
    assert all(chunk["token_count"] <= 48 for chunk in chunks)
    assert all(chunk["text"] for chunk in chunks)


def test_text_ingestion_and_grounded_retrieval(tmp_path, tiny_model, tokenizer):
    cfg = MemoryConfig(max_chunks=20)
    manager = MemoryManager(
        FastWeightMemory(tiny_model.cfg), cfg, tmp_path / "sessions", device="cpu"
    )
    ingestion = DocumentIngestion(tiny_model, tokenizer, cfg, device="cpu")
    retriever = MemoryRetriever(tiny_model, tokenizer, cfg, device="cpu")
    session = manager.create_session("docs")
    result = ingestion.ingest_text(
        session.session_id,
        "K00=V100. K01=V101. K02=V102. access code alpha_1234.",
        manager,
        filename="facts.txt",
    )
    assert result["chunks_added"] >= 1
    assert result["document_id"]
    assert manager.get_session(session.session_id).state.updates >= 1

    session = manager.get_session(session.session_id)
    lexical = retriever.search(session, "what is the value of K00?", mode="lexical")
    assert lexical
    assert lexical[0]["filename"] == "facts.txt"
    assert "K00" in lexical[0]["text"] or "V100" in lexical[0]["text"]

    neural = retriever.search(session, "what is the value of K00?", mode="neural")
    assert neural
    assert neural[0]["filename"] == "facts.txt"

    hybrid = retriever.search(session, "what is the value of K00?", mode="hybrid")
    assert hybrid
    answer = retriever.answer(session, "what is the value of K00?", mode="hybrid")
    assert answer["answer"].strip().lower() == "v100"
    assert answer["sources"]
    assert answer["grounded"] is True


def test_insufficient_information_is_explicit(tmp_path, tiny_model, tokenizer):
    cfg = MemoryConfig(max_chunks=20)
    manager = MemoryManager(
        FastWeightMemory(tiny_model.cfg), cfg, tmp_path / "sessions", device="cpu"
    )
    retriever = MemoryRetriever(tiny_model, tokenizer, cfg, device="cpu")
    session = manager.create_session("empty")
    answer = retriever.answer(session, "what is the launch code?")
    assert answer["insufficient"] is True
    assert "insufficient" in answer["answer"].lower()
    assert answer["sources"] == []


def test_load_txt_and_markdown(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("hello world", encoding="utf-8")
    doc = load_document(txt)
    assert doc.pages[0]["text"] == "hello world"
    assert doc.filename == "a.txt"

    md = tmp_path / "b.md"
    md.write_text("# Title\n\nBody text.", encoding="utf-8")
    doc = load_document(md)
    assert "Title" in doc.pages[0]["text"]


def test_load_pdf(tmp_path):
    import pymupdf

    path = tmp_path / "facts.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "K00=V100. access code alpha_1234.")
    pdf.save(path)
    pdf.close()
    doc = load_document(path)
    assert "alpha_1234" in doc.pages[0]["text"]
    assert doc.filename == "facts.pdf"


def test_document_delete_removes_chunks(tmp_path, tiny_model, tokenizer):
    cfg = MemoryConfig(max_chunks=20)
    manager = MemoryManager(
        FastWeightMemory(tiny_model.cfg), cfg, tmp_path / "sessions", device="cpu"
    )
    ingestion = DocumentIngestion(tiny_model, tokenizer, cfg, device="cpu")
    session = manager.create_session("delete")
    ingestion.ingest_text(
        session.session_id, "K00=V100. K01=V101.", manager, filename="x.txt"
    )
    session = manager.get_session(session.session_id)
    doc_id = session.documents[0]["doc_id"]
    session.documents = [d for d in session.documents if d["doc_id"] != doc_id]
    session.chunks = [c for c in session.chunks if c["doc_id"] != doc_id]
    session.rebuild_index()
    manager.save_session(session.session_id)
    restored = manager.get_session(session.session_id)
    assert restored.documents == []
    assert restored.chunks == []
    assert len(restored.bm25.docs) == 0
