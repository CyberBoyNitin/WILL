"""Memory session, ingestion, retrieval and adaptation endpoints."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.memory.document_ingestion import ALLOWED_SUFFIXES
from backend.server.deps import get_services
from backend.server.schemas import (
    MemoryAdaptRequest,
    MemoryCreateRequest,
    MemoryQueryRequest,
    MemorySearchRequest,
    MemoryTextRequest,
)
from backend.server.services import Services

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.post("/sessions")
def create_session(
    req: MemoryCreateRequest, services: Services = Depends(get_services)
) -> dict:
    return services.memory.create_session(req.name, req.metadata)


@router.get("/sessions")
def list_sessions(services: Services = Depends(get_services)) -> dict:
    return {"sessions": services.memory.list_sessions()}


@router.get("/sessions/{session_id}")
def get_session(session_id: str, services: Services = Depends(get_services)) -> dict:
    return services.memory.summary(session_id)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, services: Services = Depends(get_services)) -> dict:
    services.memory.delete_session(session_id)
    return {"deleted": session_id}


@router.post("/sessions/{session_id}/reset")
def reset_session(session_id: str, services: Services = Depends(get_services)) -> dict:
    return services.memory.reset_session(session_id)


@router.post("/sessions/{session_id}/ingest")
async def ingest_file(
    session_id: str,
    file: UploadFile = File(...),
    services: Services = Depends(get_services),
) -> dict:
    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type {suffix!r}; use PDF, TXT or Markdown",
        )
    limit = services.cfg.memory.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=413, detail=f"file exceeds the {limit} byte upload limit"
        )
    upload_dir = Path(services.cfg.server.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(data)
    try:
        document = services.memory.load_document(dest, name)
        result = services.memory.ingest_document(session_id, document)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return {
        "ingest": result,
        "session": services.memory.summary(session_id),
        "upload_path": str(dest),
    }


@router.post("/sessions/{session_id}/ingest-text")
def ingest_text(
    session_id: str,
    req: MemoryTextRequest,
    services: Services = Depends(get_services),
) -> dict:
    result = services.memory.ingest_text(session_id, req.text, filename=req.filename)
    return {"ingest": result, "session": services.memory.summary(session_id)}


@router.get("/sessions/{session_id}/documents")
def documents(session_id: str, services: Services = Depends(get_services)) -> dict:
    summary = services.memory.summary(session_id)
    return {
        "session_id": session_id,
        "documents": summary.get("documents", []),
        "chunks": summary.get("chunks", []),
    }


@router.delete("/sessions/{session_id}/documents/{doc_id}")
def delete_document(
    session_id: str, doc_id: str, services: Services = Depends(get_services)
) -> dict:
    return services.memory.delete_document(session_id, doc_id)


@router.post("/sessions/{session_id}/query")
def query(
    session_id: str,
    req: MemoryQueryRequest,
    services: Services = Depends(get_services),
) -> dict:
    return services.memory.query(
        session_id,
        req.question,
        mode=req.mode,
        top_k=req.top_k,
        use_memory_in_generation=req.use_memory_in_generation,
    )


@router.post("/sessions/{session_id}/search")
def search(
    session_id: str,
    req: MemorySearchRequest,
    services: Services = Depends(get_services),
) -> dict:
    return {
        "question": req.question,
        "mode": req.mode,
        "results": services.memory.search(
            session_id, req.question, mode=req.mode, top_k=req.top_k
        ),
    }


@router.post("/sessions/{session_id}/adapt")
def adapt(
    session_id: str,
    req: MemoryAdaptRequest,
    services: Services = Depends(get_services),
) -> dict:
    return services.memory.adapt_rounds(session_id, rounds=req.rounds)


@router.post("/sessions/{session_id}/save")
def save(session_id: str, services: Services = Depends(get_services)) -> dict:
    services.memory.manager.save_session(session_id)
    return services.memory.summary(session_id)


@router.post("/sessions/{session_id}/restore")
def restore(session_id: str, services: Services = Depends(get_services)) -> dict:
    services.memory.manager.reload_session(session_id)
    return services.memory.summary(session_id)
