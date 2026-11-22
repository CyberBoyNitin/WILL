"""OpenAI-compatible endpoints backed by :class:`will_engine.WILLForCausalLM`.

These routes are additive: every existing ``/api`` route keeps its schema and
behaviour.  The ``will`` field on responses carries engine-specific diagnostics
that standard OpenAI clients simply ignore.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.server.deps import get_services
from backend.server.schemas import (
    ChatCompletionRequest,
    MemoryAbsorbRequest,
    OpenAIMemoryQueryRequest,
)
from backend.server.services import Services
from will_engine import WILLForCausalLM
from will_engine.telemetry import GLOBAL_TELEMETRY

router = APIRouter(tags=["openai"])

MODEL_ALIASES = {"will", "will-engine", "will:latest", "default", ""}


def _checkpoint_id(
    services: Services, model_name: str, checkpoint_id: str | None
) -> str | None:
    if checkpoint_id:
        return checkpoint_id
    if model_name and model_name not in MODEL_ALIASES:
        return model_name
    return services.models.default_checkpoint_id


def _engine(
    services: Services,
    model_name: str = "will",
    checkpoint_id: str | None = None,
) -> tuple[WILLForCausalLM, str]:
    cid = _checkpoint_id(services, model_name, checkpoint_id)
    if cid is None:
        raise HTTPException(
            status_code=400,
            detail="no checkpoint is available; train or upload one first",
        )
    try:
        model, payload, resolved = services.models.get(cid, allow_fresh=False)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"checkpoint {cid!r} not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    default_id = services.models.default_checkpoint_id
    manager = services.memory.manager if resolved == default_id else None
    engine = WILLForCausalLM.from_model(
        model,
        services.models.tokenizer,
        payload=payload,
        checkpoint_id=resolved,
        config=services.cfg,
        memory_manager=manager,
    )
    return engine, resolved or "will"


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _will_meta(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_id": result.get("checkpoint_id"),
        "mode": result.get("mode"),
        "mean_depth": result.get("mean_depth"),
        "token_evaluations": result.get("token_evaluations"),
        "estimated_flops": result.get("estimated_flops"),
        "latency_ms": result.get("latency_ms"),
        "memory_session_id": result.get("memory_session_id"),
    }


def _chunk(
    *,
    completion_id: str,
    created: int,
    model_name: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    will: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    if will is not None:
        payload["will"] = will
    return payload


def _sse(payload: Any) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _generation_kwargs(req: ChatCompletionRequest) -> dict[str, Any]:
    return {
        "max_new_tokens": req.max_new_tokens,
        "mode": req.mode,
        "max_steps": req.max_steps,
        "halt_epsilon": req.halt_epsilon,
        "temperature": req.temperature,
        "top_k": req.top_k,
        "memory_session_id": req.memory_session_id,
        "stop": req.stop,
    }


@router.get("/v1/models")
def list_models(services: Services = Depends(get_services)) -> dict:
    created = int(time.time())
    data = [
        {
            "id": "will",
            "object": "model",
            "created": created,
            "owned_by": "will",
            "will": {"default": True},
        }
    ]
    for checkpoint in services.models.list_checkpoints():
        data.append(
            {
                "id": checkpoint.get("checkpoint_id"),
                "object": "model",
                "created": int(checkpoint.get("created_at", created)),
                "owned_by": "will",
                "will": {
                    "task": checkpoint.get("task"),
                    "parameters": checkpoint.get("parameters"),
                },
            }
        )
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest, services: Services = Depends(get_services)
):
    engine, model_name = _engine(services, req.model, req.checkpoint_id)
    kwargs = _generation_kwargs(req)
    created = int(time.time())
    completion_id = _completion_id()

    if not req.stream:
        try:
            result = engine.chat(
                req.messages, template=req.chat_template, **kwargs
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                    },
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": result.usage,
            "will": {
                **_will_meta(result.to_dict()),
                "parameters": engine.model_summary()["parameters"],
            },
        }

    def render_event(event: dict[str, Any]) -> Iterator[str]:
        kind = event.get("type")
        if kind == "start":
            yield _sse(
                _chunk(
                    completion_id=completion_id,
                    created=created,
                    model_name=model_name,
                    delta={"role": "assistant", "content": ""},
                )
            )
        elif kind == "token":
            text = event.get("text") or ""
            if not text:
                return
            yield _sse(
                _chunk(
                    completion_id=completion_id,
                    created=created,
                    model_name=model_name,
                    delta={"content": text},
                )
            )
        elif kind == "done":
            result = event["result"]
            yield _sse(
                _chunk(
                    completion_id=completion_id,
                    created=created,
                    model_name=model_name,
                    delta={},
                    finish_reason=result.get("finish_reason") or "stop",
                    usage=result.get("usage"),
                    will=_will_meta(result),
                )
            )
        elif kind == "error":
            yield _sse(
                {
                    "error": {
                        "message": event.get("error", "generation failed"),
                        "type": "server_error",
                        "param": None,
                        "code": None,
                    }
                }
            )

    def event_stream() -> Iterator[str]:
        try:
            for event in engine.stream_chat(
                req.messages, template=req.chat_template, **kwargs
            ):
                yield from render_event(event)
        except Exception as exc:
            yield _sse(
                {
                    "error": {
                        "message": f"{type(exc).__name__}: {exc}",
                        "type": "server_error",
                        "param": None,
                        "code": None,
                    }
                }
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/v1/memory/absorb")
def memory_absorb(
    req: MemoryAbsorbRequest, services: Services = Depends(get_services)
):
    engine, _ = _engine(services, "will", req.checkpoint_id)
    try:
        return engine.absorb(
            req.text,
            session_id=req.session_id,
            name=req.name,
            filename=req.filename,
            consolidate=req.consolidate,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/memory/query")
def memory_query(
    req: OpenAIMemoryQueryRequest, services: Services = Depends(get_services)
):
    engine, _ = _engine(services, "will", req.checkpoint_id)
    try:
        return engine.query(
            req.session_id, req.question, mode=req.mode, top_k=req.top_k
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/v1/telemetry")
def telemetry(services: Services = Depends(get_services)) -> dict:
    try:
        engine, _ = _engine(services, "will", None)
        snapshot = engine.telemetry()
    except HTTPException:
        snapshot = GLOBAL_TELEMETRY.snapshot()
        snapshot["model"] = None
    except Exception as exc:
        snapshot = GLOBAL_TELEMETRY.snapshot()
        snapshot["model"] = None
        snapshot["telemetry_error"] = f"{type(exc).__name__}: {exc}"
    snapshot["object"] = "will.telemetry"
    snapshot["default_checkpoint_id"] = services.models.default_checkpoint_id
    snapshot["memory_sessions"] = len(services.memory.list_sessions())
    return snapshot
