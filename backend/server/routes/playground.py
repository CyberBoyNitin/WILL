"""Interactive generation, forward diagnostics and tokenisation."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from fastapi import APIRouter, Depends, HTTPException

from backend.engine.adaptive_controller import serializable_stats
from backend.engine.tokenizer import BOS_ID, SEP_ID
from backend.server.deps import get_services
from backend.server.schemas import ForwardRequest, GenerateRequest
from backend.server.services import Services

router = APIRouter(prefix="/api/playground", tags=["playground"])


def _require_model(services: Services, checkpoint_id: str | None):
    try:
        return services.models.get(checkpoint_id, allow_fresh=False)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail="no checkpoint is available; train or upload one first",
        ) from exc


def _memory_state(services: Services, session_id: str | None, model):
    if not session_id:
        return None
    session = services.memory.get_session(session_id)
    state = session.state
    dims = [int(v.shape[-1]) for v in state.params.values() if v.ndim >= 1]
    if dims and dims[0] != model.cfg.memory_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"memory session width {dims[0]} does not match model width "
                f"{model.cfg.memory_dim}; use a checkpoint from the same profile"
            ),
        )
    return state


@router.get("/presets")
def presets(services: Services = Depends(get_services)) -> dict:
    checkpoints = services.models.list_checkpoints()
    by_task: dict[str, str] = {}
    for meta in checkpoints:
        task = meta.get("task")
        if task and task not in by_task:
            by_task[task] = meta["checkpoint_id"]
    return {
        "presets": [
            {
                "name": "Single-digit addition",
                "task": "arith_add",
                "prompt": "calc 3 + 6 =",
                "checkpoint_id": by_task.get("arith_add"),
                "expected_style": "a short integer",
            },
            {
                "name": "Long-context record lookup",
                "task": "longctx",
                "prompt": "records: r0 amber. r1 basil. access code alpha_1234. question: what is the access code?",
                "checkpoint_id": by_task.get("longctx"),
                "expected_style": "a code or record value",
            },
            {
                "name": "Multi-hop entity chain",
                "task": "multihop",
                "prompt": "facts: e1 links e2. e2 owns item7. question: who owns item7?",
                "checkpoint_id": by_task.get("multihop"),
                "expected_style": "an entity name",
            },
        ],
        "checkpoints_by_task": by_task,
    }


@router.post("/tokenize")
def tokenize(
    req: dict,
    services: Services = Depends(get_services),
) -> dict:
    text = str(req.get("text", ""))
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    tokenizer = services.models.tokenizer
    ids = tokenizer.encode(text)
    return {
        "text": text,
        "token_count": len(ids),
        "ids": ids,
        "tokens": [tokenizer.token_text(i) for i in ids],
    }


@router.post("/generate")
def generate(req: GenerateRequest, services: Services = Depends(get_services)) -> dict:
    model, payload, cid = _require_model(services, req.checkpoint_id)
    tokenizer = services.models.tokenizer
    body = tokenizer.encode(req.prompt)
    budget = model.cfg.max_seq_len - 1 - (1 if req.append_sep else 0)
    body = body[-max(1, budget) :]
    ids = [BOS_ID] + body + ([SEP_ID] if req.append_sep else [])
    tensor = torch.tensor([ids], device=services.models.device, dtype=torch.long)
    state = _memory_state(services, req.memory_session_id, model)
    out = model.generate(
        tensor,
        tokenizer,
        memory_state=state,
        mode=req.mode,
        max_steps=req.max_steps,
        halt_epsilon=req.halt_epsilon,
        temperature=req.temperature,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
    )
    return {
        "checkpoint_id": cid,
        "task": payload.get("task"),
        "prompt": req.prompt,
        "prompt_ids": ids,
        "prompt_tokens": len(ids),
        "memory_used": state is not None,
        "result": out,
    }


@router.post("/forward")
def forward(req: ForwardRequest, services: Services = Depends(get_services)) -> dict:
    model, payload, cid = _require_model(services, req.checkpoint_id)
    tokenizer = services.models.tokenizer
    ids = tokenizer.encode(req.text, add_bos=True, max_len=model.cfg.max_seq_len)
    tensor = torch.tensor([ids], device=services.models.device, dtype=torch.long)
    mask = torch.ones_like(tensor)
    logits, stats = model(
        tensor,
        mask,
        mode=req.mode,
        max_steps=req.max_steps,
        halt_epsilon=req.halt_epsilon,
        memory_enabled=False,
    )
    probs = F.softmax(logits[0].float(), dim=-1)
    k = min(req.top_k, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    per_token = []
    for pos, token_id in enumerate(ids):
        per_token.append(
            {
                "index": pos,
                "id": int(token_id),
                "text": tokenizer.token_text(int(token_id)),
                "depth": round(float(stats["token_depth"][0, pos].item()), 4),
                "entropy": round(float(entropy[pos].item()), 5),
                "top": [
                    {
                        "id": int(top_ids[pos, j].item()),
                        "text": tokenizer.token_text(int(top_ids[pos, j].item())),
                        "prob": round(float(top_probs[pos, j].item()), 6),
                    }
                    for j in range(k)
                ],
            }
        )
    return {
        "checkpoint_id": cid,
        "task": payload.get("task"),
        "text": req.text,
        "token_count": len(ids),
        "logits_shape": list(logits.shape),
        "stats": serializable_stats(stats),
        "tokens": per_token,
        "mean_entropy": round(
            float(entropy.mean().item()) if len(ids) else 0.0, 5
        ),
        "perplexity_proxy": round(
            float(math.exp(min(entropy.mean().item(), 50.0))) if len(ids) else 1.0,
            4,
        ),
    }
