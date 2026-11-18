"""Checkpoint listing, selection, evaluation and deletion."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.server.deps import get_services
from backend.server.services import Services
from backend.training.checkpoints import list_checkpoints, resolve_checkpoint
from backend.training.datasets import collate_lm, make_dataset
from backend.training.trainer import evaluate_generation, evaluate_loss
from torch.utils.data import DataLoader

router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])


def _meta(services: Services, checkpoint_id: str) -> dict:
    path = resolve_checkpoint(services.cfg.server.checkpoint_dir, checkpoint_id)
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="checkpoint metadata not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@router.get("")
def list_all(services: Services = Depends(get_services)) -> dict:
    return {
        "checkpoints": list_checkpoints(services.cfg.server.checkpoint_dir),
        "default_checkpoint_id": services.models.default_checkpoint_id,
    }


@router.get("/current")
def current(services: Services = Depends(get_services)) -> dict:
    return services.models.summary(services.models.default_checkpoint_id)


@router.get("/{checkpoint_id}")
def detail(checkpoint_id: str, services: Services = Depends(get_services)) -> dict:
    return _meta(services, checkpoint_id)


@router.get("/{checkpoint_id}/history")
def history(checkpoint_id: str, services: Services = Depends(get_services)) -> dict:
    meta = _meta(services, checkpoint_id)
    metrics = meta.get("metrics", {})
    return {
        "checkpoint_id": checkpoint_id,
        "history": metrics.get("history", []),
        "best_val_loss": metrics.get("best_val_loss"),
        "best_val_accuracy": metrics.get("best_val_accuracy"),
        "best_epoch": metrics.get("best_epoch"),
    }


@router.post("/{checkpoint_id}/select")
def select(checkpoint_id: str, services: Services = Depends(get_services)) -> dict:
    services.models.set_default(checkpoint_id)
    services.memory.refresh()
    return services.models.summary(checkpoint_id)


@router.get("/{checkpoint_id}/evaluate")
def evaluate(
    checkpoint_id: str,
    split: str = Query(default="test", pattern="^(train|val|test)$"),
    limit: int = Query(default=50, ge=1, le=1000),
    mode: str = Query(default="fixed_depth", pattern="^(fixed_depth|adaptive_soft|adaptive_hard)$"),
    max_steps: int | None = Query(default=None, ge=1, le=64),
    services: Services = Depends(get_services),
) -> dict:
    model, payload, cid = services.models.get(checkpoint_id, allow_fresh=False)
    tokenizer = services.models.tokenizer
    tcfg = payload.get("training_config") or {}
    ds = make_dataset(
        task=tcfg.get("task", "arith_add"),
        split=split,
        size=limit,
        seed=int(tcfg.get("seed", 1337)),
        tokenizer=tokenizer,
        max_seq_len=model.cfg.max_seq_len,
        min_complexity=int(tcfg.get("min_complexity", 1)),
        max_complexity=int(tcfg.get("max_complexity", 2)),
    )
    steps = min(max_steps or model.cfg.max_steps, model.cfg.max_steps)
    loader = DataLoader(ds, batch_size=16, collate_fn=collate_lm)
    loss = evaluate_loss(model, loader, services.models.device, mode, steps)
    gen = evaluate_generation(
        model,
        ds,
        tokenizer,
        services.models.device,
        mode,
        steps,
        batch_size=16,
        max_new_tokens=24,
        max_examples=8,
    )
    return {
        "checkpoint_id": cid,
        "split": split,
        "mode": mode,
        "max_steps": steps,
        "loss": loss,
        "generation": gen,
        "dataset": {
            "task": tcfg.get("task"),
            "size": len(ds),
            "seed": tcfg.get("seed"),
            "min_complexity": tcfg.get("min_complexity", 1),
            "max_complexity": tcfg.get("max_complexity", 2),
        },
    }


@router.delete("/{checkpoint_id}")
def delete(checkpoint_id: str, services: Services = Depends(get_services)) -> dict:
    path = resolve_checkpoint(services.cfg.server.checkpoint_dir, checkpoint_id)
    meta_path = path.with_suffix(".json")
    path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    services.models.unload(checkpoint_id)
    if services.models.default_checkpoint_id == checkpoint_id:
        remaining = list_checkpoints(services.cfg.server.checkpoint_dir)
        services.models.default_checkpoint_id = (
            remaining[0]["checkpoint_id"] if remaining else None
        )
        services.memory.refresh()
    return {
        "deleted": checkpoint_id,
        "default_checkpoint_id": services.models.default_checkpoint_id,
    }
