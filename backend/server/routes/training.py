"""Asynchronous training jobs and live event streams."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.config import TrainingConfig, apply_profile
from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer
from backend.server.deps import get_services
from backend.server.schemas import TrainRequest
from backend.server.services import Services
from backend.training.trainer import Trainer

router = APIRouter(prefix="/api/training", tags=["training"])

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def _build_job(services: Services, req: TrainRequest):
    base = apply_profile(services.cfg, req.profile)
    max_steps = req.max_steps or base.model.max_steps
    max_steps = max(max_steps, req.min_steps)
    model_cfg = base.model.model_copy(
        update={
            "max_steps": max_steps,
            "min_steps": req.min_steps,
            "halt_epsilon": req.halt_epsilon,
        }
    )
    data = req.model_dump()
    for key in ("device", "min_steps", "halt_epsilon"):
        data.pop(key, None)
    training_cfg = TrainingConfig(**data)
    device = req.device if req.device != "auto" else services.cfg.server.device
    run_name = req.checkpoint_name or (
        f"{req.task}-{req.profile}-{req.mode}-{int(time.time())}"
    )

    def run(progress, stop_event):
        tokenizer = ByteTokenizer()
        model = WILLModel(model_cfg)
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            checkpoint_dir=services.cfg.server.checkpoint_dir,
            device=device,
            progress_cb=progress,
            stop_event=stop_event,
            run_name=run_name,
        )
        progress(
            {
                "type": "start_training",
                "task": req.task,
                "mode": req.mode,
                "max_steps": max_steps,
                "min_steps": req.min_steps,
                "parameters": model.model_summary()["parameters_total"],
            }
        )
        metrics = trainer.train()
        checkpoint = metrics.get("checkpoint") or {}
        return {
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_val_accuracy": metrics.get("best_val_accuracy"),
            "best_epoch": metrics.get("best_epoch"),
            "duration_seconds": metrics.get("duration_seconds"),
            "parameters_total": metrics.get("parameters_total"),
            "device": metrics.get("device"),
            "task": req.task,
            "mode": req.mode,
            "max_steps": max_steps,
            "min_steps": req.min_steps,
            "history": metrics.get("history", []),
            "datasets": metrics.get("datasets", {}),
            "training_config": metrics.get("training_config", {}),
            "checkpoint": checkpoint,
        }

    config_public = {
        "task": req.task,
        "profile": req.profile,
        "mode": req.mode,
        "max_steps": max_steps,
        "min_steps": req.min_steps,
        "epochs": req.epochs,
        "train_size": req.train_size,
        "val_size": req.val_size,
        "test_size": req.test_size,
        "batch_size": req.batch_size,
        "lr": req.lr,
        "ponder_lambda": req.ponder_lambda,
        "ponder_warmup_epochs": req.ponder_warmup_epochs,
        "min_complexity": req.min_complexity,
        "max_complexity": req.max_complexity,
        "seed": req.seed,
        "checkpoint_name": run_name,
    }
    return run, config_public


@router.post("/start")
def start(req: TrainRequest, services: Services = Depends(get_services)) -> dict:
    fn, config = _build_job(services, req)
    job = services.jobs.submit("train", config, fn)
    return {"job_id": job["id"], "job": job}


@router.get("/defaults")
def defaults(services: Services = Depends(get_services)) -> dict:
    return {
        "training": services.cfg.training.model_dump(),
        "model": services.cfg.model.model_dump(),
        "profiles": ["tiny", "standard", "experimental", "custom"],
        "tasks": [
            "arith_add", "arithmetic", "sequence", "multihop",
            "longctx", "lm", "mixture",
        ],
    }


@router.get("/jobs")
def jobs(
    limit: int = Query(default=50, ge=1, le=500),
    services: Services = Depends(get_services),
) -> dict:
    return {"jobs": services.db.list_jobs(kind="train", limit=limit)}


@router.get("/jobs/{job_id}")
def job_detail(job_id: str, services: Services = Depends(get_services)) -> dict:
    job = services.db.get_job(job_id)
    if job is None or job["kind"] != "train":
        raise HTTPException(status_code=404, detail="training job not found")
    events = services.db.get_events(job_id, after_id=0, limit=2000)[-250:]
    return {"job": job, "events": events}


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, services: Services = Depends(get_services)) -> dict:
    job = services.db.get_job(job_id)
    if job is None or job["kind"] != "train":
        raise HTTPException(status_code=404, detail="training job not found")
    updated = services.jobs.cancel(job_id)
    return {"job": updated}


@router.get("/jobs/{job_id}/events")
async def events(
    job_id: str,
    request: Request,
    after: int = Query(default=0, ge=0),
    services: Services = Depends(get_services),
):
    job = services.db.get_job(job_id)
    if job is None or job["kind"] != "train":
        raise HTTPException(status_code=404, detail="training job not found")

    async def stream():
        last = after
        idle = 0
        while True:
            if await request.is_disconnected():
                break
            batch = services.db.get_events(job_id, after_id=last, limit=200)
            for event in batch:
                last = int(event["_event_id"])
                yield f"id: {last}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            current = services.db.get_job(job_id)
            if current and current["status"] in TERMINAL and not batch:
                yield f"event: end\ndata: {json.dumps({'status': current['status']})}\n\n"
                break
            idle += 1
            if idle > 1200:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
