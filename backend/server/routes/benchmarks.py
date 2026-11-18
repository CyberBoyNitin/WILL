"""Asynchronous benchmark jobs and report access."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from backend.benchmarks.runner import (
    SUITES,
    list_reports,
    load_report,
    run_benchmark_suite,
)
from backend.server.deps import get_services
from backend.server.schemas import BenchmarkRequest
from backend.server.services import Services

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}

SUITE_DESCRIPTIONS = {
    "reasoning": "Exact-match generation and answer-only loss on a held-out task split at fixed recurrent depths.",
    "adaptive_compute": "Fixed-depth, differentiable ACT and hard early-exit comparison on the same held-out split.",
    "memory_recall": "Session-local fast-weight recall on synthetic key/value records with lexical and hybrid baselines.",
    "inference": "Warmup-controlled latency, throughput, depth and token-evaluation measurements.",
    "ablation": "Execution-mode ablation plus the memory-subsystem recall ablation.",
}


@router.get("/suites")
def suites() -> dict:
    return {
        "suites": [
            {"name": name, "description": SUITE_DESCRIPTIONS[name]} for name in SUITES
        ]
    }


@router.post("/start")
def start(req: BenchmarkRequest, services: Services = Depends(get_services)) -> dict:
    def run(progress, stop_event):
        progress(
            {
                "type": "start_benchmark",
                "suite": req.suite,
                "checkpoint_id": req.checkpoint_id,
                "options": req.options,
            }
        )
        report = run_benchmark_suite(
            services.cfg,
            req.suite,
            options={**req.options, "checkpoint_id": req.checkpoint_id},
            progress_cb=progress,
            stop_event=stop_event,
        )
        return {
            "run_id": report["run_id"],
            "suite": report["suite"],
            "checkpoint_id": report.get("checkpoint_id"),
            "summary": report.get("summary", {}),
            "duration_seconds": report.get("duration_seconds"),
            "artifacts": report.get("artifacts", {}),
            "cancelled": report.get("cancelled", False),
            "result_count": len(report.get("results", [])),
        }

    job = services.jobs.submit(
        "benchmark",
        {
            "suite": req.suite,
            "checkpoint_id": req.checkpoint_id,
            "options": req.options,
        },
        run,
    )
    return {"job_id": job["id"], "job": job}


@router.get("/jobs")
def jobs(
    limit: int = Query(default=50, ge=1, le=500),
    services: Services = Depends(get_services),
) -> dict:
    return {"jobs": services.db.list_jobs(kind="benchmark", limit=limit)}


@router.get("/jobs/{job_id}")
def job_detail(job_id: str, services: Services = Depends(get_services)) -> dict:
    job = services.db.get_job(job_id)
    if job is None or job["kind"] != "benchmark":
        raise HTTPException(status_code=404, detail="benchmark job not found")
    events = services.db.get_events(job_id, after_id=0, limit=2000)[-250:]
    return {"job": job, "events": events}


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, services: Services = Depends(get_services)) -> dict:
    job = services.db.get_job(job_id)
    if job is None or job["kind"] != "benchmark":
        raise HTTPException(status_code=404, detail="benchmark job not found")
    return {"job": services.jobs.cancel(job_id)}


@router.get("/jobs/{job_id}/events")
async def events(
    job_id: str,
    request: Request,
    after: int = Query(default=0, ge=0),
    services: Services = Depends(get_services),
):
    job = services.db.get_job(job_id)
    if job is None or job["kind"] != "benchmark":
        raise HTTPException(status_code=404, detail="benchmark job not found")

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


@router.get("/reports")
def reports(services: Services = Depends(get_services)) -> dict:
    return {"reports": list_reports(services.cfg.server.experiment_dir)}


@router.get("/reports/{run_id}")
def report(run_id: str, services: Services = Depends(get_services)) -> dict:
    try:
        return load_report(services.cfg.server.experiment_dir, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/reports/{run_id}/csv")
def report_csv(run_id: str, services: Services = Depends(get_services)):
    if not run_id or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        for ch in run_id
    ):
        raise HTTPException(status_code=400, detail="invalid benchmark run id")
    path = Path(services.cfg.server.experiment_dir) / "benchmarks" / run_id / "summary.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="benchmark csv not found")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{run_id}.csv",
        headers={"Cache-Control": "no-store"},
    )
