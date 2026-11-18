"""Health, system and overview endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.benchmarks.runner import list_reports
from backend.engine.metrics import hardware_info
from backend.server.deps import get_services
from backend.server.services import Services

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(services: Services = Depends(get_services)) -> dict:
    return {
        "status": "ok",
        "device": str(services.models.device),
        "torch_cuda": services.models.device.type == "cuda",
        "default_checkpoint_id": services.models.default_checkpoint_id,
        "database": str(services.db.path),
    }


@router.get("/system")
def system(services: Services = Depends(get_services)) -> dict:
    info = hardware_info()
    info.update(
        {
            "default_checkpoint_id": services.models.default_checkpoint_id,
            "checkpoint_count": len(services.models.list_checkpoints()),
            "experiment_dir": str(services.cfg.server.experiment_dir),
            "checkpoint_dir": str(services.cfg.server.checkpoint_dir),
            "memory_dir": str(services.cfg.server.memory_dir),
        }
    )
    return info


@router.get("/config")
def config(services: Services = Depends(get_services)) -> dict:
    data = services.cfg.model_dump()
    data["server"].pop("api_key", None)
    return data


@router.get("/overview")
def overview(services: Services = Depends(get_services)) -> dict:
    checkpoints = services.models.list_checkpoints()
    sessions = services.memory.list_sessions()
    jobs = services.db.list_jobs(limit=12)
    reports = list_reports(services.cfg.server.experiment_dir)[:8]
    latest_ckpt = checkpoints[0] if checkpoints else None
    return {
        "model": {
            "default_checkpoint_id": services.models.default_checkpoint_id,
            "device": str(services.models.device),
            "latest_checkpoint": latest_ckpt,
            "checkpoint_count": len(checkpoints),
            "parameters_total": (
                (latest_ckpt or {}).get("metrics", {}).get("parameters_total")
            ),
        },
        "counts": {
            "checkpoints": len(checkpoints),
            "memory_sessions": len(sessions),
            "jobs_total": len(services.db.list_jobs(limit=1000)),
            "jobs_running": len(
                [j for j in services.db.list_jobs(limit=1000) if j["status"] == "running"]
            ),
            "benchmark_reports": len(list_reports(services.cfg.server.experiment_dir)),
        },
        "recent_jobs": jobs,
        "recent_reports": reports,
        "recent_sessions": sessions[:5],
        "recent_checkpoints": checkpoints[:5],
    }
