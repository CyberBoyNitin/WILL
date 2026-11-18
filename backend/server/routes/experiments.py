"""Experiment and training-result file access."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from backend.server.deps import get_services
from backend.server.services import Services

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _root(services: Services) -> Path:
    root = Path(services.cfg.server.experiment_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_path(services: Services, name: str) -> Path:
    root = _root(services)
    candidate = (root / name).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=400, detail="invalid experiment path")
    return candidate


@router.get("")
def list_experiments(services: Services = Depends(get_services)) -> dict:
    root = _root(services)
    items = []
    for path in sorted(root.rglob("*.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        lower = rel.lower()
        if lower.endswith("report.json") or lower.endswith("summary.json"):
            kind = "benchmark_report"
        elif "memory_sessions" in lower:
            kind = "memory_session"
        else:
            kind = "training_result"
        item = {
            "name": rel,
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "kind": kind,
        }
        if item["kind"] == "training_result" and rel.count("/") == 0:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                item["summary"] = {
                    "task": payload.get("training_config", {}).get("task"),
                    "mode": payload.get("training_config", {}).get("mode"),
                    "max_steps": payload.get("training_config", {}).get("max_steps"),
                    "best_val_loss": payload.get("best_val_loss"),
                    "best_val_accuracy": payload.get("best_val_accuracy"),
                    "best_epoch": payload.get("best_epoch"),
                    "parameters_total": payload.get("parameters_total"),
                    "duration_seconds": payload.get("duration_seconds"),
                }
            except Exception:
                item["summary"] = None
        items.append(item)
    items.sort(key=lambda x: x["modified_at"], reverse=True)
    return {"experiments": items}


@router.get("/{name:path}")
def get_experiment(name: str, services: Services = Depends(get_services)) -> dict:
    path = _safe_path(services, name)
    if path.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="only JSON experiment files are readable")
    if not path.exists():
        raise HTTPException(status_code=404, detail="experiment not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid experiment JSON: {exc}") from exc
