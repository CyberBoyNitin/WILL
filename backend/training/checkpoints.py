"""Checkpoint persistence and safe resolution."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from backend.config import ModelConfig, TrainingConfig
from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")


def software_info() -> dict[str, Any]:
    import platform

    import torch as _torch

    return {
        "python": platform.python_version(),
        "torch": _torch.__version__,
        "platform": platform.platform(),
        "cuda_available": bool(_torch.cuda.is_available()),
    }


def safe_checkpoint_id(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-")
    if not name:
        name = f"run-{int(time.time())}"
    return name[:96]


def resolve_checkpoint(checkpoint_dir: str | Path, checkpoint_id: str) -> Path:
    if not SAFE_ID.match(checkpoint_id):
        raise ValueError("invalid checkpoint id")
    root = Path(checkpoint_dir).resolve()
    path = (root / f"{checkpoint_id}.pt").resolve()
    if root not in path.parents:
        raise ValueError("checkpoint path escapes checkpoint directory")
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_id}")
    return path


def save_checkpoint(
    checkpoint_dir: str | Path,
    checkpoint_id: str,
    model: WILLModel,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    cid = safe_checkpoint_id(checkpoint_id)
    path = root / f"{cid}.pt"
    payload = {
        "format": "will-checkpoint-v1",
        "checkpoint_id": cid,
        "created_at": time.time(),
        "model_config": model_cfg.model_dump(),
        "training_config": training_cfg.model_dump(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "history": history,
        "software": software_info(),
        "seed": training_cfg.seed,
        "task": training_cfg.task,
        "profile": training_cfg.profile,
        "extra": extra or {},
    }
    torch.save(payload, path)
    meta = {
        "checkpoint_id": cid,
        "created_at": payload["created_at"],
        "model_config": payload["model_config"],
        "training_config": payload["training_config"],
        "metrics": metrics,
        "task": training_cfg.task,
        "profile": training_cfg.profile,
        "seed": training_cfg.seed,
        "software": payload["software"],
        "extra": extra or {},
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    (root / f"{cid}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    payload = torch.load(p, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "will-checkpoint-v1":
        raise ValueError("not a WILL checkpoint")
    return payload


def load_model_from_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[WILLModel, ModelConfig, dict[str, Any]]:
    payload = load_checkpoint(path, map_location="cpu")
    cfg = ModelConfig(**payload["model_config"])
    model = WILLModel(cfg)
    # v4 replaces the scalar ``mem_gate`` with ``memory_bridge``,
    # ``query_scale`` and ``memory_scale``.  Older checkpoints lack those keys;
    # strict=False loads them with the new defaults.
    model.load_state_dict(payload["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, cfg, payload


def list_checkpoints(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(checkpoint_dir)
    if not root.exists():
        return []
    out = []
    for meta_path in sorted(root.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = meta.get("checkpoint_id", meta_path.stem)
        pt = root / f"{cid}.pt"
        if not pt.exists():
            continue
        meta["path"] = str(pt)
        meta["size_bytes"] = pt.stat().st_size
        out.append(meta)
    out.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return out


def load_tokenizer_for_checkpoint(
    payload: dict[str, Any], tokenizer_dir: str | Path | None = None
) -> ByteTokenizer:
    return ByteTokenizer()
