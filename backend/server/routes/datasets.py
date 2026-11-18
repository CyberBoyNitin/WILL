"""Synthetic dataset inspection endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.server.deps import get_services
from backend.server.services import Services
from backend.training.datasets import TASK_NAMES, make_dataset

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DESCRIPTIONS = {
    "arith_add": "Focused addition with a controlled digit-width curriculum.",
    "arithmetic": "Mixed +, -, *, comparison and two-step arithmetic.",
    "sequence": "Sorting, parity, maximum, next-element and boolean logic.",
    "multihop": "Multi-fact entity chains with distractors.",
    "longctx": "Record/code retrieval from a longer synthetic context.",
    "lm": "Causal language modelling over generated technical sentences.",
    "mixture": "Round-robin mixture of arith_add, multihop, sequence and longctx.",
}


@router.get("")
def list_datasets() -> dict:
    return {
        "tasks": [
            {"name": name, "description": DESCRIPTIONS.get(name, "")}
            for name in TASK_NAMES
        ]
    }


@router.get("/{task}/sample")
def sample(
    task: str,
    split: str = Query(default="val", pattern="^(train|val|test)$"),
    n: int = Query(default=5, ge=1, le=50),
    seed: int = Query(default=1337, ge=0, le=2**31 - 1),
    min_complexity: int = Query(default=1, ge=1, le=4),
    max_complexity: int = Query(default=2, ge=1, le=4),
    services: Services = Depends(get_services),
) -> dict:
    if task not in TASK_NAMES:
        raise HTTPException(status_code=404, detail=f"unknown task {task!r}")
    if max_complexity < min_complexity:
        raise HTTPException(status_code=400, detail="max_complexity must be >= min_complexity")
    ds = make_dataset(
        task=task,
        split=split,
        size=n,
        seed=seed,
        tokenizer=services.models.tokenizer,
        max_seq_len=services.cfg.model.max_seq_len,
        min_complexity=min_complexity,
        max_complexity=max_complexity,
    )
    return {
        "task": task,
        "split": split,
        "seed": seed,
        "min_complexity": min_complexity,
        "max_complexity": max_complexity,
        "examples": [
            {
                "prompt": item["prompt"],
                "answer": item["answer_text"],
                "complexity": item["complexity"],
                "prompt_tokens": len(item["prompt_ids"]),
                "meta": item["meta"],
            }
            for item in ds.items
        ],
    }
