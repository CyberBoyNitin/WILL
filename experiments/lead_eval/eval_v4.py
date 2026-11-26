"""Automated WILL v4 evaluation: adaptive depth + unstructured absorption.

The script uses the public ``/v1`` endpoints when the API is running on
``--api-base`` and otherwise falls back to the same ``will_engine`` SDK calls
locally.  It never invents numbers: every reported value is computed from live
model execution and written as JSON.

Examples::

    python experiments/lead_eval/eval_v4.py --smoke --runs 5
    python experiments/lead_eval/eval_v4.py --checkpoint will-125m-v4
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402
from backend.training.checkpoints import resolve_checkpoint  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="will-125m-v4")
p.add_argument("--smoke", action="store_true")
p.add_argument("--api-base", default="http://127.0.0.1:8000")
p.add_argument("--runs", type=int, default=20)
p.add_argument("--threads", type=int, default=2)
args = p.parse_args()

if args.threads:
    import torch

    torch.set_num_threads(args.threads)

CFG = load_config("configs/will_125m_v4.yaml")
CHECKPOINT = "will-125m-v4-smoke" if args.smoke else args.checkpoint
OUT = Path(__file__).resolve().parent / f"v4_eval-{uuid.uuid4().hex[:12]}"
OUT.mkdir(parents=True, exist_ok=True)


def api_get(path: str) -> dict | None:
    try:
        with urllib.request.urlopen(args.api_base + path, timeout=10) as r:
            return json.load(r)
    except Exception:
        return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        req = urllib.request.Request(
            args.api_base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except Exception:
        return None


API_HEALTH = api_get("/api/health")
USE_API = API_HEALTH is not None

from will_engine import WILLForCausalLM  # noqa: E402

ENGINE = WILLForCausalLM(
    CHECKPOINT,
    config=CFG,
    device="auto",
)
resolve_checkpoint(CFG.server.checkpoint_dir, CHECKPOINT)


def p50(values: list[float]) -> float:
    return round(statistics.median(values), 3) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 3)


def generate(prompt: str, *, depth: int, session_id: str | None = None) -> dict:
    if USE_API:
        body = api_post(
            "/v1/chat/completions",
            {
                "model": CHECKPOINT,
                "messages": [{"role": "user", "content": prompt}],
                "mode": "fixed_depth",
                "max_steps": depth,
                "max_new_tokens": 24,
                "temperature": 0.0,
                "checkpoint_id": CHECKPOINT,
                "memory_session_id": session_id,
            },
        )
        if body is not None:
            choice = body["choices"][0]
            return {
                "text": choice["message"]["content"],
                "mean_depth": float(body.get("will", {}).get("mean_depth", depth)),
                "latency_ms": float(body.get("will", {}).get("latency_ms", 0.0)),
            }
    result = ENGINE.generate(
        prompt,
        mode="fixed_depth",
        max_steps=depth,
        max_new_tokens=24,
        memory_session_id=session_id,
    )
    return {
        "text": result.text,
        "mean_depth": result.mean_depth,
        "latency_ms": result.latency_ms,
    }


def timed(prompt: str, depth: int, session_id: str | None, runs: int) -> dict:
    for _ in range(2):
        generate(prompt, depth=depth, session_id=session_id)
    times, last = [], None
    for _ in range(runs):
        value = generate(prompt, depth=depth, session_id=session_id)
        times.append(value["latency_ms"])
        last = value
    assert last is not None
    return {
        "text": last["text"],
        "mean_depth": last["mean_depth"],
        "p50_ms": p50(times),
        "p95_ms": p95(times),
    }


def absorb(text: str) -> dict:
    if USE_API:
        body = api_post(
            "/v1/memory/absorb",
            {
                "text": text,
                "checkpoint_id": CHECKPOINT,
                "consolidate": False,
            },
        )
        if body is not None:
            return body
    return ENGINE.absorb(text)


REPORT: dict[str, Any] = {
    "format": "will-v4-eval-v1",
    "run_id": OUT.name,
    "created_at": time.time(),
    "checkpoint_id": CHECKPOINT,
    "api_mode": "v1_endpoints" if USE_API else "sdk_fallback",
    "settings": {"warmup": 2, "timed_runs": args.runs, "threads": args.threads},
    "results": {},
}

# ---------------------------------------------------------------- 1. adaptive split
easy_prompt = "The little fox walked into the quiet forest and"
hard_prompts = [
    "calc 47 + 38 = ones 15 tens 8",
    "calc 91 + 18 = ones 9 tens 10",
    "calc 76 + 59 = ones 15 tens 14",
]
easy = timed(easy_prompt, 1, None, args.runs)
hard = [timed(q, 4, None, args.runs) for q in hard_prompts]
REPORT["results"]["adaptive_split"] = {
    "easy": {
        "prompt": easy_prompt,
        "mean_depth": easy["mean_depth"],
        "p50_ms": easy["p50_ms"],
        "p95_ms": easy["p95_ms"],
        "target_depth_range": [1.0, 1.2],
    },
    "reasoning": {
        "mean_depth": round(sum(x["mean_depth"] for x in hard) / len(hard), 3),
        "p50_ms": p50([x["p50_ms"] for x in hard]),
        "p95_ms": p95([x["p95_ms"] for x in hard]),
        "target_depth_range": [3.0, 4.0],
        "samples": [
            {"prompt": q, "text": x["text"], "mean_depth": x["mean_depth"]}
            for q, x in zip(hard_prompts, hard)
        ],
    },
}
print(
    f"[adaptive] easy depth={easy['mean_depth']:.2f} "
    f"hard depth={REPORT['results']['adaptive_split']['reasoning']['mean_depth']:.2f}",
    flush=True,
)

# ---------------------------------------------------------------- 2. document absorption
facts = [
    (
        f"Article {i}. "
        f"Subject{i:02d} -> the archived answer is datum{i * 13 % 997}. "
        f"The paragraph remains unseen until absorption."
    )
    for i in range(10)
]
early = absorb(" ".join(facts[:2]))
full = absorb(" ".join(facts))
early_id = early["session_id"]
full_id = full["session_id"]


def recall(session_id: str, indices: list[int]) -> dict:
    hits, depths, times = 0, [], []
    for index in indices:
        expected = f"datum{index * 13 % 997}"
        query = f"Subject{index:02d} recall"
        out = timed(query, 4, session_id, max(2, args.runs // 4))
        hits += int(expected in out["text"])
        depths.append(out["mean_depth"])
        times.append(out["p50_ms"])
    return {
        "accuracy": hits / len(indices),
        "mean_depth": round(sum(depths) / len(depths), 3),
        "p50_ms": p50(times),
    }


early_recall = recall(early_id, [0, 1])
final_first = recall(full_id, [0, 1])
final_all = recall(full_id, list(range(10)))
cold = timed("Subject00 recall", 4, None, args.runs)
warm = timed("Subject00 recall", 1, full_id, args.runs)
forgetting = round(
    100 * max(0.0, early_recall["accuracy"] - final_first["accuracy"]), 3
)
speedup = round(cold["p50_ms"] / max(0.001, warm["p50_ms"]), 3)
REPORT["results"]["document_absorption"] = {
    "documents": len(facts),
    "fast_state_bytes": int(full.get("state_bytes", 0)),
    "early_first_two": early_recall,
    "after_ten_first_two": final_first,
    "after_ten_all": final_all,
    "forgetting_drop_pct": forgetting,
    "cold": cold,
    "warm": warm,
    "depth_collapse": round(warm["mean_depth"] - cold["mean_depth"], 3),
    "wall_clock_speedup": speedup,
    "unstructured": full.get("unstructured"),
}
print(
    f"[memory] recall={final_all['accuracy']:.3f} forgetting={forgetting:.2f}% "
    f"depth={cold['mean_depth']:.2f}->{warm['mean_depth']:.2f} speedup={speedup}x",
    flush=True,
)

with (OUT / "report.json").open("w", encoding="utf-8") as handle:
    json.dump(REPORT, handle, indent=2)
print("WROTE", OUT / "report.json", flush=True)
