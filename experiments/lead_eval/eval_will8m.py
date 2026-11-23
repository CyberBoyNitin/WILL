"""Test 3 + Test 6 side-by-side: old checkpoint vs will-8m-synergy-v1.

Test 6 uses the exact Test 5 writer: a closed-form ridge consolidation of the
question->answer pairs into the fast state, not the online adapt path.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.benchmarks.runner import run_benchmark_suite  # noqa: E402
from backend.config import load_config  # noqa: E402
from backend.engine.tokenizer import BOS_ID, ByteTokenizer  # noqa: E402
from backend.memory.fast_memory import FastWeightMemory  # noqa: E402
from backend.memory.memory_encoder import pool_memory_tokens  # noqa: E402
from backend.memory.memory_manager import MemoryManager  # noqa: E402
from backend.training.checkpoints import (  # noqa: E402
    load_model_from_checkpoint,
    resolve_checkpoint,
)
from backend.training.datasets import (  # noqa: E402
    make_dataset,
    make_mixed_dataset,
    normalize_answer,
)

RUN_ID = "will8m_eval-" + uuid.uuid4().hex[:12]
CFG = load_config()
OUT = Path(CFG.server.experiment_dir) / "lead_eval" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
TOK = ByteTokenizer()
DEVICE = torch.device("cpu")
WARMUP, TIMED = 5, 50
OLD, NEW = "add-mixed-70-30-v2-final", "will-8m-synergy-v1"


def api_get(path):
    with urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=15) as r:
        return json.loads(r.read().decode())


def pct(samples, q):
    s = sorted(samples)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def load(cid):
    return load_model_from_checkpoint(
        resolve_checkpoint(CFG.server.checkpoint_dir, cid), device=DEVICE
    )


def warm(model, prompts, depth):
    for _ in range(WARMUP):
        model.generate_batch(prompts[:16], TOK, mode="adaptive_hard",
                             max_steps=depth, max_new_tokens=8)


def timed(model, prompts, depth, memory_state=None):
    times, last = [], None
    for _ in range(TIMED):
        t0 = time.perf_counter()
        last = model.generate_batch(
            prompts, TOK, mode="adaptive_hard", max_steps=depth,
            max_new_tokens=8, memory_state=memory_state,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": round(statistics.median(times), 4),
        "p95_ms": round(pct(times, 0.95), 4),
        "runs": len(times),
        "mean_depth": last["mean_depth"],
        "generated_ids": last["generated_ids"],
        "token_evaluations": last["token_evaluations"],
    }


REPORT: dict[str, Any] = {
    "format": "will-lead-eval-v1", "run_id": RUN_ID, "created_at": time.time(),
    "comparison": {"old": OLD, "new": NEW},
    "live_api_health": api_get("/api/health"),
    "settings": {"warmup_passes": WARMUP, "timed_runs": TIMED, "device": "cpu"},
    "results": {},
}

for tag, cid in (("old", OLD), ("new", NEW)):
    print(f"[{tag}] {cid}", flush=True)
    model, mcfg, payload = load(cid)
    depth = mcfg.resolved_adaptive_steps()
    seed = int((payload.get("training_config") or {}).get("seed", 1337))
    entry: dict[str, Any] = {
        "checkpoint_id": cid,
        "min_steps": mcfg.min_steps,
        "max_steps": mcfg.max_steps,
        "adaptive_depth": depth,
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    # ------------------------------------------------------------ Test 3
    rep3 = run_benchmark_suite(CFG, "mixed_difficulty", {
        "checkpoint_id": cid, "split": "test", "limit": 200,
        "mixed_easy_ratio": 0.5, "depths": [depth], "modes": ["adaptive_hard"],
        "batch_size": 16, "max_new_tokens": 12, "generation_repeats": 5,
        "max_examples": 0, "compile_probe": False, "device": "cpu",
    })
    ah = next(r for r in rep3["results"] if r["mode"] == "adaptive_hard")
    ds = make_mixed_dataset("arith_add", "test", 200, seed, TOK,
                            mcfg.max_seq_len, easy_ratio=0.5)
    easy = [list(x["prompt_ids"]) for x in ds.items if x["complexity"] == 1]
    hard = [list(x["prompt_ids"]) for x in ds.items if x["complexity"] == 2]
    warm(model, easy + hard, depth)
    te, th = timed(model, easy, depth), timed(model, hard, depth)
    strata = {}
    for label, count, t, key in (("easy", len(easy), te, "trivial"),
                                 ("hard", len(hard), th, "hard")):
        acc = ah["strata"].get(key, {})
        strata[label] = {
            "count": count,
            "accuracy_pct": round(100 * (acc.get("accuracy") or 0), 4),
            "mean_depth": acc.get("generation_mean_depth"),
            "latency_p50_ms": t["p50_ms"], "latency_p95_ms": t["p95_ms"],
            "runs": t["runs"],
        }
    entry["test3"] = {
        "suite_report_id": rep3["run_id"],
        "composition_ok": rep3["summary"]["composition_ok"],
        "strata": strata,
    }

    # ------------------------------------------------------------ Test 6
    seed6 = seed + 999_999
    qa_ds = make_dataset("arith_add", "test", 30, seed6, TOK,
                         mcfg.max_seq_len, complexity=2)
    qa = [(list(x["prompt_ids"]), x["answer_text"]) for x in qa_ds.items]
    prompts = [q for q, _ in qa]

    def accuracy_of(ids):
        texts = [TOK.decode(g, skip_special=True) for g in ids]
        return sum(normalize_answer(t) == normalize_answer(a)
                   for t, (_, a) in zip(texts, qa)) / len(qa)

    warm(model, prompts, depth)
    cold = timed(model, prompts, depth)
    cold_acc = accuracy_of(cold["generated_ids"])

    def enc(texts, max_len):
        seqs = [[BOS_ID] + TOK.encode(t, max_len=max_len) for t in texts]
        width = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), width), dtype=torch.long, device=DEVICE)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=DEVICE)
            mask[i, :len(s)] = 1
        with torch.no_grad():
            _, k_all, v_all = model.encode_memory(ids, mask)
        k = torch.stack([pool_memory_tokens(k_all[i, 1:int(mask[i].sum().item())])
                         for i in range(len(seqs))])
        v = torch.stack([pool_memory_tokens(v_all[i, 1:int(mask[i].sum().item())])
                         for i in range(len(seqs))])
        return k, v

    kp, _ = enc([TOK.decode(q, skip_special=True) for q, _ in qa], 64)
    _, va = enc([a for _, a in qa], 32)
    manager = MemoryManager(FastWeightMemory(mcfg), CFG.memory,
                            OUT / f"synergy_sessions_{tag}", device=DEVICE)
    sess = manager.create_session(f"synergy-{tag}")
    t0 = time.perf_counter()
    state = manager.fast_memory.consolidate(
        manager.get_session(sess.session_id).state, kp, va
    )
    manager.get_session(sess.session_id).state = state
    adapt_s = time.perf_counter() - t0
    wr = timed(model, prompts, depth, memory_state=state)
    entry["test6"] = {
        "questions": len(qa),
        "cold": {"mean_depth": cold["mean_depth"],
                 "accuracy_pct": round(100 * cold_acc, 4),
                 "latency_p50_ms": cold["p50_ms"],
                 "latency_p95_ms": cold["p95_ms"]},
        "adapt": {"pairs": len(qa), "seconds": round(adapt_s, 4),
                  "fast_state_bytes": state.state_bytes(),
                  "writer": "test5_ridge_consolidate"},
        "warm": {"mean_depth": wr["mean_depth"],
                 "accuracy_pct": round(100 * accuracy_of(wr["generated_ids"]), 4),
                 "latency_p50_ms": wr["p50_ms"],
                 "latency_p95_ms": wr["p95_ms"]},
    }
    entry["test6"]["depth_delta"] = round(
        entry["test6"]["warm"]["mean_depth"] - entry["test6"]["cold"]["mean_depth"], 4)
    entry["test6"]["p50_delta_ms"] = round(
        entry["test6"]["warm"]["latency_p50_ms"] - entry["test6"]["cold"]["latency_p50_ms"], 4)
    REPORT["results"][tag] = entry

(OUT / "report.json").write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
print("REPORT_ID", RUN_ID)
print(json.dumps(REPORT, indent=2))
