"""Lead AI Evaluation & Benchmarking run for WILL.

Every number in the emitted report comes from real execution.  No mocking,
no hardcoded metrics.  Suite runs use backend.benchmarks.runner; custom
tests (baseline-forgetting and two-subsystem synergy) call the real model,
memory manager, and retriever directly.
"""
from __future__ import annotations

import inspect
import json
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

from backend.benchmarks.runner import (  # noqa: E402
    _eval_dataset,
    _hybrid_recall,
    _memory_state_hash,
    _model_state_hash,
    _neural_recall,
    run_benchmark_suite,
)
from backend.config import load_config  # noqa: E402
from backend.engine.compilation import maybe_compile  # noqa: E402
from backend.engine.metrics import count_parameters, hardware_info  # noqa: E402
from backend.engine.model import WILLModel  # noqa: E402
from backend.engine.tokenizer import BOS_ID, ByteTokenizer  # noqa: E402
from backend.memory.document_ingestion import DocumentIngestion  # noqa: E402
from backend.memory.fast_memory import FastWeightMemory  # noqa: E402
from backend.memory.memory_encoder import pool_memory_tokens  # noqa: E402
from backend.memory.memory_manager import MemoryManager  # noqa: E402
from backend.memory.memory_retriever import MemoryRetriever  # noqa: E402
from backend.training.checkpoints import (  # noqa: E402
    load_model_from_checkpoint,
    resolve_checkpoint,
)
from backend.training.datasets import (  # noqa: E402
    make_dataset,
    make_mixed_dataset,
    normalize_answer,
)
from backend.training.trainer import evaluate_generation  # noqa: E402

RUN_ID = "lead_eval-" + uuid.uuid4().hex[:12]
CFG = load_config()
OUT_DIR = Path(CFG.server.experiment_dir) / "lead_eval" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER = ByteTokenizer()
DEVICE = torch.device("cpu")
WARMUP = 5
TIMED = 50


def api_get(path: str) -> dict[str, Any]:
    with urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def p50_p95(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(-(-0.95 * len(ordered) // 1)) - 1)]
    return round(statistics.median(ordered), 4), round(p95, 4)


def load(cid: str):
    path = resolve_checkpoint(CFG.server.checkpoint_dir, cid)
    model, mcfg, payload = load_model_from_checkpoint(path, device=DEVICE)
    return model, mcfg, payload


def batch_prompts(ds) -> list[list[int]]:
    return [list(item["prompt_ids"]) for item in ds.items]


def warm_generation(model, prompts, mode, depth, n=WARMUP):
    for _ in range(n):
        model.generate_batch(
            prompts[:16], TOKENIZER, mode=mode, max_steps=depth, max_new_tokens=8
        )


def timed_passes(model, prompts, mode, depth, passes=TIMED, max_new_tokens=8,
                 memory_state=None):
    times = []
    last = None
    for _ in range(passes):
        t0 = time.perf_counter()
        last = model.generate_batch(
            prompts, TOKENIZER, mode=mode, max_steps=depth,
            max_new_tokens=max_new_tokens, memory_state=memory_state,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    p50, p95 = p50_p95(times)
    return {
        "p50_ms": p50, "p95_ms": p95, "runs": len(times),
        "all_ms": [round(t, 3) for t in times],
        "mean_depth": last["mean_depth"] if last else None,
        "token_evaluations": last["token_evaluations"] if last else None,
        "generated_ids": last["generated_ids"] if last else None,
    }


def generated_token_count(model, ds, mode, depth, batch_size=16):
    prompts = batch_prompts(ds)
    total = 0
    for i in range(0, len(prompts), batch_size):
        out = model.generate_batch(
            prompts[i:i + batch_size], TOKENIZER, mode=mode, max_steps=depth,
            max_new_tokens=12,
        )
        total += sum(len(g) for g in out["generated_ids"])
    return total


REPORT: dict[str, Any] = {
    "format": "will-lead-eval-v1",
    "run_id": RUN_ID,
    "created_at": time.time(),
    "settings": {
        "warmup_passes": WARMUP,
        "timed_runs": TIMED,
        "device": str(DEVICE),
        "api_base": "http://127.0.0.1:8000",
        "notes": [
            "All values are produced by real model execution on held-out data.",
            "MAC/FLOP values are analytic estimates, not power measurements.",
            "No energy savings are claimed.",
        ],
    },
}


# ----------------------------------------------------------------------
# TEST 1: Hardware & baseline audit
# ----------------------------------------------------------------------
print("[1/6] hardware + baseline audit", flush=True)
api_health = api_get("/api/health")
api_system = api_get("/api/system")
m1, mcfg1, p1 = load("add-single-act-final")
slow_params = count_parameters(m1)
fast_bytes = m1.memory.init_state().state_bytes()
hard_src = inspect.getsource(
    __import__("backend.engine.adaptive_controller", fromlist=["x"])
    .AdaptiveComputationController._hard
)
compacted = "index_select" in hard_src and "nonzero" in hard_src
probe_ids = torch.randint(1, mcfg1.vocab_size, (1, 8))
def _probe(module):
    logits, _ = module(
        probe_ids, torch.ones_like(probe_ids), mode="fixed_depth",
        max_steps=mcfg1.resolved_adaptive_steps(), memory_enabled=False,
    )
    return logits
_, compile_status = maybe_compile(m1, enabled=True, mode="reduce-overhead", probe=_probe)
REPORT["test1_hardware_audit"] = {
    "api_health": api_health,
    "api_system": api_system,
    "hardware": hardware_info(),
    "slow_parameters": slow_params,
    "fast_state_bytes": fast_bytes,
    "fast_state_parameters": m1.memory.init_state().parameter_count(),
    "max_depth": mcfg1.max_steps,
    "resolved_adaptive_depth": mcfg1.resolved_adaptive_steps(),
    "min_steps": mcfg1.min_steps,
    "halting_mechanism": {
        "physically_compacted_active_batch": compacted,
        "evidence": "active.nonzero() gathers via index_select; halted rows are written out once",
    },
    "torch_compile": compile_status,
}


# ----------------------------------------------------------------------
# TEST 2: Adaptive compute + real wall-clock speedup (200 held-out)
# ----------------------------------------------------------------------
print("[2/6] adaptive compute (200 held-out prompts, 5 repeats)", flush=True)
ds2, _, meta2 = _eval_dataset(p1, TOKENIZER, {"split": "test", "limit": 200})
warm_generation(m1, batch_prompts(ds2), "adaptive_hard", mcfg1.resolved_adaptive_steps())
rep2 = run_benchmark_suite(CFG, "adaptive_compute", {
    "checkpoint_id": "add-single-act-final", "split": "test", "limit": 200,
    "depths": [1, 2, 3, 4], "modes": ["adaptive_soft", "adaptive_hard"],
    "batch_size": 16, "max_new_tokens": 12, "generation_repeats": 5,
    "max_examples": 0, "compile_probe": False, "device": "cpu",
})
t4 = next(r for r in rep2["results"] if r["mode"] == "fixed_depth" and r["max_steps"] == 4)
rows2 = []
for r in rep2["results"]:
    mode = r["mode"]
    depth = r["max_steps"]
    gen_tok = generated_token_count(m1, ds2, mode, depth)
    rows2.append({
        "config": f"{mode}@{depth}", "mode": mode, "depth": depth,
        "accuracy_pct": round(100 * (r["accuracy"] or 0), 4),
        "mean_depth": r["generation_mean_depth"],
        "token_evaluations": r["generation_token_evaluations"],
        "estimated_macs": r["generation_estimated_macs"],
        "wall_clock_seconds": r["generation_seconds"],
        "wall_clock_samples": r["generation_seconds_all"],
        "speedup_vs_fixed4": round(t4["generation_seconds"] / r["generation_seconds"], 4),
        "generated_tokens": gen_tok,
        "tokens_per_second": round(gen_tok / r["generation_seconds"], 2),
    })
REPORT["test2_adaptive_compute"] = {
    "suite_report_id": rep2["run_id"],
    "dataset": meta2,
    "results": rows2,
    "speedup_definition": "fixed_depth@4 generation_seconds / adaptive_hard@3 generation_seconds",
}


# ----------------------------------------------------------------------
# TEST 3: Mixed difficulty 100 easy / 100 hard, p50/p95 latency
# ----------------------------------------------------------------------
print("[3/6] mixed difficulty (100 easy + 100 hard)", flush=True)
m3, mcfg3, p3 = load("add-mixed-70-30-v2-best")
tcfg3 = p3.get("training_config") or {}
seed3 = int(tcfg3.get("seed", 1337))
rep3 = run_benchmark_suite(CFG, "mixed_difficulty", {
    "checkpoint_id": "add-mixed-70-30-v2-best", "split": "test", "limit": 200,
    "mixed_easy_ratio": 0.5, "depths": [3], "modes": ["adaptive_hard"],
    "batch_size": 16, "max_new_tokens": 12, "generation_repeats": 5,
    "max_examples": 0, "compile_probe": False, "device": "cpu",
})
ah3 = next(r for r in rep3["results"] if r["mode"] == "adaptive_hard")
ds3 = make_mixed_dataset(
    "arith_add", "test", 200, seed3, TOKENIZER, mcfg3.max_seq_len, easy_ratio=0.5
)
easy = [list(x["prompt_ids"]) for x in ds3.items if x["complexity"] == 1]
hard = [list(x["prompt_ids"]) for x in ds3.items if x["complexity"] == 2]
warm_generation(m3, easy + hard, "adaptive_hard", 3)
late_easy = timed_passes(m3, easy, "adaptive_hard", 3)
late_hard = timed_passes(m3, hard, "adaptive_hard", 3)
strata = {}
for label, prompts, timed, count in (
    ("easy", easy, late_easy, len(easy)), ("hard", hard, late_hard, len(hard))
):
    acc = ah3["strata"].get("trivial" if label == "easy" else "hard", {})
    strata[label] = {
        "count": count,
        "accuracy_pct": round(100 * (acc.get("accuracy") or 0), 4),
        "mean_depth": acc.get("generation_mean_depth"),
        "total_wall_clock_seconds_suite": acc.get("generation_seconds"),
        "latency_p50_ms_per_pass": timed["p50_ms"],
        "latency_p95_ms_per_pass": timed["p95_ms"],
        "latency_p50_ms_per_prompt": round(timed["p50_ms"] / count, 4),
        "runs": timed["runs"],
    }
REPORT["test3_mixed_difficulty"] = {
    "suite_report_id": rep3["run_id"],
    "composition": rep3["dataset"],
    "composition_ok": rep3["summary"]["composition_ok"],
    "strata": strata,
}


# ----------------------------------------------------------------------
# TEST 4: Short vs long prompt latency profiling
# ----------------------------------------------------------------------
print("[4/6] prompt-length latency profile (16..256 tokens, 50 runs)", flush=True)
rep4 = run_benchmark_suite(CFG, "inference", {
    "checkpoint_id": "add-single-act-final",
    "prompt_lengths": [16, 32, 64, 128, 256],
    "depths": [2, 4], "modes": ["adaptive_hard"],
    "repeats": TIMED, "max_new_tokens": 8, "compile_probe": False, "device": "cpu",
})
REPORT["test4_latency_profile"] = {
    "suite_report_id": rep4["run_id"],
    "results": [
        {
            "config": f"{r['mode']}@{r['max_steps']}",
            "prompt_tokens": r["prompt_tokens"],
            "p50_ms": r["latency_ms_p50"],
            "p95_ms": r["latency_ms_p95"],
            "token_evaluations": r["token_evaluations"],
            "runs": r["repeats"],
        }
        for r in rep4["results"]
    ],
}


# ----------------------------------------------------------------------
# TEST 5: Sequential continual learning, baseline before/after
# ----------------------------------------------------------------------
print("[5/6] sequential continual learning (3 waves x 20 facts)", flush=True)
m5, mcfg5, p5 = load("add-single-act-final")
tcfg5 = p5.get("training_config") or {}
base_ds = make_dataset(
    "arith_add", "test", 150, int(tcfg5.get("seed", 1337)), TOKENIZER,
    mcfg5.max_seq_len, complexity=1,
)
def baseline_accuracy():
    g = evaluate_generation(
        m5, base_ds, TOKENIZER, DEVICE, "adaptive_hard",
        mcfg5.resolved_adaptive_steps(), batch_size=16, max_new_tokens=12,
        max_examples=0,
    )
    return g["accuracy"]

base_before = baseline_accuracy()
sess_dir = OUT_DIR / "continual_sessions"
manager5 = MemoryManager(FastWeightMemory(mcfg5), CFG.memory, sess_dir, device=DEVICE)
sess5 = manager5.create_session("lead-continual")
ing5 = DocumentIngestion(m5, TOKENIZER, CFG.memory, device=DEVICE)
ret5 = MemoryRetriever(m5, TOKENIZER, CFG.memory, device=DEVICE)
slow_before = _model_state_hash(m5)
sets: list[list[tuple[str, str]]] = []
waves = []
for wave in range(3):
    records = [(f"K{wave * 20 + i:02d}", f"V{100 + wave * 20 + i}") for i in range(20)]
    sets.append(records)
    seen = [rec for s in sets for rec in s]
    text = " ".join(f"{k}={v}." for k, v in records)
    t0 = time.perf_counter()
    ing5.ingest_text(sess5.session_id, text, manager5, filename=f"wave_{wave}.txt")
    sess5 = manager5.get_session(sess5.session_id)
    cons = manager5.consolidate_session(sess5.session_id)
    adapt_s = time.perf_counter() - t0
    sess5 = manager5.get_session(sess5.session_id)
    recall = {}
    for idx, recs in enumerate(sets):
        neural, _ = _neural_recall(ret5, sess5, recs)
        hybrid, _ = _hybrid_recall(ret5, sess5, recs, CFG.memory)
        recall[f"set_{chr(65 + idx)}"] = {
            "neural_recall_at_1": round(neural, 4),
            "hybrid_recall_at_1": round(hybrid, 4),
        }
    waves.append({
        "wave": wave + 1, "facts_added": len(records), "facts_seen": len(seen),
        "recall": recall, "adaptation_seconds": round(adapt_s, 4),
        "fast_state_bytes": sess5.state.state_bytes(),
        "weight_norm": round(sess5.state.weight_norm(), 6),
        "consolidations": cons["pairs"],
    })
base_after = baseline_accuracy()
slow_after = _model_state_hash(m5)
set_a_first = waves[0]["recall"]["set_A"]["neural_recall_at_1"]
set_a_last = waves[-1]["recall"]["set_A"]["neural_recall_at_1"]
REPORT["test5_continual"] = {
    "baseline_accuracy_before": round(base_before, 4),
    "baseline_accuracy_after_wave3": round(base_after, 4),
    "baseline_forgetting_pct": round(100 * max(0.0, base_before - base_after), 4),
    "set_A_recall_wave1": set_a_first,
    "set_A_recall_wave3": set_a_last,
    "set_A_forgetting_pct": round(100 * max(0.0, set_a_first - set_a_last), 4),
    "slow_weights_unchanged": slow_before == slow_after,
    "waves": waves,
}


# ----------------------------------------------------------------------
# TEST 6: Two-subsystem synergy (memory + adaptive compute)
# ----------------------------------------------------------------------
print("[6/6] two-subsystem synergy (30 unfamiliar hard questions)", flush=True)
m6, mcfg6, p6 = load("add-mixed-70-30-v2-best")
tcfg6 = p6.get("training_config") or {}
seed6 = int(tcfg6.get("seed", 1337)) + 999_999
qa_ds = make_dataset(
    "arith_add", "test", 30, seed6, TOKENIZER, mcfg6.max_seq_len, complexity=2
)
qa = [(list(x["prompt_ids"]), x["answer_text"]) for x in qa_ds.items]
prompts6 = [q for q, _ in qa]

def accuracy_of(out):
    texts = [TOKENIZER.decode(ids, skip_special=True) for ids in out["generated_ids"]]
    correct = sum(
        normalize_answer(t) == normalize_answer(a)
        for t, (_, a) in zip(texts, qa)
    )
    return correct / len(qa)

warm_generation(m6, prompts6, "adaptive_hard", 3)
cold = timed_passes(m6, prompts6, "adaptive_hard", 3)
# Accuracy from the final timed pass.
cold_acc = accuracy_of({"generated_ids": cold["generated_ids"]})

# Encode supporting facts as (question-key, answer-value) fast-weight pairs.
q_ids, a_ids = [], []
for q, a in qa:
    q_ids.append([BOS_ID] + TOKENIZER.encode(TOKENIZER.decode(q, skip_special=True), max_len=64))
    a_ids.append([BOS_ID] + TOKENIZER.encode(a, max_len=32))
max_q = max(len(x) for x in q_ids)
max_a = max(len(x) for x in a_ids)
qb = torch.zeros((len(q_ids), max_q), dtype=torch.long)
qmask = torch.zeros_like(qb)
for i, seq in enumerate(q_ids):
    qb[i, :len(seq)] = torch.tensor(seq)
    qmask[i, :len(seq)] = 1
ab = torch.zeros((len(a_ids), max_a), dtype=torch.long)
amask = torch.zeros_like(ab)
for i, seq in enumerate(a_ids):
    ab[i, :len(seq)] = torch.tensor(seq)
    amask[i, :len(seq)] = 1
with torch.no_grad():
    _c, k_all, v_all = m6.encode_memory(qb, qmask)
    _c2, _k2, va = m6.encode_memory(ab, amask)
    keys = torch.stack([
        pool_memory_tokens(k_all[i, 1:int(qmask[i].sum().item())])
        for i in range(len(q_ids))
    ])
    values = torch.stack([
        pool_memory_tokens(va[i, 1:int(amask[i].sum().item())])
        for i in range(len(a_ids))
    ])
manager6 = MemoryManager(
    FastWeightMemory(mcfg6), CFG.memory, OUT_DIR / "synergy_sessions", device=DEVICE
)
sess6 = manager6.create_session("lead-synergy")
t_adapt = time.perf_counter()
adapt = manager6.adapt_pairs(sess6.session_id, keys, values, persist=True)
warm_state = manager6.get_session(sess6.session_id).state
adapt_s = time.perf_counter() - t_adapt
# Prompt context is unchanged: no facts are concatenated into the questions.
warm = timed_passes(m6, prompts6, "adaptive_hard", 3, memory_state=warm_state)
warm_acc = accuracy_of({"generated_ids": warm["generated_ids"]})
REPORT["test6_synergy"] = {
    "questions": len(qa),
    "cold_memory": {
        "mean_depth": cold["mean_depth"],
        "accuracy_pct": round(100 * cold_acc, 4),
        "latency_p50_ms": cold["p50_ms"],
        "latency_p95_ms": cold["p95_ms"],
    },
    "adaptation": {
        "pairs": len(qa), "seconds": round(adapt_s, 4),
        "fast_state_bytes": warm_state.state_bytes(), "updates": adapt["updates"],
    },
    "warm_memory": {
        "mean_depth": warm["mean_depth"],
        "accuracy_pct": round(100 * warm_acc, 4),
        "latency_p50_ms": warm["p50_ms"],
        "latency_p95_ms": warm["p95_ms"],
    },
    "mean_depth_delta_warm_minus_cold": round(
        (warm["mean_depth"] or 0) - (cold["mean_depth"] or 0), 4
    ),
    "latency_delta_p50_ms": round(warm["p50_ms"] - cold["p50_ms"], 4),
    "prompt_contains_facts": False,
}

(OUT_DIR / "report.json").write_text(
    json.dumps(REPORT, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"REPORT_ID={RUN_ID}")
print(f"REPORT_PATH={OUT_DIR / 'report.json'}")
