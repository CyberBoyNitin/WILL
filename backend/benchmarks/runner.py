"""Real benchmark suites for WILL.

Every number in a report is produced by executing the model on data.  MAC/FLOP
values are analytic estimates and are labelled as such; no energy or efficiency
claim is made without a power measurement.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.config import AppConfig, ModelConfig, TrainingConfig
from backend.engine.metrics import (
    count_parameters,
    estimate_forward_macs,
    flops_from_macs,
    hardware_info,
    process_rss_mb,
)
from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer
from backend.memory.bm25 import BM25Index
from backend.memory.document_ingestion import DocumentIngestion
from backend.memory.fast_memory import FastWeightMemory
from backend.memory.memory_manager import MemoryManager
from backend.memory.memory_retriever import MemoryRetriever
from backend.training.checkpoints import (
    list_checkpoints,
    load_model_from_checkpoint,
    resolve_checkpoint,
)
from backend.training.datasets import collate_lm, make_dataset
from backend.training.trainer import evaluate_generation, evaluate_loss, resolve_device

ProgressCallback = Callable[[dict[str, Any]], None]

SUITES = ("reasoning", "adaptive_compute", "memory_recall", "inference", "ablation")

DEFAULTS: dict[str, Any] = {
    "task": None,
    "split": "test",
    "limit": None,
    "depths": None,
    "modes": None,
    "batch_size": 16,
    "max_examples": 5,
    "max_new_tokens": 12,
    "repeats": 3,
    "prompt_lengths": [32, 64, 128],
    "halt_epsilon": None,
    "memory_records": 20,
    "adapt_rounds": 20,
    "device": "auto",
    "top_k": 5,
    "seed": None,
    "checkpoint_id": None,
}


class BenchmarkError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _resolve_checkpoint(app_cfg: AppConfig, checkpoint_id: str | None) -> str | None:
    if checkpoint_id:
        return checkpoint_id
    metas = list_checkpoints(app_cfg.server.checkpoint_dir)
    return metas[0]["checkpoint_id"] if metas else None


def load_bundle(
    app_cfg: AppConfig,
    checkpoint_id: str | None,
    device: torch.device,
) -> tuple[WILLModel, dict[str, Any], str | None]:
    cid = _resolve_checkpoint(app_cfg, checkpoint_id)
    if cid:
        path = resolve_checkpoint(app_cfg.server.checkpoint_dir, cid)
        model, _cfg, payload = load_model_from_checkpoint(path, device=device)
        return model, payload, cid
    model = WILLModel(ModelConfig())
    model.to(device)
    model.eval()
    payload = {
        "model_config": ModelConfig().model_dump(),
        "training_config": TrainingConfig().model_dump(),
        "metrics": {},
        "task": "arith_add",
    }
    return model, payload, None


def _eval_dataset(
    payload: dict[str, Any],
    tokenizer: ByteTokenizer,
    options: dict[str, Any],
) -> tuple[Any, str, dict[str, Any]]:
    tcfg = dict(payload.get("training_config") or {})
    task = options.get("task") or tcfg.get("task") or "arith_add"
    split = options.get("split") or "test"
    seed = options.get("seed") if options.get("seed") is not None else tcfg.get("seed", 1337)
    limit = options.get("limit")
    size = int(limit or tcfg.get(f"{split}_size", 100))
    min_c = int(tcfg.get("min_complexity", 1))
    max_c = int(tcfg.get("max_complexity", 2))
    ds = make_dataset(
        task=task,
        split=split,
        size=max(1, size),
        seed=int(seed),
        tokenizer=tokenizer,
        max_seq_len=int(payload["model_config"]["max_seq_len"]),
        min_complexity=min_c,
        max_complexity=max_c,
    )
    meta = {
        "task": task,
        "split": split,
        "size": len(ds),
        "seed": int(seed),
        "min_complexity": min_c,
        "max_complexity": max_c,
    }
    return ds, task, meta


def _configs(
    model: WILLModel,
    options: dict[str, Any],
    include_adaptive: bool,
) -> list[tuple[str, int]]:
    max_steps = model.cfg.max_steps
    depths = options.get("depths") or list(range(1, max_steps + 1))
    depths = sorted({max(1, min(int(d), max_steps)) for d in depths})
    configs = [("fixed_depth", d) for d in depths]
    modes = options.get("modes")
    if include_adaptive:
        allowed = set(modes or ["adaptive_soft", "adaptive_hard"])
        for mode in ("adaptive_soft", "adaptive_hard"):
            if mode in allowed:
                configs.append((mode, max_steps))
    return configs


def _evaluate_config(
    model: WILLModel,
    ds: Any,
    tokenizer: ByteTokenizer,
    device: torch.device,
    mode: str,
    max_steps: int,
    options: dict[str, Any],
) -> dict[str, Any]:
    loader = DataLoader(ds, batch_size=int(options["batch_size"]), collate_fn=collate_lm)
    t0 = time.perf_counter()
    loss = evaluate_loss(
        model,
        loader,
        device,
        mode,
        max_steps,
        halt_epsilon=options.get("halt_epsilon"),
    )
    loss_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    gen = evaluate_generation(
        model,
        ds,
        tokenizer,
        device,
        mode,
        max_steps,
        batch_size=int(options["batch_size"]),
        max_new_tokens=int(options["max_new_tokens"]),
        max_examples=int(options["max_examples"]),
        halt_epsilon=options.get("halt_epsilon"),
    )
    gen_seconds = time.perf_counter() - t1
    row = {
        "mode": mode,
        "max_steps": max_steps,
        "accuracy": gen["accuracy"],
        "correct": gen["correct"],
        "count": gen["count"],
        "loss": loss["loss"],
        "perplexity": loss["perplexity"],
        "eval_mean_depth": loss["mean_depth"],
        "generation_mean_depth": gen["mean_depth"],
        "token_evaluations": loss["token_evaluations"],
        "generation_token_evaluations": gen["token_evaluations"],
        "estimated_macs": loss["estimated_macs"],
        "estimated_flops": flops_from_macs(loss["estimated_macs"]),
        "generation_estimated_macs": gen["estimated_macs"],
        "loss_seconds": loss_seconds,
        "generation_seconds": gen_seconds,
        "generation_examples": gen["examples"],
    }
    return row


def _add_compute_deltas(rows: list[dict[str, Any]]) -> None:
    fixed = [r for r in rows if r["mode"] == "fixed_depth"]
    if not fixed:
        return
    baseline = max(fixed, key=lambda r: r["max_steps"])
    base_evals = max(1, baseline["token_evaluations"])
    base_gen_evals = max(1, baseline["generation_token_evaluations"])
    base_seconds = max(1e-9, baseline["generation_seconds"])
    for row in rows:
        row["compute_vs_max_fixed_pct"] = round(
            100.0 * (1.0 - row["token_evaluations"] / base_evals), 2
        )
        row["generation_compute_vs_max_fixed_pct"] = round(
            100.0 * (1.0 - row["generation_token_evaluations"] / base_gen_evals), 2
        )
        row["accuracy_delta_vs_max_fixed"] = (
            None
            if row["accuracy"] is None
            else round(float(row["accuracy"]) - float(baseline["accuracy"]), 4)
        )
        row["generation_speedup_vs_max_fixed"] = round(
            base_seconds / max(1e-9, row["generation_seconds"]), 3
        )
        row["accuracy_per_million_macs"] = (
            None
            if not row["generation_estimated_macs"]
            else round(
                float(row["accuracy"]) / (row["generation_estimated_macs"] / 1e6), 4
            )
        )


# ----------------------------------------------------------------------
# Suites
# ----------------------------------------------------------------------
def run_reasoning(
    app_cfg: AppConfig,
    options: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    progress_cb: ProgressCallback | None,
    stop_event,
) -> dict[str, Any]:
    model, payload, cid = load_bundle(app_cfg, options.get("checkpoint_id"), device)
    tokenizer = ByteTokenizer()
    ds, task, ds_meta = _eval_dataset(payload, tokenizer, options)
    configs = _configs(model, options, include_adaptive=False)
    rows = []
    for i, (mode, depth) in enumerate(configs):
        if stop_event is not None and stop_event.is_set():
            break
        if progress_cb:
            progress_cb(
                {
                    "type": "benchmark_step",
                    "suite": "reasoning",
                    "completed": i,
                    "total": len(configs),
                    "label": f"{mode}@{depth}",
                }
            )
        row = _evaluate_config(model, ds, tokenizer, device, mode, depth, options)
        rows.append(row)
    _add_compute_deltas(rows)
    best = max(rows, key=lambda r: (r["accuracy"] or 0.0)) if rows else None
    return {
        "checkpoint_id": cid,
        "dataset": ds_meta,
        "task": task,
        "results": rows,
        "summary": {
            "best_accuracy": best["accuracy"] if best else None,
            "best_config": f"{best['mode']}@{best['max_steps']}" if best else None,
            "max_steps": model.cfg.max_steps,
            "parameters": count_parameters(model),
        },
    }


def run_adaptive_compute(
    app_cfg: AppConfig,
    options: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    progress_cb: ProgressCallback | None,
    stop_event,
) -> dict[str, Any]:
    model, payload, cid = load_bundle(app_cfg, options.get("checkpoint_id"), device)
    tokenizer = ByteTokenizer()
    ds, task, ds_meta = _eval_dataset(payload, tokenizer, options)
    configs = _configs(model, options, include_adaptive=True)
    rows = []
    for i, (mode, depth) in enumerate(configs):
        if stop_event is not None and stop_event.is_set():
            break
        if progress_cb:
            progress_cb(
                {
                    "type": "benchmark_step",
                    "suite": "adaptive_compute",
                    "completed": i,
                    "total": len(configs),
                    "label": f"{mode}@{depth}",
                }
            )
        rows.append(_evaluate_config(model, ds, tokenizer, device, mode, depth, options))
    _add_compute_deltas(rows)
    fixed = [r for r in rows if r["mode"] == "fixed_depth"]
    adaptive = [r for r in rows if r["mode"] != "fixed_depth"]
    best_fixed = max(fixed, key=lambda r: (r["accuracy"] or 0.0)) if fixed else None
    best_adaptive = (
        max(adaptive, key=lambda r: (r["accuracy"] or 0.0)) if adaptive else None
    )
    return {
        "checkpoint_id": cid,
        "dataset": ds_meta,
        "task": task,
        "results": rows,
        "summary": {
            "best_fixed_accuracy": best_fixed["accuracy"] if best_fixed else None,
            "best_fixed_config": (
                f"fixed_depth@{best_fixed['max_steps']}" if best_fixed else None
            ),
            "best_adaptive_accuracy": (
                best_adaptive["accuracy"] if best_adaptive else None
            ),
            "best_adaptive_config": (
                f"{best_adaptive['mode']}@{best_adaptive['max_steps']}"
                if best_adaptive
                else None
            ),
            "max_steps": model.cfg.max_steps,
            "parameters": count_parameters(model),
        },
    }


def _record_rows(session) -> list[dict[str, Any]]:
    rows = []
    for chunk in session.chunks:
        for rec in chunk.get("records", []):
            value_vec = rec.get("value_vec")
            if isinstance(value_vec, torch.Tensor):
                rows.append(
                    {
                        "key": str(rec.get("key_text", "")),
                        "value": str(rec.get("value_text", "")),
                        "value_vec": value_vec.float(),
                    }
                )
    return rows


def _neural_recall(
    retriever: MemoryRetriever,
    session,
    records: list[tuple[str, str]],
) -> tuple[float, list[dict[str, Any]]]:
    stored = _record_rows(session)
    if not stored:
        return 0.0, []
    correct = 0
    details = []
    for key, expected in records:
        pred = retriever.neural_prediction(session, f"what is the value of {key}?")
        sims = []
        for row in stored:
            vec = F.normalize(row["value_vec"], dim=-1)
            sims.append(float((pred * vec).sum().item()))
        best_idx = max(range(len(stored)), key=lambda i: sims[i])
        got = stored[best_idx]["value"]
        ok = got.strip().lower() == expected.strip().lower()
        correct += int(ok)
        details.append(
            {
                "key": key,
                "expected": expected,
                "predicted": got,
                "correct": ok,
                "score": round(sims[best_idx], 6),
            }
        )
    return correct / len(records), details


def _lexical_recall(
    records: list[tuple[str, str]],
    query_keys: list[str],
) -> tuple[float, list[dict[str, Any]]]:
    index = BM25Index().fit([f"{k} {v}" for k, v in records])
    correct = 0
    details = []
    for key, expected in records:
        scores = index.scores(key)
        best_idx = max(range(len(records)), key=lambda i: scores[i]) if scores else -1
        got = records[best_idx][1] if best_idx >= 0 else ""
        ok = got.strip().lower() == expected.strip().lower()
        correct += int(ok)
        details.append(
            {
                "key": key,
                "expected": expected,
                "predicted": got,
                "correct": ok,
                "score": round(scores[best_idx], 6) if best_idx >= 0 else None,
            }
        )
    return correct / max(1, len(records)), details


def run_memory_recall(
    app_cfg: AppConfig,
    options: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    progress_cb: ProgressCallback | None,
    stop_event,
) -> dict[str, Any]:
    model, payload, cid = load_bundle(app_cfg, options.get("checkpoint_id"), device)
    tokenizer = ByteTokenizer()
    n = max(2, int(options.get("memory_records") or 20))
    records = [(f"K{i:02d}", f"V{100 + i}") for i in range(n)]
    text = " ".join(f"{k}={v}." for k, v in records)

    session_dir = run_dir / "memory_sessions"
    manager = MemoryManager(
        FastWeightMemory(model.cfg),
        app_cfg.memory,
        session_dir,
        device=device,
    )
    session = manager.create_session("benchmark-recall")
    ingestion = DocumentIngestion(model, tokenizer, app_cfg.memory, device=device)
    retriever = MemoryRetriever(model, tokenizer, app_cfg.memory, device=device)

    t0 = time.perf_counter()
    ingest_stats = ingestion.ingest_text(
        session.session_id, text, manager, filename="recall.txt"
    )
    ingest_seconds = time.perf_counter() - t0
    session = manager.get_session(session.session_id)

    history: list[dict[str, Any]] = []

    def measure(round_idx: int) -> None:
        neural_acc, neural_details = _neural_recall(retriever, session, records)
        lexical_acc, lexical_details = _lexical_recall(records, [k for k, _ in records])
        # Hybrid: normalised neural + lexical scores over the same record set.
        stored = _record_rows(session)
        hybrid_correct = 0
        for key, expected in records:
            pred = retriever.neural_prediction(session, f"what is the value of {key}?")
            neural_sims = [
                float(
                    (
                        pred
                        * F.normalize(row["value_vec"], dim=-1)
                    ).sum().item()
                )
                for row in stored
            ]
            lex_index = BM25Index().fit([f"{k} {v}" for k, v in records])
            lex_scores = lex_index.scores(key)
            # Min-max normalise both signals per query.
            def norm(vals: list[float]) -> list[float]:
                if not vals:
                    return []
                lo, hi = min(vals), max(vals)
                if hi - lo < 1e-9:
                    return [0.0 for _ in vals]
                return [(v - lo) / (hi - lo) for v in vals]

            ns = norm(neural_sims)
            ls = norm(lex_scores)
            combined = [
                app_cfg.memory.lexical_weight * ls[i]
                + app_cfg.memory.neural_weight * ns[i]
                for i in range(min(len(ns), len(ls), len(stored)))
            ]
            if combined:
                best_idx = max(range(len(combined)), key=lambda i: combined[i])
                got = stored[best_idx]["value"]
                hybrid_correct += int(
                    got.strip().lower() == expected.strip().lower()
                )
        hybrid_acc = hybrid_correct / max(1, len(records))
        history.append(
            {
                "adapt_round": round_idx,
                "neural_recall_at_1": round(neural_acc, 4),
                "lexical_recall_at_1": round(lexical_acc, 4),
                "hybrid_recall_at_1": round(hybrid_acc, 4),
                "session_updates": session.updates,
                "state_updates": session.state.updates,
                "adapt_seconds": round(session.adapt_seconds, 4),
                "state_bytes": session.state_bytes(),
                "neural_details": neural_details[:5],
                "lexical_details": lexical_details[:5],
            }
        )

    measure(0)
    pairs = _record_rows(session)
    rounds_executed = 0
    if pairs:
        keys = torch.stack(
            [
                next(
                    rec["key_vec"].float()
                    for chunk in session.chunks
                    for rec in chunk.get("records", [])
                    if str(rec.get("value_text", "")) == row["value"]
                )
                for row in pairs
            ]
        ).to(device)
        values = torch.stack([row["value_vec"] for row in pairs]).to(device)
        rounds = max(0, int(options.get("adapt_rounds") or 20))
        for r in range(1, rounds + 1):
            if stop_event is not None and stop_event.is_set():
                break
            manager.adapt_pairs(session.session_id, keys, values)
            rounds_executed = r
            session = manager.get_session(session.session_id)
            if r == 1 or r in (5, 10, 20, 40, rounds):
                measure(r)
                if progress_cb:
                    progress_cb(
                        {
                            "type": "benchmark_step",
                            "suite": "memory_recall",
                            "completed": r,
                            "total": rounds,
                            "label": f"adapt_round={r}",
                        }
                    )
    final = history[-1] if history else {}
    return {
        "checkpoint_id": cid,
        "dataset": {
            "kind": "synthetic_key_value",
            "records": n,
            "text_preview": text[:240],
        },
        "ingest": ingest_stats,
        "ingest_seconds": round(ingest_seconds, 4),
        "results": history,
        "summary": {
            "neural_recall_at_1": final.get("neural_recall_at_1"),
            "lexical_recall_at_1": final.get("lexical_recall_at_1"),
            "hybrid_recall_at_1": final.get("hybrid_recall_at_1"),
            "records": n,
            "adapt_rounds": rounds_executed,
            "fast_state_bytes": final.get("state_bytes"),
            "slow_parameters": count_parameters(model),
        },
    }


def _inference_prompt(tokenizer: ByteTokenizer, length: int) -> torch.Tensor:
    text = (
        "records: "
        + " ".join(f"r{i} alpha_{1000 + i}." for i in range(40))
        + " question: what is the access code?"
    )
    ids = tokenizer.encode(text, add_bos=True, max_len=max(8, length))
    return torch.tensor([ids], dtype=torch.long)


def run_inference(
    app_cfg: AppConfig,
    options: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    progress_cb: ProgressCallback | None,
    stop_event,
) -> dict[str, Any]:
    model, payload, cid = load_bundle(app_cfg, options.get("checkpoint_id"), device)
    tokenizer = ByteTokenizer()
    configs = _configs(model, options, include_adaptive=True)
    lengths = [int(x) for x in options.get("prompt_lengths") or [32, 64, 128]]
    repeats = max(1, int(options.get("repeats") or 3))
    max_new = max(1, int(options.get("max_new_tokens") or 8))
    rows = []
    total = len(configs) * len(lengths)
    done = 0
    for mode, depth in configs:
        for length in lengths:
            if stop_event is not None and stop_event.is_set():
                break
            ids = _inference_prompt(tokenizer, length).to(device)
            # Warmup is excluded from latency statistics.
            model.generate(
                ids,
                tokenizer,
                mode=mode,
                max_steps=depth,
                max_new_tokens=max_new,
            )
            latencies = []
            last = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                last = model.generate(
                    ids,
                    tokenizer,
                    mode=mode,
                    max_steps=depth,
                    max_new_tokens=max_new,
                )
                latencies.append((time.perf_counter() - t0) * 1000.0)
            latencies_sorted = sorted(latencies)
            p50 = statistics.median(latencies)
            p95 = latencies_sorted[min(len(latencies_sorted) - 1, math.ceil(0.95 * len(latencies_sorted)) - 1)]
            row = {
                "mode": mode,
                "max_steps": depth,
                "prompt_tokens": int(ids.shape[1]),
                "max_new_tokens": max_new,
                "generated_tokens": last["generated_tokens"] if last else 0,
                "mean_depth": last["mean_depth"] if last else None,
                "token_evaluations": last["token_evaluations"] if last else None,
                "estimated_macs": (
                    estimate_forward_macs(
                        model.cfg,
                        int(ids.shape[1]) + max_new,
                        int(last["token_evaluations"]),
                    )
                    if last
                    else None
                ),
                "latency_ms_p50": round(p50, 3),
                "latency_ms_p95": round(p95, 3),
                "tokens_per_second": round(
                    (last["generated_tokens"] / (p50 / 1000.0)) if last and p50 > 0 else 0.0,
                    3,
                ),
                "repeats": repeats,
                "process_rss_mb": round(process_rss_mb(), 1),
            }
            rows.append(row)
            done += 1
            if progress_cb:
                progress_cb(
                    {
                        "type": "benchmark_step",
                        "suite": "inference",
                        "completed": done,
                        "total": total,
                        "label": f"{mode}@{depth} len={length}",
                    }
                )
    return {
        "checkpoint_id": cid,
        "results": rows,
        "summary": {
            "configs": len(configs),
            "prompt_lengths": lengths,
            "repeats": repeats,
            "fastest_p50_ms": min((r["latency_ms_p50"] for r in rows), default=None),
            "slowest_p50_ms": max((r["latency_ms_p50"] for r in rows), default=None),
            "parameters": count_parameters(model),
        },
    }


def run_ablation(
    app_cfg: AppConfig,
    options: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    progress_cb: ProgressCallback | None,
    stop_event,
) -> dict[str, Any]:
    # Execution-mode ablation is the adaptive-compute table; memory-subsystem
    # ablation is the real fast-weight recall table.  Both are measured, not
    # simulated.
    compute = run_adaptive_compute(
        app_cfg, options, device, run_dir, progress_cb, stop_event
    )
    memory = run_memory_recall(
        app_cfg, options, device, run_dir, progress_cb, stop_event
    )
    return {
        "checkpoint_id": compute.get("checkpoint_id"),
        "dataset": compute.get("dataset"),
        "results": compute.get("results", []),
        "memory_recall": memory.get("results", []),
        "summary": {
            "execution_mode": compute.get("summary", {}),
            "memory_subsystem": memory.get("summary", {}),
        },
        "sections": {
            "execution_mode": compute,
            "memory_subsystem": memory,
        },
    }


RUNNERS = {
    "reasoning": run_reasoning,
    "adaptive_compute": run_adaptive_compute,
    "memory_recall": run_memory_recall,
    "inference": run_inference,
    "ablation": run_ablation,
}


# ----------------------------------------------------------------------
# Report writing
# ----------------------------------------------------------------------
def _flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if key in ("generation_examples", "neural_details", "lexical_details"):
                continue
            if isinstance(value, (dict, list)):
                out[key] = json.dumps(value, ensure_ascii=False)
            else:
                out[key] = value
        flat.append(out)
    return flat


def write_report(run_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = report.get("results") or []
    csv_path = run_dir / "summary.csv"
    flat = _flatten_rows(rows)
    if flat:
        fieldnames = sorted({key for row in flat for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat)
    else:
        csv_path.write_text("", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path)}


def run_benchmark_suite(
    app_cfg: AppConfig,
    suite: str,
    options: dict[str, Any] | None = None,
    progress_cb: ProgressCallback | None = None,
    stop_event=None,
) -> dict[str, Any]:
    if suite not in RUNNERS:
        raise BenchmarkError(f"unknown benchmark suite: {suite}")
    opts = dict(DEFAULTS)
    opts.update(options or {})
    device = resolve_device(opts.get("device"))
    run_id = f"{suite}-{int(time.time())}"
    run_dir = Path(app_cfg.server.experiment_dir) / "benchmarks" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = RUNNERS[suite](
        app_cfg, opts, device, run_dir, progress_cb, stop_event
    )
    report = {
        "format": "will-benchmark-v1",
        "run_id": run_id,
        "suite": suite,
        "created_at": time.time(),
        "device": str(device),
        "hardware": hardware_info(),
        "options": opts,
        "duration_seconds": round(time.perf_counter() - start, 3),
        "cancelled": bool(stop_event is not None and stop_event.is_set()),
        "notes": [
            "All accuracy/loss/recall values are measured on generated or held-out data.",
            "MAC/FLOP values are analytic estimates, not power measurements.",
            "No energy savings are claimed.",
        ],
        **result,
    }
    artifacts = write_report(run_dir, report)
    report["artifacts"] = artifacts
    # Keep the artifact paths in the JSON as well.
    Path(artifacts["json"]).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def list_reports(experiment_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(experiment_dir) / "benchmarks"
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*/report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(
            {
                "run_id": report.get("run_id", path.parent.name),
                "suite": report.get("suite"),
                "created_at": report.get("created_at"),
                "duration_seconds": report.get("duration_seconds"),
                "device": report.get("device"),
                "checkpoint_id": report.get("checkpoint_id"),
                "summary": report.get("summary", {}),
                "path": str(path),
            }
        )
    out.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return out


def load_report(experiment_dir: str | Path, run_id: str) -> dict[str, Any]:
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for ch in run_id):
        raise ValueError("invalid benchmark run id")
    path = Path(experiment_dir) / "benchmarks" / run_id / "report.json"
    if not path.exists():
        raise FileNotFoundError("benchmark report not found")
    return json.loads(path.read_text(encoding="utf-8"))
