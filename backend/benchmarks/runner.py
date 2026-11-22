"""Real benchmark suites for WILL.

Every number in a report is produced by executing the model on data.  MAC/FLOP
values are analytic estimates and are labelled as such; no energy or efficiency
claim is made without a power measurement.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from backend.config import (
    SCALING_PRESETS,
    AppConfig,
    ModelConfig,
    TrainingConfig,
    scaling_model_config,
)
from backend.engine.compilation import maybe_compile
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
from backend.training.datasets import collate_lm, make_dataset, make_mixed_dataset
from backend.training.losses import total_loss
from backend.training.precision import (
    PrecisionPlan,
    autocast_context,
    resolve_precision,
)
from backend.training.trainer import evaluate_generation, evaluate_loss, resolve_device

ProgressCallback = Callable[[dict[str, Any]], None]

SUITES = (
    "reasoning",
    "adaptive_compute",
    "mixed_difficulty",
    "memory_recall",
    "continual",
    "inference",
    "scaling",
    "ablation",
)

DEFAULTS: dict[str, Any] = {
    "task": None,
    "split": "test",
    "limit": None,
    "depths": None,
    "modes": None,
    "adaptive_max_steps": None,
    "batch_size": 16,
    "max_examples": 5,
    "max_new_tokens": 12,
    "repeats": 3,
    "generation_repeats": 3,
    "prompt_lengths": [32, 64, 128],
    "halt_epsilon": None,
    "memory_records": 20,
    "adapt_rounds": 20,
    "continual_domains": 3,
    "continual_facts_per_domain": 20,
    "consolidation_ridge": None,
    "device": "auto",
    "top_k": 5,
    "seed": None,
    "checkpoint_id": None,
    "mixed_easy_ratio": 0.7,
    "compile_probe": True,
    "presets": None,
    "precisions": None,
    "scaling_seq_len": 32,
    "scaling_batch_size": 1,
    "scaling_steps": 2,
    "scaling_warmup": 1,
    "scaling_repeats": 2,
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
        adaptive_depth = int(
            options.get("adaptive_max_steps")
            or model.cfg.resolved_adaptive_steps()
        )
        adaptive_depth = max(1, min(adaptive_depth, max_steps))
        for mode in ("adaptive_soft", "adaptive_hard"):
            if mode in allowed:
                configs.append((mode, adaptive_depth))
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
    generation_repeats = max(1, int(options.get("generation_repeats") or 1))
    # Warm the cached generation path, then report the median of repeated
    # measurements.  A single cold run is dominated by allocator and CPU
    # frequency noise on the small CPU model.
    evaluate_generation(
        model,
        ds,
        tokenizer,
        device,
        mode,
        max_steps,
        batch_size=int(options["batch_size"]),
        max_new_tokens=int(options["max_new_tokens"]),
        max_examples=0,
        halt_epsilon=options.get("halt_epsilon"),
    )
    generation_times: list[float] = []
    gen = None
    for _ in range(generation_repeats):
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
        generation_times.append(time.perf_counter() - t1)
    assert gen is not None
    gen_seconds = statistics.median(generation_times)
    gen_estimated_macs = int(gen.get("estimated_macs") or 0)
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
        "generation_estimated_macs": gen_estimated_macs,
        "loss_seconds": loss_seconds,
        "generation_seconds": gen_seconds,
        "generation_seconds_all": [round(t, 6) for t in generation_times],
        "generation_repeats": generation_repeats,
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
            if not row.get("generation_estimated_macs")
            else round(
                float(row["accuracy"]) / (row["generation_estimated_macs"] / 1e6), 4
            )
        )
        # Per-token evaluation counts are B*T*steps in fixed depth; compute
        # savings relative to the best fixed-depth baseline.
        row["generation_token_evaluations_per_step"] = round(
            row["generation_token_evaluations"] / max(1, row["max_steps"]), 2
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


def _complexity_strata(ds) -> dict[int, Subset]:
    levels = sorted({int(item["complexity"]) for item in ds.items})
    return {
        level: Subset(
            ds,
            [
                index
                for index, item in enumerate(ds.items)
                if int(item["complexity"]) == level
            ],
        )
        for level in levels
    }


def run_mixed_difficulty(
    app_cfg: AppConfig,
    options: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    progress_cb: ProgressCallback | None,
    stop_event,
) -> dict[str, Any]:
    """Evaluate a 70/30 trivial/hard split overall and per stratum."""
    model, payload, cid = load_bundle(app_cfg, options.get("checkpoint_id"), device)
    tokenizer = ByteTokenizer()
    tcfg = dict(payload.get("training_config") or {})
    task = options.get("task") or tcfg.get("task") or "arith_add"
    split = options.get("split") or "test"
    seed = (
        options.get("seed")
        if options.get("seed") is not None
        else tcfg.get("seed", 1337)
    )
    size = max(1, int(options.get("limit") or tcfg.get(f"{split}_size", 100)))
    easy_ratio = float(options.get("mixed_easy_ratio") or 0.7)
    ds = make_mixed_dataset(
        task=task,
        split=split,
        size=size,
        seed=int(seed),
        tokenizer=tokenizer,
        max_seq_len=int(payload["model_config"]["max_seq_len"]),
        easy_ratio=easy_ratio,
        easy_complexity=1,
        hard_complexity=2,
    )
    strata = _complexity_strata(ds)
    stratum_labels = {1: "trivial", 2: "hard"}
    config_options = dict(options)
    if not config_options.get("depths"):
        config_options["depths"] = [model.cfg.max_steps]
    configs = _configs(model, config_options, include_adaptive=True)
    rows: list[dict[str, Any]] = []
    total_steps = max(1, len(configs) * (1 + len(strata)))
    done = 0
    for mode, depth in configs:
        if stop_event is not None and stop_event.is_set():
            break
        row = _evaluate_config(model, ds, tokenizer, device, mode, depth, options)
        row["strata"] = {}
        for level, subset in strata.items():
            if stop_event is not None and stop_event.is_set():
                break
            stratum = _evaluate_config(
                model,
                subset,
                tokenizer,
                device,
                mode,
                depth,
                {**options, "max_examples": 0},
            )
            label = stratum_labels.get(level, f"complexity_{level}")
            row["strata"][label] = {
                "complexity": level,
                "count": stratum["count"],
                "accuracy": stratum["accuracy"],
                "correct": stratum["correct"],
                "loss": stratum["loss"],
                "perplexity": stratum["perplexity"],
                "generation_mean_depth": stratum["generation_mean_depth"],
                "generation_token_evaluations": stratum[
                    "generation_token_evaluations"
                ],
                "generation_seconds": stratum["generation_seconds"],
            }
        trivial = row["strata"].get("trivial", {}).get("accuracy")
        hard = row["strata"].get("hard", {}).get("accuracy")
        row["trivial_accuracy"] = trivial
        row["hard_accuracy"] = hard
        row["hard_minus_trivial_accuracy"] = (
            None
            if trivial is None or hard is None
            else round(float(hard) - float(trivial), 6)
        )
        rows.append(row)
        done += 1 + len(strata)
        if progress_cb:
            progress_cb(
                {
                    "type": "benchmark_step",
                    "suite": "mixed_difficulty",
                    "completed": min(done, total_steps),
                    "total": total_steps,
                    "label": f"{mode}@{depth}",
                }
            )
    best = max(rows, key=lambda r: (r["accuracy"] or 0.0)) if rows else None
    hard_rows = [r for r in rows if r.get("hard_accuracy") is not None]
    best_hard = (
        max(hard_rows, key=lambda r: (r["hard_accuracy"] or 0.0))
        if hard_rows
        else None
    )
    observed_easy = ds.complexity_counts.get(1, 0) / max(1, len(ds))
    return {
        "checkpoint_id": cid,
        "dataset": {
            "task": task,
            "split": split,
            "size": len(ds),
            "seed": int(seed),
            "easy_ratio_requested": easy_ratio,
            "easy_ratio_observed": round(observed_easy, 6),
            "complexity_counts": {
                str(level): count
                for level, count in sorted(ds.complexity_counts.items())
            },
        },
        "task": task,
        "results": rows,
        "summary": {
            "best_accuracy": best["accuracy"] if best else None,
            "best_config": f"{best['mode']}@{best['max_steps']}" if best else None,
            "best_hard_accuracy": best_hard["hard_accuracy"] if best_hard else None,
            "best_hard_config": (
                f"{best_hard['mode']}@{best_hard['max_steps']}"
                if best_hard
                else None
            ),
            "easy_ratio_requested": easy_ratio,
            "easy_ratio_observed": round(observed_easy, 6),
            "composition_ok": abs(observed_easy - easy_ratio)
            <= 1.0 / max(1, len(ds)) + 1e-9,
            "strata": {
                stratum_labels.get(level, f"complexity_{level}"): len(subset)
                for level, subset in strata.items()
            },
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


def _hybrid_recall(
    retriever: MemoryRetriever,
    session,
    records: list[tuple[str, str]],
    cfg,
) -> tuple[float, list[dict[str, Any]]]:
    stored = _record_rows(session)
    if not stored:
        return 0.0, []
    # Fit on the same row order as the neural scores.  Fitting on the query
    # list and zipping it with stored rows silently misaligns lexical and
    # neural candidates whenever a chunk contains duplicate records.
    lexical_index = BM25Index().fit(
        [f"{row.get('key_text', '')} {row.get('value_text', '')}" for row in stored]
    )
    correct = 0
    details = []
    for key, expected in records:
        pred = retriever.neural_prediction(session, f"what is the value of {key}?")
        neural_sims = [
            float((pred * F.normalize(row["value_vec"], dim=-1)).sum().item())
            for row in stored
        ]
        lex_scores = lexical_index.scores(key)

        def norm(values: list[float]) -> list[float]:
            if not values:
                return []
            lo, hi = min(values), max(values)
            if hi - lo < 1e-9:
                return [0.0 for _ in values]
            return [(v - lo) / (hi - lo) for v in values]

        ns = norm(neural_sims)
        ls = norm(lex_scores)
        combined = [
            cfg.lexical_weight * ls[i] + cfg.neural_weight * ns[i]
            for i in range(min(len(ns), len(ls), len(stored)))
        ]
        best_idx = max(range(len(combined)), key=lambda i: combined[i])
        got = stored[best_idx]["value"]
        ok = got.strip().lower() == expected.strip().lower()
        correct += int(ok)
        details.append(
            {
                "key": key,
                "expected": expected,
                "predicted": got,
                "correct": ok,
                "score": round(combined[best_idx], 6),
            }
        )
    return correct / max(1, len(records)), details


def _model_state_hash(model: WILLModel) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _memory_state_hash(session) -> str:
    digest = hashlib.sha256()
    digest.update(str(session.state.version).encode("utf-8"))
    for name, value in sorted(session.state.params.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_continual(
    app_cfg: AppConfig,
    options: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    progress_cb: ProgressCallback | None,
    stop_event,
) -> dict[str, Any]:
    """Sequential 3x20-fact stress test with replay consolidation.

    Each domain is ingested into the same session in order.  After every
    domain, stored key/value pairs are replayed into the bounded fast state
    with a ridge solve.  Recall is measured on every previously seen domain,
    and the slow model weights are hashed before and after to prove that
    continual learning never mutates them.
    """
    model, payload, cid = load_bundle(app_cfg, options.get("checkpoint_id"), device)
    tokenizer = ByteTokenizer()
    domains = max(1, int(options.get("continual_domains") or 3))
    per_domain = max(1, int(options.get("continual_facts_per_domain") or 20))
    ridge = options.get("consolidation_ridge")
    ridge = None if ridge is None else float(ridge)
    seed = int(options.get("seed") or 1337)

    session_dir = run_dir / "memory_sessions"
    manager = MemoryManager(
        FastWeightMemory(model.cfg),
        app_cfg.memory,
        session_dir,
        device=device,
    )
    session = manager.create_session("benchmark-continual")
    ingestion = DocumentIngestion(model, tokenizer, app_cfg.memory, device=device)
    retriever = MemoryRetriever(model, tokenizer, app_cfg.memory, device=device)
    slow_before = _model_state_hash(model)

    domain_records: list[list[tuple[str, str]]] = []
    history: list[dict[str, Any]] = []
    ingest_history: list[dict[str, Any]] = []
    domain_recall_history: list[list[dict[str, Any]]] = []
    seen: list[tuple[str, str]] = []
    max_neural_drop = 0.0
    max_hybrid_drop = 0.0
    total = domains
    for domain in range(domains):
        if stop_event is not None and stop_event.is_set():
            break
        records = [
            (
                f"K{domain * per_domain + i:02d}",
                f"V{100 + domain * per_domain + i}",
            )
            for i in range(per_domain)
        ]
        domain_records.append(records)
        seen.extend(records)
        text = " ".join(f"{key}={value}." for key, value in records)
        t0 = time.perf_counter()
        ingest = ingestion.ingest_text(
            session.session_id,
            text,
            manager,
            filename=f"domain_{domain:02d}_{seed}.txt",
        )
        ingest["seconds"] = round(time.perf_counter() - t0, 4)
        ingest_history.append(ingest)
        session = manager.get_session(session.session_id)

        consolidation = manager.consolidate_session(
            session.session_id, ridge=ridge
        )
        session = manager.get_session(session.session_id)
        neural, neural_details = _neural_recall(retriever, session, seen)
        lexical, lexical_details = _lexical_recall(
            seen, [key for key, _ in seen]
        )
        hybrid, hybrid_details = _hybrid_recall(
            retriever, session, seen, app_cfg.memory
        )
        per_domain_metrics = []
        for d in range(domain + 1):
            d_records = domain_records[d]
            d_neural, _ = _neural_recall(retriever, session, d_records)
            d_lexical, _ = _lexical_recall(
                d_records, [key for key, _ in d_records]
            )
            d_hybrid, _ = _hybrid_recall(
                retriever, session, d_records, app_cfg.memory
            )
            per_domain_metrics.append(
                {
                    "domain": d,
                    "facts": len(d_records),
                    "neural_recall_at_1": round(d_neural, 4),
                    "lexical_recall_at_1": round(d_lexical, 4),
                    "hybrid_recall_at_1": round(d_hybrid, 4),
                }
            )
            if d == domain:
                domain_recall_history.append([])
            domain_recall_history[d].append(
                {
                    "after_domain": domain,
                    "neural": d_neural,
                    "hybrid": d_hybrid,
                }
            )
        for d in range(domain):
            first = domain_recall_history[d][0]
            current = domain_recall_history[d][-1]
            max_neural_drop = max(
                max_neural_drop, max(0.0, first["neural"] - current["neural"])
            )
            max_hybrid_drop = max(
                max_hybrid_drop, max(0.0, first["hybrid"] - current["hybrid"])
            )
        row = {
            "stage": domain + 1,
            "domain": domain,
            "facts_seen": len(seen),
            "domain_facts": len(records),
            "neural_recall_at_1": round(neural, 4),
            "lexical_recall_at_1": round(lexical, 4),
            "hybrid_recall_at_1": round(hybrid, 4),
            "per_domain": per_domain_metrics,
            "max_neural_forgetting_pct": round(100.0 * max_neural_drop, 4),
            "max_hybrid_forgetting_pct": round(100.0 * max_hybrid_drop, 4),
            "consolidation": consolidation,
            "state_bytes": session.state_bytes(),
            "memory_state_bytes": session.state.state_bytes(),
            "weight_norm": round(session.state.weight_norm(), 6),
            "state_version": session.state.version,
            "neural_details": neural_details[:5],
            "lexical_details": lexical_details[:5],
            "hybrid_details": hybrid_details[:5],
        }
        history.append(row)
        if progress_cb:
            progress_cb(
                {
                    "type": "benchmark_step",
                    "suite": "continual",
                    "completed": domain + 1,
                    "total": total,
                    "label": f"domain={domain + 1} facts={len(seen)}",
                }
            )

    slow_after = _model_state_hash(model)
    slow_unchanged = slow_before == slow_after
    final = history[-1] if history else {}
    save_load_exact = False
    reload_recall = None
    reload_state_hash = None
    state_hash_before = None
    if history:
        session = manager.get_session(session.session_id)
        state_hash_before = _memory_state_hash(session)
        manager.save_session(session.session_id)
        reloaded = manager.reload_session(session.session_id)
        reload_state_hash = _memory_state_hash(reloaded)
        reload_neural, _ = _neural_recall(retriever, reloaded, seen)
        reload_hybrid, _ = _hybrid_recall(
            retriever, reloaded, seen, app_cfg.memory
        )
        reload_recall = {
            "neural_recall_at_1": round(reload_neural, 4),
            "hybrid_recall_at_1": round(reload_hybrid, 4),
        }
        save_load_exact = (
            state_hash_before == reload_state_hash
            and abs(reload_neural - (final.get("neural_recall_at_1") or 0.0))
            < 1e-9
            and abs(reload_hybrid - (final.get("hybrid_recall_at_1") or 0.0))
            < 1e-9
        )
    return {
        "checkpoint_id": cid,
        "dataset": {
            "kind": "sequential_key_value",
            "domains": domains,
            "facts_per_domain": per_domain,
            "facts_total": domains * per_domain,
            "seed": seed,
            "preview": [
                [f"{k}={v}" for k, v in domain[:3]]
                for domain in domain_records[:3]
            ],
        },
        "ingest": ingest_history,
        "results": history,
        "save_load": {
            "exact": save_load_exact,
            "state_hash_before": state_hash_before,
            "state_hash_after": reload_state_hash,
            "recall": reload_recall,
        },
        "summary": {
            "facts_total": domains * per_domain,
            "domains": domains,
            "facts_per_domain": per_domain,
            "neural_recall_at_1": final.get("neural_recall_at_1"),
            "lexical_recall_at_1": final.get("lexical_recall_at_1"),
            "hybrid_recall_at_1": final.get("hybrid_recall_at_1"),
            "max_neural_forgetting_pct": round(100.0 * max_neural_drop, 4),
            "max_hybrid_forgetting_pct": round(100.0 * max_hybrid_drop, 4),
            "catastrophic_forgetting_ok": (
                100.0 * max(max_neural_drop, max_hybrid_drop) < 1.0
            ),
            "slow_weights_unchanged": slow_unchanged,
            "slow_weights_hash_before": slow_before,
            "slow_weights_hash_after": slow_after,
            "save_load_exact": save_load_exact,
            "state_version": final.get("state_version"),
            "consolidations": (
                final.get("consolidation", {}).get("consolidations")
                if final
                else None
            ),
            "fast_state_bytes": final.get("state_bytes"),
            "fast_parameters": session.parameter_count() if history else 0,
            "slow_parameters": count_parameters(model),
            "ridge": ridge,
        },
    }


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

    ingest_stats = {"chars_ingested": 0, "chunks_added": 0, "updates": 0, "seconds": 0}

    t0 = time.perf_counter()
    ingest_text_result = ingestion.ingest_text(
        session.session_id, text, manager, filename="recall.txt"
    )
    ingest_seconds = round(time.perf_counter() - t0, 4)
    ingest_stats = {
        "chars_ingested": ingest_text_result.get("chunks_added", 0),
        "chunks_added": ingest_text_result.get("chunks_added", 0),
        "updates": ingest_text_result.get("updates", 0),
        "seconds": ingest_seconds,
    }
    session = manager.get_session(session.session_id)

    # Compute zero-update baseline: neural/lexical/hybrid recall with
    # fast weights freshly ingested but no additional adaptation rounds.
    neural_acc_0, _ = _neural_recall(retriever, session, records)
    lexical_acc_0, _ = _lexical_recall(records, [k for k, _ in records])
    hybrid_acc_0, _ = _hybrid_recall(retriever, session, records, app_cfg.memory)

    history: list[dict[str, Any]] = []
    history.append(
        {
            "adapt_round": 0,
            "neural_recall_at_1": round(neural_acc_0, 4),
            "lexical_recall_at_1": round(lexical_acc_0, 4),
            "hybrid_recall_at_1": round(hybrid_acc_0, 4),
            "session_updates": session.updates,
            "state_updates": session.state.updates,
            "adapt_seconds": round(session.adapt_seconds, 4),
            "state_bytes": session.state_bytes(),
            "neural_details": [],
            "lexical_details": [],
        }
    )

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
                neural_acc, _ = _neural_recall(retriever, session, records)
                lexical_acc, _ = _lexical_recall(records, [k for k, _ in records])
                hybrid_acc, _ = _hybrid_recall(
                    retriever, session, records, app_cfg.memory
                )
                history.append(
                    {
                        "adapt_round": r,
                        "neural_recall_at_1": round(neural_acc, 4),
                        "lexical_recall_at_1": round(lexical_acc, 4),
                        "hybrid_recall_at_1": round(hybrid_acc, 4),
                        "session_updates": session.updates,
                        "state_updates": session.state.updates,
                        "adapt_seconds": round(session.adapt_seconds, 4),
                        "state_bytes": session.state_bytes(),
                        "neural_details": [],
                        "lexical_details": [],
                    }
                )
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


def _compile_probe(
    model: WILLModel,
    tokenizer: ByteTokenizer,
    device: torch.device,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Attempt ``torch.compile`` and verify it against eager, without using it.

    The probe answers one question: can this environment compile the
    full-sequence forward and reproduce eager logits?  The KV-cached generation
    path deliberately stays eager, so a successful probe is not reported as a
    latency result.
    """
    if not options.get("compile_probe"):
        return {
            "requested": False,
            "compiled": False,
            "used": "eager",
            "reason": "compile probe disabled",
            "generation_path": "eager",
            "applied_to_generation": False,
        }
    ids = _inference_prompt(tokenizer, 32).to(device)
    mask = torch.ones_like(ids)
    was_training = model.training
    model.eval()

    def probe(module):
        logits, _ = module(
            ids,
            mask,
            mode="fixed_depth",
            max_steps=model.cfg.resolved_adaptive_steps(),
            memory_enabled=False,
        )
        return logits

    start = time.perf_counter()
    try:
        _, status = maybe_compile(
            model, enabled=True, mode="reduce-overhead", probe=probe
        )
    finally:
        model.train(was_training)
    status["seconds"] = round(time.perf_counter() - start, 3)
    status["applied_to_generation"] = False
    status["note"] = (
        "Probe only: the KV-cached generation path stays eager, so this status "
        "is not a latency claim."
    )
    return status


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
    compilation = _compile_probe(model, tokenizer, device, options)
    configs = _configs(model, options, include_adaptive=True)
    lengths = [int(x) for x in options.get("prompt_lengths") or [32, 64, 128]]
    repeats = max(1, int(options.get("repeats") or 20))
    max_new = max(1, int(options.get("max_new_tokens") or 8))
    rows = []
    total = len(configs) * len(lengths)
    done = 0
    for mode, depth in configs:
        for length in lengths:
            if stop_event is not None and stop_event.is_set():
                break
            ids = _inference_prompt(tokenizer, length).to(device)
            # Warmup.
            with torch.no_grad():
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
                if stop_event is not None and stop_event.is_set():
                    break
                t0 = time.perf_counter()
                torch.cuda.synchronize() if device.type == "cuda" else None
                last = model.generate(
                    ids,
                    tokenizer,
                    mode=mode,
                    max_steps=depth,
                    max_new_tokens=max_new,
                )
                torch.cuda.synchronize() if device.type == "cuda" else None
                latencies.append((time.perf_counter() - t0) * 1000.0)
            latencies_sorted = sorted(latencies)
            p50 = statistics.median(latencies) if latencies else 0.0
            # p95 only reported when there are enough repeats for statistical meaning.
            p95 = round(
                latencies_sorted[
                    min(
                        len(latencies_sorted) - 1,
                        math.ceil(0.95 * len(latencies_sorted)) - 1,
                    )
                ],
                3,
            ) if len(latencies_sorted) >= 5 else None
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
                "latency_ms_p95": p95,
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
            "compilation": compilation,
        },
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _scaling_batch(
    tokenizer: ByteTokenizer, seq_len: int, batch_size: int, seed: int
) -> dict[str, Any]:
    ds = make_dataset(
        "arith_add",
        "train",
        batch_size,
        seed,
        tokenizer,
        seq_len,
        complexity=1,
    )
    return collate_lm([ds[i] for i in range(len(ds))])


def _grad_norm(model: WILLModel) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def _scaling_step(
    model: WILLModel,
    ids: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    plan: PrecisionPlan,
    steps: int,
) -> torch.Tensor:
    with autocast_context(plan):
        logits, stats = model(
            ids,
            mask,
            mode="fixed_depth",
            max_steps=steps,
            memory_enabled=False,
        )
        loss, _ = total_loss(logits, labels, stats, ponder_lambda=0.0)
    loss.backward()
    return loss.detach()


def run_scaling(
    app_cfg: AppConfig,
    options: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    progress_cb: ProgressCallback | None,
    stop_event,
) -> dict[str, Any]:
    """Parameter-matched 15M/50M/125M real forward/backward steps.

    No preset is simulated.  Every row instantiates the model, reports the
    exact parameter count, and executes real training steps in the requested
    precision.  CUDA is reported as unavailable when it is unavailable.
    """
    tokenizer = ByteTokenizer()
    preset_names = list(options.get("presets") or SCALING_PRESETS.keys())
    for name in preset_names:
        if name not in SCALING_PRESETS:
            raise BenchmarkError(f"unknown scaling preset: {name}")
    precisions = list(options.get("precisions") or ["fp32", "bf16", "fp16"])
    seq_len = max(8, int(options.get("scaling_seq_len") or 32))
    batch_size = max(1, int(options.get("scaling_batch_size") or 1))
    recurrent_steps = max(1, int(options.get("scaling_steps") or 2))
    warmup = max(0, int(options.get("scaling_warmup") or 1))
    repeats = max(1, int(options.get("scaling_repeats") or 2))
    seed = int(options.get("seed") or 1337)
    batch = _scaling_batch(tokenizer, seq_len, batch_size, seed)
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["attention_mask"].to(device)
    rows: list[dict[str, Any]] = []
    preset_summary: dict[str, Any] = {}
    total = max(1, len(preset_names) * len(precisions))
    done = 0

    for name in preset_names:
        if stop_event is not None and stop_event.is_set():
            break
        preset = SCALING_PRESETS[name]
        torch.manual_seed(seed)
        model_cfg = scaling_model_config(name)
        model = WILLModel(model_cfg).to(device)
        model.train()
        params = count_parameters(model)
        preset_summary[name] = {
            "target_parameters": int(preset["target_parameters"]),
            "parameters": params,
            "delta_pct": round(
                100.0 * (params - preset["target_parameters"])
                / preset["target_parameters"],
                3,
            ),
            "within_5pct": abs(params - preset["target_parameters"])
            <= 0.05 * preset["target_parameters"],
            "hidden_dim": model_cfg.hidden_dim,
            "layers": model_cfg.context_layers,
            "max_steps": model_cfg.max_steps,
            "parameter_bytes_fp32": params * 4,
            "parameter_bytes_bf16": params * 2,
        }
        for precision in precisions:
            if stop_event is not None and stop_event.is_set():
                break
            base_row: dict[str, Any] = {
                "preset": name,
                "target_parameters": int(preset["target_parameters"]),
                "parameters": params,
                "parameters_millions": round(params / 1e6, 3),
                "device": str(device),
                "seq_len": int(ids.shape[1]),
                "batch_size": int(ids.shape[0]),
                "recurrent_steps": recurrent_steps,
                "warmup_steps": warmup,
                "repeats": repeats,
                "precision": precision,
            }
            try:
                plan = resolve_precision(precision, device)
            except Exception as exc:
                rows.append(
                    {
                        **base_row,
                        "supported": False,
                        "effective_precision": None,
                        "autocast": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                done += 1
                continue
            model.zero_grad(set_to_none=True)
            for _ in range(warmup):
                _scaling_step(model, ids, labels, mask, plan, recurrent_steps)
                model.zero_grad(set_to_none=True)
            times: list[float] = []
            losses: list[float] = []
            grad_norm = 0.0
            for index in range(repeats):
                _sync_device(device)
                start = time.perf_counter()
                loss = _scaling_step(
                    model, ids, labels, mask, plan, recurrent_steps
                )
                _sync_device(device)
                times.append(time.perf_counter() - start)
                losses.append(float(loss.detach().item()))
                if index == repeats - 1:
                    grad_norm = _grad_norm(model)
                model.zero_grad(set_to_none=True)
            finite = all(math.isfinite(value) for value in losses)
            median_seconds = statistics.median(times)
            tokens = int(ids.numel())
            rows.append(
                {
                    **base_row,
                    "supported": True,
                    "effective_precision": plan.effective,
                    "autocast": plan.enabled,
                    "precision_reason": plan.reason,
                    "grad_scaler": plan.grad_scaler,
                    "step_seconds": round(median_seconds, 6),
                    "step_seconds_all": [round(value, 6) for value in times],
                    "tokens_per_second": round(tokens / max(1e-9, median_seconds), 3),
                    "loss": round(losses[-1], 6),
                    "losses_all": [round(value, 6) for value in losses],
                    "finite": finite,
                    "grad_norm": round(grad_norm, 6),
                    "rss_mb": round(process_rss_mb(), 1),
                    "error": None,
                }
            )
            done += 1
            if progress_cb:
                progress_cb(
                    {
                        "type": "benchmark_step",
                        "suite": "scaling",
                        "completed": done,
                        "total": total,
                        "label": f"{name} {precision}",
                    }
                )
        del model
        gc.collect()
        _sync_device(device)

    cuda_available = bool(torch.cuda.is_available())
    supported_rows = [row for row in rows if row.get("supported")]
    return {
        "checkpoint_id": None,
        "dataset": {
            "task": "arith_add",
            "split": "train",
            "seed": seed,
            "seq_len": int(ids.shape[1]),
            "batch_size": int(ids.shape[0]),
            "recurrent_steps": recurrent_steps,
        },
        "task": "arith_add",
        "results": rows,
        "summary": {
            "presets": preset_summary,
            "precisions_requested": precisions,
            "device": str(device),
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "cuda_note": (
                "CUDA is available and was used for these rows."
                if device.type == "cuda"
                else (
                    "CUDA is not available in this PyTorch build; CPU rows only. "
                    "No CUDA speedup is claimed."
                )
            ),
            "all_finite": bool(supported_rows)
            and all(row["finite"] for row in supported_rows),
            "supported_rows": len(supported_rows),
            "unsupported_rows": len(rows) - len(supported_rows),
            "warmup_steps": warmup,
            "repeats": repeats,
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
    "mixed_difficulty": run_mixed_difficulty,
    "memory_recall": run_memory_recall,
    "continual": run_continual,
    "inference": run_inference,
    "scaling": run_scaling,
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
    run_id = f"{suite}-{uuid.uuid4().hex[:12]}"
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
