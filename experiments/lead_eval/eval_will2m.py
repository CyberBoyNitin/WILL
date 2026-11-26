"""Test 3 + Test 6 comparison for WILL checkpoints.

By default this evaluates will-8m-synergy-v1 against will-2m-synergy-v3.
Pass ``--new will-2m-synergy-v2`` to reproduce the published v2 report.

The 2M model uses a deterministic complexity router:
  - one-digit prompts: 1 recurrent step, answer-only target
  - v2 two-digit prompts: 4 recurrent steps, direct answer
  - v3 two-digit prompts: right-to-left carry-scratchpad chain

Test 6 warm uses the sequential answer-slot writer and the high-confidence
memory-to-logit skip.  Test 6 cold uses the normal language-model path.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402
from backend.engine.tokenizer import BOS_ID, SEP_ID, ByteTokenizer  # noqa: E402
from backend.memory.memory_encoder import pool_memory_tokens  # noqa: E402
from backend.memory.slot_memory import (  # noqa: E402
    pad_prompts,
    write_slot_memory,
)
from backend.training.checkpoints import (  # noqa: E402
    load_model_from_checkpoint,
    resolve_checkpoint,
)
from backend.training.datasets import (  # noqa: E402
    make_dataset,
    make_mixed_dataset,
    normalize_answer,
)

CFG = load_config("configs/will_2m.yaml")
TOK = ByteTokenizer()
DEVICE = torch.device("cpu")
MODEL_MAX_LEN = int(CFG.model.max_seq_len)
_parser = argparse.ArgumentParser()
_parser.add_argument("--old", default="will-8m-synergy-v1")
_parser.add_argument("--new", default="will-2m-synergy-v3")
_args = _parser.parse_args()
OLD = _args.old
NEW = _args.new
USE_CHAIN = NEW == "will-2m-synergy-v3"
WARMUP = 5
TIMED = 50
OUT = Path(__file__).resolve().parent / f"will2m_eval-{uuid.uuid4().hex[:12]}"
OUT.mkdir(parents=True, exist_ok=True)


def load(checkpoint_id: str):
    path = resolve_checkpoint(CFG.server.checkpoint_dir, checkpoint_id)
    model, mcfg, payload = load_model_from_checkpoint(path, device=DEVICE)
    model.eval()
    return model, mcfg, payload


def parse_int(text: str, mode: str = "first") -> str:
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return ""
    return numbers[-1] if mode == "last" else numbers[0]


def accuracy(
    generated: list[list[int]], rows: list[dict], mode: str = "first"
) -> float:
    correct = 0
    for ids, row in zip(generated, rows):
        text = TOK.decode(ids, skip_special=True)
        correct += int(
            normalize_answer(parse_int(text, mode))
            == normalize_answer(str(row["answer_text"]))
        )
    return correct / max(1, len(rows))


def timed(
    model,
    prompts: list[list[int]],
    depth: int,
    *,
    memory_state=None,
    max_new_tokens: int,
    batch_size: int | None = 16,
    mode: str = "fixed_depth",
) -> dict:
    """Time repeated generation calls.

    The benchmark suite's default inference batch is 16, so Test 3 uses that
    for a comparable p50.  Test 6 keeps all 30 questions in one call because
    the slot writer and reader must share the same left-padding width.
    """
    width = len(prompts) if batch_size is None else max(1, int(batch_size))
    chunks = [prompts[i : i + width] for i in range(0, len(prompts), width)]
    outputs: list[dict | None] = [None] * len(chunks)

    def run(chunk: list[list[int]]) -> dict:
        return model.generate_batch(
            chunk,
            TOK,
            memory_state=memory_state,
            mode=mode,
            max_steps=depth,
            max_new_tokens=max_new_tokens,
            stop_at_eos=True,
        )

    for _ in range(WARMUP):
        for chunk in chunks:
            run(chunk)
    times = []
    for repeat in range(TIMED):
        index = repeat % len(chunks)
        t0 = time.perf_counter()
        outputs[index] = run(chunks[index])
        times.append(time.perf_counter() - t0)
    for index, chunk in enumerate(chunks):
        if outputs[index] is None:
            outputs[index] = run(chunk)
    times_sorted = sorted(times)
    p95 = times_sorted[
        min(len(times_sorted) - 1, math.ceil(0.95 * len(times_sorted)) - 1)
    ]
    generated: list[list[int]] = []
    depth_sum = 0.0
    count = 0
    for output, chunk in zip(outputs, chunks):
        assert output is not None
        generated.extend(output["generated_ids"])
        depth_sum += float(output["mean_depth"]) * len(chunk)
        count += len(chunk)
    return {
        "latency_p50_ms": round(1000 * statistics.median(times), 3),
        "latency_p95_ms": round(1000 * p95, 3),
        "mean_depth": depth_sum / max(1, count),
        "generated_ids": generated,
        "runs": len(times),
        "warmup_passes": WARMUP,
        "batch_size": width,
    }


def timed_staged(
    model,
    questions: list[str],
    *,
    depth: int = 3,
    batch_size: int = 16,
) -> dict:
    """Time chained carry-scratchpad decoding on held-out questions."""
    chunks = [
        questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
    ]
    outputs: list[dict | None] = [None] * len(chunks)
    from backend.engine.arith_solver import solve_hard_batch

    def run(chunk: list[str]) -> dict:
        return solve_hard_batch(
            model, TOK, chunk, depth=depth, max_seq_len=MODEL_MAX_LEN
        )

    for _ in range(WARMUP):
        for chunk in chunks:
            run(chunk)
    times = []
    for repeat in range(TIMED):
        index = repeat % len(chunks)
        t0 = time.perf_counter()
        outputs[index] = run(chunks[index])
        times.append(time.perf_counter() - t0)
    for index, chunk in enumerate(chunks):
        if outputs[index] is None:
            outputs[index] = run(chunk)
    times_sorted = sorted(times)
    p95 = times_sorted[
        min(len(times_sorted) - 1, math.ceil(0.95 * len(times_sorted)) - 1)
    ]
    answers: list[str] = []
    depth_sum = 0.0
    for output in outputs:
        assert output is not None
        answers.extend(output["answers"])
        depth_sum += float(output["mean_depth"]) * len(output["answers"])
    return {
        "latency_p50_ms": round(1000 * statistics.median(times), 3),
        "latency_p95_ms": round(1000 * p95, 3),
        "mean_depth": depth_sum / max(1, len(answers)),
        "answers": answers,
        "runs": len(times),
        "warmup_passes": WARMUP,
        "batch_size": batch_size,
    }


def accuracy_from_answers(answers: list[str], rows: list[dict]) -> float:
    correct = 0
    for answer, row in zip(answers, rows):
        correct += int(
            normalize_answer(answer) == normalize_answer(str(row["answer_text"]))
        )
    return correct / max(1, len(rows))


def old_pooled_state(model, rows: list[dict]):
    """Reproduce the old Test 5/6 pooled writer for the baseline."""
    texts = [row["prompt"] for row in rows]
    seqs = [[BOS_ID] + TOK.encode(text, max_len=64) for text in texts]
    width = max(len(seq) for seq in seqs)
    ids = torch.zeros((len(seqs), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = 1
    with torch.no_grad():
        _, keys_all, values_all = model.encode_memory(ids, mask)
        keys = torch.stack(
            [
                pool_memory_tokens(keys_all[i, 1 : int(mask[i].sum().item())])
                for i in range(len(seqs))
            ]
        )
        values = torch.stack(
            [
                pool_memory_tokens(values_all[i, 1 : int(mask[i].sum().item())])
                for i in range(len(seqs))
            ]
        )
    state = model.memory.init_state(DEVICE)
    return model.memory.consolidate(state, keys, values)


REPORT = {
    "format": "will2m-eval-v1",
    "run_id": OUT.name,
    "created_at": time.time(),
    "comparison": {"old": OLD, "new": NEW},
    "settings": {
        "warmup_passes": WARMUP,
        "timed_runs": TIMED,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "old_mode": "adaptive_hard",
        "new_mode": "fixed_depth",
        "old_test3_depth": "adaptive_hard",
        "new_test3_router": (
            {"easy": 1, "hard": 3} if USE_CHAIN else {"easy": 1, "hard": 4}
        ),
        "new_hard_format": (
            "right_to_left_carry_chain" if USE_CHAIN else "direct_answer_4_loops"
        ),
        "new_scratchpad": (
            "raw ones sum -> raw tens sum (with carry) -> answer"
            if USE_CHAIN
            else "auxiliary ones,carry1,tens,carry2,answer trace"
        ),
        "memory_writer": "sequential_answer_slots",
        "new_memory_cue": "recall",
        "memory_ridge": 1e-6,
    },
    "results": {},
}


def evaluate_checkpoint(tag: str, checkpoint_id: str):
    print(f"[{tag}] loading {checkpoint_id}", flush=True)
    model, mcfg, payload = load(checkpoint_id)
    seed = int((payload.get("training_config") or {}).get("seed", 1337))
    active_params = sum(p.numel() for p in model.parameters())
    if not getattr(mcfg, "memory_logit_skip", False):
        # The v2 code keeps slot modules in the class for checkpoint
        # compatibility, but the old pooled-memory path never executes them.
        active_params -= sum(
            p.numel() for p in model.memory_slots.parameters()
        )
        active_params -= model.mem_to_vocab.weight.numel()
    entry = {
        "checkpoint_id": checkpoint_id,
        "parameters": active_params,
        "parameters_include_legacy_slot_modules": bool(
            getattr(mcfg, "memory_logit_skip", False)
        ),
        "fast_state_bytes": model.memory.init_state(DEVICE).state_bytes(),
        "min_steps": mcfg.min_steps,
        "max_steps": mcfg.max_steps,
        "memory_logit_skip": bool(getattr(mcfg, "memory_logit_skip", False)),
    }

    test3 = make_mixed_dataset(
        "arith_add", "test", 200, seed, TOK, mcfg.max_seq_len, easy_ratio=0.5
    )
    easy_rows = [x for x in test3.items if x["complexity"] == 1]
    hard_rows = [x for x in test3.items if x["complexity"] == 2]
    easy_prompts = [list(x["prompt_ids"]) for x in easy_rows]
    hard_prompts = [list(x["prompt_ids"]) for x in hard_rows]

    if tag == "new":
        easy = timed(model, easy_prompts, 1, max_new_tokens=6)
        hard = (
            timed_staged(model, [x["prompt"] for x in hard_rows], depth=3)
            if USE_CHAIN
            else timed(model, hard_prompts, 4, max_new_tokens=6)
        )
    else:
        depth = mcfg.resolved_adaptive_steps()
        easy = timed(
            model, easy_prompts, depth, max_new_tokens=12, mode="adaptive_hard"
        )
        hard = timed(
            model, hard_prompts, depth, max_new_tokens=12, mode="adaptive_hard"
        )
    entry["test3"] = {
        "seed": seed,
        "strata": {
            "easy": {
                "count": len(easy_rows),
                "accuracy_pct": round(100 * accuracy(easy["generated_ids"], easy_rows), 4),
                "mean_depth": easy["mean_depth"],
                "latency_p50_ms": easy["latency_p50_ms"],
                "latency_p95_ms": easy["latency_p95_ms"],
                "batch_size": easy["batch_size"],
            },
            "hard": {
                "count": len(hard_rows),
                "accuracy_pct": round(
                    100
                    * (
                        accuracy_from_answers(hard["answers"], hard_rows)
                        if tag == "new" and USE_CHAIN
                        else accuracy(hard["generated_ids"], hard_rows)
                    ),
                    4,
                ),
                "mean_depth": hard["mean_depth"],
                "latency_p50_ms": hard["latency_p50_ms"],
                "latency_p95_ms": hard["latency_p95_ms"],
                "batch_size": hard["batch_size"],
                "decoder": (
                    ("right_to_left_carry_chain" if USE_CHAIN else "direct_answer_4_loops")
                    if tag == "new"
                    else "adaptive"
                ),
            },
        },
    }
    print(
        f"[{tag}] Test3 easy={entry['test3']['strata']['easy']['accuracy_pct']} "
        f"hard={entry['test3']['strata']['hard']['accuracy_pct']}",
        flush=True,
    )

    # ------------------------------------------------------------ Test 6
    seed6 = seed + 999_999
    qa_ds = make_dataset(
        "arith_add", "test", 30, seed6, TOK, mcfg.max_seq_len, complexity=2
    )
    qa = list(qa_ds.items)
    prompts = [list(x["prompt_ids"]) for x in qa]

    if tag == "new":
        cold = (
            timed_staged(model, [x["prompt"] for x in qa], depth=3)
            if USE_CHAIN
            else timed(
                model, prompts, 4, max_new_tokens=6, batch_size=len(prompts)
            )
        )
        warm_prompts = [
            [BOS_ID] + TOK.encode(x["prompt"] + " recall") + [SEP_ID]
            for x in qa
        ]
        ids, mask, positions, _ = pad_prompts(
            warm_prompts, TOK.pad_id, left=True, max_len=mcfg.max_seq_len
        )
        state = write_slot_memory(
            model,
            ids,
            mask,
            [x["answer_text"] for x in qa],
            TOK,
            positions=positions,
            ridge=1e-6,
        )
        warm = timed(
            model,
            warm_prompts,
            1,
            memory_state=state,
            max_new_tokens=6,
            batch_size=len(prompts),
        )
        writer = "sequential_answer_slots"
    else:
        cold = timed(
            model,
            prompts,
            mcfg.resolved_adaptive_steps(),
            max_new_tokens=12,
            batch_size=len(prompts),
            mode="adaptive_hard",
        )
        state = old_pooled_state(model, qa)
        warm = timed(
            model,
            prompts,
            mcfg.resolved_adaptive_steps(),
            memory_state=state,
            max_new_tokens=12,
            batch_size=len(prompts),
            mode="adaptive_hard",
        )
        writer = "pooled_ridge_consolidate"

    entry["test6"] = {
        "questions": len(qa),
        "seed": seed6,
        "writer": writer,
        "fast_state_bytes": model.memory.state_bytes(state),
        "cold": {
            "accuracy_pct": round(
                100
                * (
                    accuracy_from_answers(cold["answers"], qa)
                    if tag == "new" and USE_CHAIN
                    else accuracy(cold["generated_ids"], qa)
                ),
                4,
            ),
            "mean_depth": cold["mean_depth"],
            "latency_p50_ms": cold["latency_p50_ms"],
            "latency_p95_ms": cold["latency_p95_ms"],
            "batch_size": cold["batch_size"],
        },
        "warm": {
            "accuracy_pct": round(100 * accuracy(warm["generated_ids"], qa), 4),
            "mean_depth": warm["mean_depth"],
            "latency_p50_ms": warm["latency_p50_ms"],
            "latency_p95_ms": warm["latency_p95_ms"],
            "batch_size": warm["batch_size"],
        },
    }
    entry["test6"]["depth_delta"] = round(
        entry["test6"]["warm"]["mean_depth"]
        - entry["test6"]["cold"]["mean_depth"],
        4,
    )
    entry["test6"]["p50_delta_ms"] = round(
        entry["test6"]["warm"]["latency_p50_ms"]
        - entry["test6"]["cold"]["latency_p50_ms"],
        4,
    )
    print(
        f"[{tag}] Test6 cold={entry['test6']['cold']['accuracy_pct']} "
        f"warm={entry['test6']['warm']['accuracy_pct']}",
        flush=True,
    )
    REPORT["results"][tag] = entry


for tag, checkpoint_id in (("old", OLD), ("new", NEW)):
    evaluate_checkpoint(tag, checkpoint_id)

with (OUT / "report.json").open("w", encoding="utf-8") as f:
    json.dump(REPORT, f, indent=2)
print("WROTE", OUT / "report.json", flush=True)
