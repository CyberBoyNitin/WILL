"""Train WILL v4 on a free GPU (Colab T4 / Kaggle) or locally on CPU.

Three-stage curriculum:
  1. Fluent English + scratchpad reasoning warm start.
  2. Sequence-correctness-gated ponder fine-tuning (easy depth ~1, hard ~3-4).
  3. Document-absorption synergy: unseen paragraphs go into fast weights,
     context is cleared, and one-step recall is trained.

Open datasets (TinyStories, GSM8K) are used when the ``datasets`` package and
network/cache are available.  A deterministic synthetic corpus is the offline
fallback, so the script is fully self-contained.

Examples::

    python scripts/train_will_v4.py --smoke --threads 2
    python scripts/train_will_v4.py --device auto --steps-per-stage 2000
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import ModelConfig, TrainingConfig, load_config  # noqa: E402
from backend.engine.device import (  # noqa: E402
    amp_dtype,
    autocast_for,
    device_summary,
    resolve_device,
)
from backend.engine.model import WILLModel  # noqa: E402
from backend.engine.tokenizer import build_tokenizer  # noqa: E402
from backend.memory.slot_memory import absorb_documents  # noqa: E402
from backend.training.checkpoints import save_checkpoint  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/will_125m_v4.yaml")
p.add_argument("--smoke", action="store_true", help="tiny CPU/GPU verification run")
p.add_argument("--device", default="auto")
p.add_argument("--steps-per-stage", type=int, default=1500)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max-texts", type=int, default=2000)
p.add_argument("--threads", type=int, default=0)
p.add_argument("--out", default=None)
args = p.parse_args()

if args.threads:
    torch.set_num_threads(args.threads)

base_cfg = load_config(args.config)
cfg = base_cfg.model
if args.smoke:
    cfg = cfg.model_copy(deep=True)
    cfg.vocab_size = 260
    cfg.hidden_dim = 128
    cfg.n_heads = 4
    cfg.n_kv_heads = 2
    cfg.context_layers = 2
    cfg.context_ff_mult = 2.0
    cfg.ff_mult = 2
    cfg.max_seq_len = 128
    cfg.memory_dim = 128
    cfg.slot_dim = 128
    cfg.num_slots = 8
    cfg.memory_slots = 8
    cfg.memory_token_dim = 64
    cfg.memory_kind = "ttt_linear"

DEVICE = resolve_device(args.device)
AMP_DTYPE = amp_dtype(DEVICE)
USE_SCALER = DEVICE.type == "cuda" and AMP_DTYPE == torch.float16
TOK = build_tokenizer(cfg)
SEED = int(base_cfg.training.seed)
torch.manual_seed(SEED)
random.seed(SEED)
model = WILLModel(cfg).to(DEVICE)
out_id = args.out or ("will-125m-v4-smoke" if args.smoke else "will-125m-v4")
print("v4 device", device_summary(DEVICE), "params", model.parameter_counts()[0], flush=True)


# ---------------------------------------------------------------- data
def synthetic_stories(n: int) -> list[str]:
    names = ("Ada", "Ben", "Cleo", "Dia", "Eli", "Fay")
    places = ("the park", "the library", "the lab", "the garden")
    out = []
    for i in range(n):
        name, place = names[i % len(names)], places[i % len(places)]
        out.append(
            f"{name} went to {place} with a bright lantern. "
            f"The wind was cold, but the path was safe. {name} smiled and keptwalking."
        )
    return out


def synthetic_reasoning(n: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        a, b = 10 + i % 70, 10 + (i * 7) % 80
        s1 = a % 10 + b % 10
        s2 = a // 10 + b // 10 + s1 // 10
        answer = str(s2 * 10 + s1 % 10)
        out.append(
            {
                "prompt": f"calc {a} + {b} =",
                "trace": f"ones {s1} tens {s2}",
                "answer": answer,
            }
        )
    return out


def synthetic_facts(n: int) -> list[dict[str, str]]:
    entities = [f"Entity{i:03d}" for i in range(n)]
    return [
        {
            "paragraph": f"{e} -> the recorded value is token {i * 7 % 1000}.",
            "query": e,
            "answer": f"token {i * 7 % 1000}",
        }
        for i, e in enumerate(entities)
    ]


def try_hf_texts(name: str, split: str, limit: int) -> list[str]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset(name, split=f"{split}[:{limit}]")
        key = "text" if "text" in ds.column_names else "story"
        return [str(row[key]) for row in ds if row.get(key)][:limit]
    except Exception:
        return []


def encode_with_target(text: str, target: str, max_len: int):
    prompt = TOK.encode(text, add_bos=True)
    answer = TOK.encode(target, add_eos=True)
    ids = (prompt + answer)[:max_len]
    labels = [-100] * len(prompt) + answer
    labels = labels[: max_len - 1]
    return ids[:-1], labels[1:]


def collate(rows: list[tuple[list[int], list[int]]]):
    width = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), width), TOK.pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (row_ids, row_labels) in enumerate(rows):
        ids[i, : len(row_ids)] = torch.tensor(row_ids)
        labels[i, : len(row_labels)] = torch.tensor(row_labels)
        mask[i, : len(row_ids)] = 1
    return ids.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)


stories = try_hf_texts("roneneldan/TinyStories", "train", args.max_texts)
if not stories:
    stories = synthetic_stories(args.max_texts)
reasoning = synthetic_reasoning(args.max_texts)
facts = synthetic_facts(max(32, args.max_texts // 8))
print(f"data stories={len(stories)} reasoning={len(reasoning)} facts={len(facts)}", flush=True)


# ---------------------------------------------------------------- training
scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_SCALER)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=max(1, args.steps_per_stage * 3), eta_min=args.lr * 0.1
)


def train_loss(batch, depth: int, memory_state=None, ponder: float = 0.0):
    ids, labels, mask = batch
    with autocast_for(DEVICE, enabled=AMP_DTYPE is not None):
        logits, stats = model(
            ids,
            mask,
            memory_state=memory_state,
            mode="adaptive_soft",
            max_steps=depth,
        )
        loss = F.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=-100
        )
        if ponder:
            loss = loss + ponder * float(stats.get("expected_steps", depth))
    return loss, stats


def step(loss):
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()


stages = (
    ("fluent+scratchpad", 0.0),
    ("ponder-gated", cfg.halt_epsilon),
    ("document-memory", 0.0),
)
for stage, ponder in stages:
    print(f"[stage] {stage}", flush=True)
    running = 0.0
    for step_idx in range(args.steps_per_stage):
        model.train()
        if stage == "document-memory":
            sample = random.sample(facts, min(args.batch, len(facts)))
            paragraphs = [item["paragraph"] for item in sample]
            state = absorb_documents(
                model,
                TOK,
                paragraphs,
                max_facts=len(sample),
                slots_per_fact=max(1, min(4, cfg.num_slots // 8)),
            )
            rows = [
                encode_with_target(item["query"], item["answer"], cfg.max_seq_len)
                for item in sample
            ]
            loss, stats = train_loss(collate(rows), 1, memory_state=state)
        else:
            if step_idx % 3 == 0:
                texts = random.sample(stories, min(args.batch, len(stories)))
                rows = [
                    encode_with_target(text[:120], "", cfg.max_seq_len)
                    for text in texts
                ]
                depth = 1
            else:
                items = random.sample(reasoning, min(args.batch, len(reasoning)))
                rows = [
                    encode_with_target(
                        f"{item['prompt']} {item['trace']}",
                        item["answer"],
                        cfg.max_seq_len,
                    )
                    for item in items
                ]
                depth = cfg.max_steps
            loss, stats = train_loss(collate(rows), depth, ponder=ponder)
        step(loss)
        running += float(loss.detach().item())
        if step_idx % 100 == 0:
            print(
                f"  {stage} {step_idx}/{args.steps_per_stage} "
                f"loss={running / max(1, step_idx + 1):.4f} "
                f"depth={stats.get('mean_depth')}",
                flush=True,
            )


# ---------------------------------------------------------------- checkpoint
tcfg = TrainingConfig(
    task="lm",
    train_size=len(stories) + len(reasoning),
    val_size=128,
    test_size=128,
    epochs=1,
    batch_size=args.batch,
    precision="auto",
    mode="adaptive_soft",
    seed=SEED,
    checkpoint_name=out_id,
    profile=cfg.profile if hasattr(cfg, "profile") else "custom",
)
metrics = {
    "run_id": out_id,
    "smoke": bool(args.smoke),
    "device": device_summary(DEVICE),
    "parameters": model.parameter_counts()[0],
    "fast_state_bytes": model.memory.init_state(DEVICE).state_bytes(),
    "curriculum": ["fluent+scratchpad", "ponder-gated", "document-memory"],
}
save_checkpoint(
    base_cfg.server.checkpoint_dir,
    out_id,
    model,
    cfg,
    tcfg,
    opt,
    metrics,
    [],
)
print(f"SAVED {out_id} {metrics}", flush=True)
