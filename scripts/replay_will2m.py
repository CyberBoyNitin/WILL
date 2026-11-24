"""Finish will-2m-synergy-v2 with cue-separated 40/40/20 replay + easy polish.

The hard checkpoint already has 100% easy before hard fine-tuning and ~17%
held-out two-digit accuracy.  Plain replay used the same standard prompt for
both easy decoding and pooled-memory recall, so the depth-1 memory objective
overwrote the easy mapping.  Adding a ``recall`` cue makes the memory mode a
different prompt and keeps the two behaviours separable.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import TrainingConfig, load_config  # noqa: E402
from backend.engine.model import WILLModel  # noqa: E402
from backend.engine.tokenizer import ByteTokenizer  # noqa: E402
from backend.memory.slot_memory import (  # noqa: E402
    answer_slot_targets,
    pad_prompts,
    write_slot_memory,
)
from backend.training.checkpoints import (  # noqa: E402
    load_model_from_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)
from backend.training.datasets import (  # noqa: E402
    build_supervised_tokens,
    collate_lm,
    make_mixed_dataset,
)
from backend.training.losses import language_model_loss  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--init", default="will-2m-synergy-v2-hard")
p.add_argument("--out", default="will-2m-synergy-v2")
p.add_argument("--replay-steps", type=int, default=80)
p.add_argument("--replay-lr", type=float, default=1e-4)
p.add_argument("--polish-steps", type=int, default=40)
p.add_argument("--polish-lr", type=float, default=5e-4)
p.add_argument("--batch", type=int, default=48)
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()

if args.threads > 0:
    torch.set_num_threads(args.threads)
CFG = load_config("configs/will_2m.yaml")
TOK = ByteTokenizer()
DEVICE = torch.device("cpu")
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
path = resolve_checkpoint(CFG.server.checkpoint_dir, args.init)
model, MCFG, payload = load_model_from_checkpoint(path, device=DEVICE)
model.memory_logit_alpha.requires_grad_(False)


def rebuild(prompt: str, target: str, answer: str) -> dict:
    ids, labels, mask, prompt_ids, _ = build_supervised_tokens(
        TOK, prompt, target, MCFG.max_seq_len
    )
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "prompt_ids": prompt_ids,
        "answer": answer,
    }


train = make_mixed_dataset(
    "arith_add", "train", 1600, SEED, TOK, MCFG.max_seq_len, easy_ratio=0.5
)
val = make_mixed_dataset(
    "arith_add", "val", 400, SEED, TOK, MCFG.max_seq_len, easy_ratio=0.5
)
easy_rows, hard_rows, mem_rows = [], [], []
for item in train.items:
    answer = item["answer_text"]
    easy_rows.append(rebuild(item["prompt"], answer, answer))
    if item["complexity"] == 2:
        hard_rows.append(rebuild(item["prompt"], answer, answer))
        mem_rows.append(rebuild(item["prompt"] + " recall", answer, answer))
val_easy_raw = [x for x in val.items if x["complexity"] == 1][:64]
val_hard_raw = [x for x in val.items if x["complexity"] == 2][:64]
val_easy = [
    rebuild(x["prompt"], x["answer_text"], x["answer_text"])
    for x in val_easy_raw
]
val_hard = [
    rebuild(x["prompt"], x["answer_text"], x["answer_text"])
    for x in val_hard_raw
]


def sample(rows, n):
    return [rows[random.randrange(len(rows))] for _ in range(n)]


def batch_loss(rows, depth, memory=False):
    batch = collate_lm(
        [
            {
                "input_ids": x["input_ids"],
                "labels": x["labels"],
                "attention_mask": x["attention_mask"],
            }
            for x in rows
        ]
    )
    ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)
    state = None
    if memory:
        state = write_slot_memory(
            model, ids, mask, [x["answer"] for x in rows], TOK, ridge=1e-6
        )
    logits, stats = model(
        ids, mask, memory_state=state, mode="fixed_depth",
        max_steps=depth, memory_enabled=memory,
    )
    loss = language_model_loss(logits, labels)
    slot_loss = logits.new_zeros(())
    if memory:
        targets = answer_slot_targets(
            [x["answer"] for x in rows], TOK, MCFG.memory_slots, DEVICE
        )
        slot = stats["memory_slot_logits"]
        slot_loss = F.cross_entropy(
            slot.reshape(-1, slot.shape[-1]), targets.reshape(-1)
        )
    return loss, slot_loss


@torch.no_grad()
def accuracy(rows, depth, memory=False):
    model.eval()
    prompts = [list(x["prompt_ids"]) for x in rows]
    state = None
    if memory:
        ids, mask, positions, _ = pad_prompts(
            prompts, TOK.pad_id, left=True, max_len=MCFG.max_seq_len
        )
        state = write_slot_memory(
            model, ids, mask, [x["answer"] for x in rows], TOK,
            positions=positions, ridge=1e-6,
        )
    out = model.generate_batch(
        prompts, TOK, memory_state=state, mode="fixed_depth",
        max_steps=depth, max_new_tokens=6, stop_at_eos=True,
    )
    correct = 0
    for ids, row in zip(out["generated_ids"], rows):
        text = TOK.decode(ids, skip_special=True)
        nums = re.findall(r"-?\d+", text)
        correct += int((nums[0] if nums else "") == row["answer"])
    model.train()
    return correct / len(rows)


def memory_val_rows():
    return [
        rebuild(x["prompt"] + " recall", x["answer_text"], x["answer_text"])
        for x in val_hard_raw[:32]
    ]


history = []
start = time.perf_counter()
print("[R] cue-separated 40/40/20 replay", flush=True)
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.replay_lr, weight_decay=0.01,
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.replay_steps, eta_min=args.replay_lr * 0.2
)
n_easy = int(round(args.batch * 0.4))
n_hard = int(round(args.batch * 0.4))
n_mem = args.batch - n_easy - n_hard
for step in range(args.replay_steps):
    model.train()
    opt.zero_grad(set_to_none=True)
    e = sample(easy_rows, n_easy)
    h = sample(hard_rows, n_hard)
    m = sample(mem_rows, n_mem)
    le, _ = batch_loss(e, 1)
    lh, _ = batch_loss(h, MCFG.max_steps)
    lm, slot = batch_loss(m, 1, memory=True)
    loss = 0.4 * le + 0.4 * lh + 0.2 * lm + 0.5 * slot
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 10 == 0:
        print(f"R {step}/{args.replay_steps} {float(loss.detach()):.4f}", flush=True)
mem_rows_val = memory_val_rows()
print(
    f"[R] easy={accuracy(val_easy, 1):.3f} "
    f"hard={accuracy(val_hard, MCFG.max_steps):.3f} "
    f"mem={accuracy(mem_rows_val, 1, memory=True):.3f}",
    flush=True,
)

print("[P] easy polish", flush=True)
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.polish_lr, weight_decay=0.01,
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.polish_steps, eta_min=args.polish_lr * 0.2
)
for step in range(args.polish_steps):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, _ = batch_loss(sample(easy_rows, 128), 1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 10 == 0:
        print(f"P {step}/{args.polish_steps} {float(loss.detach()):.4f}", flush=True)
easy = accuracy(val_easy, 1)
hard = accuracy(val_hard, MCFG.max_steps)
mem = accuracy(mem_rows_val, 1, memory=True)
final = {
    "easy": {"accuracy": easy},
    "hard": {"accuracy": hard},
    "memory": {"accuracy": mem},
    "cue": "recall",
    "replay_steps": args.replay_steps,
    "polish_steps": args.polish_steps,
}
history.append({"phase": "replay", "metrics": final})
tcfg = TrainingConfig(
    task="arith_add",
    mixed_easy_ratio=0.4,
    min_complexity=1,
    max_complexity=2,
    mode="fixed_depth",
    precision="fp32",
    seed=SEED,
    max_steps=MCFG.max_steps,
    train_size=1600,
    val_size=len(val),
    test_size=200,
    batch_size=args.batch,
    checkpoint_name=args.out,
    profile="custom",
    device="cpu",
)
save_checkpoint(
    CFG.server.checkpoint_dir, args.out, model, MCFG, tcfg, opt, final,
    history, extra=final,
)
print(
    f"SAVED {args.out} easy={easy:.3f} hard={hard:.3f} mem={mem:.3f} "
    f"params={model.parameter_counts()[0]} elapsed_s={time.perf_counter()-start:.1f}",
    flush=True,
)
