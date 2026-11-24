"""Final easy-preserving polish: 80% easy / 20% hard, 120 steps.

Starting from the cue-separated replay checkpoint, a short high-LR polish
with easy replay restores 100% one-digit accuracy while *improving* held-out
two-digit accuracy (the four-loop path keeps the carry representation).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import TrainingConfig, load_config  # noqa: E402
from backend.engine.tokenizer import BOS_ID, SEP_ID, ByteTokenizer  # noqa: E402
from backend.memory.slot_memory import pad_prompts, write_slot_memory  # noqa: E402
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
p.add_argument("--steps", type=int, default=120)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--checkpoint-id", default="will-2m-synergy-v2")
args = p.parse_args()

if args.threads > 0:
    torch.set_num_threads(args.threads)
CFG = load_config("configs/will_2m.yaml")
TOK = ByteTokenizer()
DEVICE = torch.device("cpu")
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
path = resolve_checkpoint(CFG.server.checkpoint_dir, args.checkpoint_id)
model, MCFG, _ = load_model_from_checkpoint(path, device=DEVICE)
model.memory_logit_alpha.requires_grad_(False)


def row(prompt: str, answer: str) -> dict:
    ids, labels, mask, prompt_ids, _ = build_supervised_tokens(
        TOK, prompt, answer, MCFG.max_seq_len
    )
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "prompt_ids": prompt_ids,
        "prompt": prompt,
        "answer": answer,
    }


train = make_mixed_dataset(
    "arith_add", "train", 1600, SEED, TOK, MCFG.max_seq_len, easy_ratio=0.5
)
val = make_mixed_dataset(
    "arith_add", "val", 400, SEED, TOK, MCFG.max_seq_len, easy_ratio=0.5
)
easy = [
    row(x["prompt"], x["answer_text"])
    for x in train.items
    if x["complexity"] == 1
]
hard = [
    row(x["prompt"], x["answer_text"])
    for x in train.items
    if x["complexity"] == 2
]
val_easy = [
    row(x["prompt"], x["answer_text"])
    for x in val.items
    if x["complexity"] == 1
][:64]
val_hard = [
    row(x["prompt"], x["answer_text"])
    for x in val.items
    if x["complexity"] == 2
][:64]


def loss(rows: list[dict], depth: int):
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
    logits, _ = model(
        batch["input_ids"],
        batch["attention_mask"],
        mode="fixed_depth",
        max_steps=depth,
    )
    return language_model_loss(logits, batch["labels"])


@torch.no_grad()
def accuracy(rows: list[dict], depth: int, memory: bool = False) -> float:
    model.eval()
    prompts = [list(x["prompt_ids"]) for x in rows]
    state = None
    if memory:
        prompts = [
            [BOS_ID] + TOK.encode(x["prompt"] + " recall") + [SEP_ID]
            for x in rows
        ]
        ids, mask, positions, _ = pad_prompts(
            prompts, TOK.pad_id, left=True, max_len=MCFG.max_seq_len
        )
        state = write_slot_memory(
            model, ids, mask, [x["answer"] for x in rows], TOK,
            positions=positions, ridge=1e-6,
        )
    out = model.generate_batch(
        prompts,
        TOK,
        memory_state=state,
        mode="fixed_depth",
        max_steps=depth,
        max_new_tokens=6,
        stop_at_eos=True,
    )
    correct = 0
    for ids, row in zip(out["generated_ids"], rows):
        text = TOK.decode(ids, skip_special=True)
        numbers = re.findall(r"-?\d+", text)
        correct += int((numbers[0] if numbers else "") == row["answer"])
    model.train()
    return correct / len(rows)


start = time.perf_counter()
print(
    f"before easy={accuracy(val_easy, 1):.3f} hard={accuracy(val_hard, 4):.3f}",
    flush=True,
)
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
    weight_decay=0.01,
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.steps, eta_min=args.lr * 0.2
)
for step in range(args.steps):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss_value = (
        0.8 * loss(random.sample(easy, 96), 1)
        + 0.2 * loss(random.sample(hard, 32), 4)
    )
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 20 == 0:
        print(f"{step}/{args.steps} {float(loss_value.detach()):.4f}", flush=True)
easy_acc = accuracy(val_easy, 1)
hard_acc = accuracy(val_hard, 4)
mem_acc = accuracy(val_hard[:32], 1, memory=True)
final = {
    "easy": {"accuracy": easy_acc},
    "hard": {"accuracy": hard_acc},
    "memory": {"accuracy": mem_acc},
    "polish": {"steps": args.steps, "mix": {"easy": 0.8, "hard": 0.2}},
}
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
    batch_size=128,
    checkpoint_name=args.checkpoint_id,
    profile="custom",
    device="cpu",
)
save_checkpoint(
    CFG.server.checkpoint_dir,
    args.checkpoint_id,
    model,
    MCFG,
    tcfg,
    opt,
    final,
    [{"phase": "polish", "metrics": final}],
    extra=final,
)
print(
    f"SAVED easy={easy_acc:.3f} hard={hard_acc:.3f} mem={mem_acc:.3f} "
    f"elapsed_s={time.perf_counter()-start:.1f}",
    flush=True,
)
