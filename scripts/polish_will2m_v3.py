"""Restore Easy while keeping the v3 right-to-left carry chain.

The stage curriculum briefly lowers one-digit accuracy because hard stages
dominate the shared representation. A short 80% Easy / 20% hard-answer-stage
polish restores Easy >=99% and usually improves the final chain as well.
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
from backend.engine.arith_solver import solve_hard_batch  # noqa: E402
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
p.add_argument("--hard-weight", type=float, default=0.2)
p.add_argument("--init", default="will-2m-synergy-v3-base")
p.add_argument("--out", default="will-2m-synergy-v3")
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
model, MCFG, _ = load_model_from_checkpoint(
    resolve_checkpoint(CFG.server.checkpoint_dir, args.init), device=DEVICE
)
model.memory_logit_alpha.requires_grad_(False)


def columns(a: int, b: int) -> dict[str, int]:
    s1 = a % 10 + b % 10
    s2 = a // 10 + b // 10 + s1 // 10
    return {"s1": s1, "s2": s2, "answer": a + b}


def make_row(prompt: str, target: str, answer: str) -> dict:
    ids, labels, mask, prompt_ids, _ = build_supervised_tokens(
        TOK, prompt, target, MCFG.max_seq_len
    )
    return {"input_ids": ids, "labels": labels, "attention_mask": mask,
            "prompt_ids": prompt_ids, "prompt": prompt, "answer": answer}


def easy_row() -> dict:
    a, b = random.randint(0, 9), random.randint(0, 9)
    answer = str(a + b)
    return make_row(f"calc {a} + {b} =", answer, answer)


def hard_answer_row() -> dict:
    a, b = random.randint(10, 99), random.randint(10, 99)
    c = columns(a, b)
    prompt = f"calc {a} + {b} = ones {c['s1']} tens {c['s2']}"
    return make_row(prompt, str(c["answer"]), str(c["answer"]))


train = make_mixed_dataset(
    "arith_add", "train", 1600, SEED, TOK, MCFG.max_seq_len, easy_ratio=0.5
)
val = make_mixed_dataset(
    "arith_add", "val", 400, SEED, TOK, MCFG.max_seq_len, easy_ratio=0.5
)
val_easy = [x for x in val.items if x["complexity"] == 1][:64]
val_hard = [x for x in val.items if x["complexity"] == 2][:64]
hard_prompts = [x["prompt"] for x in val_hard]


def loss(rows: list[dict], depth: int):
    batch = collate_lm(
        [{"input_ids": x["input_ids"], "labels": x["labels"],
          "attention_mask": x["attention_mask"]} for x in rows]
    )
    logits, _ = model(
        batch["input_ids"], batch["attention_mask"],
        mode="fixed_depth", max_steps=depth,
    )
    return language_model_loss(logits, batch["labels"])


@torch.no_grad()
def accuracy_easy() -> float:
    model.eval()
    prompts = [[BOS_ID] + TOK.encode(x["prompt"]) + [SEP_ID] for x in val_easy]
    out = model.generate_batch(
        prompts, TOK, mode="fixed_depth", max_steps=1, max_new_tokens=6
    )
    correct = 0
    for ids, row in zip(out["generated_ids"], val_easy):
        numbers = re.findall(r"-?\d+", TOK.decode(ids, skip_special=True))
        correct += int((numbers[0] if numbers else "") == row["answer_text"])
    model.train()
    return correct / len(val_easy)


@torch.no_grad()
def accuracy_chain() -> dict:
    model.eval()
    solved = solve_hard_batch(
        model, TOK, hard_prompts, depth=2, max_seq_len=MCFG.max_seq_len
    )
    correct = sum(
        int(answer == row["answer_text"])
        for answer, row in zip(solved["answers"], val_hard)
    )
    model.train()
    return {
        "accuracy": correct / len(val_hard),
        "example_trace": [trace[:3] for trace in solved["stages"]],
    }


@torch.no_grad()
def accuracy_memory() -> float:
    model.eval()
    rows = val_hard[:32]
    prompts = [[BOS_ID] + TOK.encode(x["prompt"] + " recall") + [SEP_ID]
               for x in rows]
    ids, mask, positions, _ = pad_prompts(
        prompts, TOK.pad_id, left=True, max_len=MCFG.max_seq_len
    )
    state = write_slot_memory(
        model, ids, mask, [x["answer_text"] for x in rows], TOK,
        positions=positions, ridge=1e-6,
    )
    out = model.generate_batch(
        prompts, TOK, memory_state=state, mode="fixed_depth",
        max_steps=1, max_new_tokens=6,
    )
    correct = 0
    for ids, row in zip(out["generated_ids"], rows):
        numbers = re.findall(r"-?\d+", TOK.decode(ids, skip_special=True))
        correct += int((numbers[0] if numbers else "") == row["answer_text"])
    model.train()
    return correct / len(rows)


start = time.perf_counter()
print(
    f"before easy={accuracy_easy():.3f} hard={accuracy_chain()['accuracy']:.3f} "
    f"mem={accuracy_memory():.3f}",
    flush=True,
)
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr, weight_decay=0.01,
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.steps, eta_min=args.lr * 0.2
)
for step in range(args.steps):
    model.train()
    opt.zero_grad(set_to_none=True)
    n_easy = max(1, int(128 * (1.0 - args.hard_weight)))
    n_hard = max(1, 128 - n_easy)
    value = (
        (1.0 - args.hard_weight) * loss([easy_row() for _ in range(n_easy)], 1)
        + args.hard_weight * loss([hard_answer_row() for _ in range(n_hard)], 2)
    )
    value.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 20 == 0:
        print(f"{step}/{args.steps} {float(value.detach()):.4f}", flush=True)
easy, hard, mem = accuracy_easy(), accuracy_chain(), accuracy_memory()
final = {
    "easy": {"accuracy": easy},
    "hard": hard,
    "memory": {"accuracy": mem},
    "polish": {
        "steps": args.steps,
        "mix": {
            "easy": 1.0 - args.hard_weight,
            "hard_chain": args.hard_weight,
        },
    },
}
tcfg = TrainingConfig(
    task="arith_add", mixed_easy_ratio=0.4, min_complexity=1,
    max_complexity=2, mode="fixed_depth", precision="fp32", seed=SEED,
    max_steps=MCFG.max_steps, train_size=1600, val_size=len(val),
    test_size=200, batch_size=128, checkpoint_name=args.out,
    profile="custom", device="cpu",
)
save_checkpoint(
    CFG.server.checkpoint_dir, args.out, model, MCFG, tcfg, opt,
    final, [{"phase": "polish", "metrics": final}], extra=final,
)
print(
    f"SAVED easy={easy:.3f} hard={hard['accuracy']:.3f} mem={mem:.3f} "
    f"elapsed_s={time.perf_counter()-start:.1f}",
    flush=True,
)
