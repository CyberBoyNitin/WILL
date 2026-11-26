"""Fine-tune will-2m-synergy-v3 with right-to-left carry scratchpad + replay.

Starting from v2 (which already has 100% Easy and 100% warm memory), each
training step samples one of three tasks:

  * 40% Easy one-digit addition (depth 1) in the final replay phase
  * 40% Hard Complexity-2 scratchpad stages (depth 2)
  * 20% Memory-conditioned replay with a ``recall`` cue (depth 1)

The hard curriculum emits the raw ones-column sum, then the raw tens-column
sum (which already includes the carry), then reads the answer off those two
sums by copying the last ones digit.  All operand pairs are sampled fresh, so
this is rule learning, not memorising the 1,600-example training file.
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
from backend.engine.arith_solver import solve_hard_batch  # noqa: E402
from backend.engine.tokenizer import BOS_ID, SEP_ID, ByteTokenizer  # noqa: E402
from backend.engine.model import WILLModel  # noqa: E402
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

p = argparse.ArgumentParser()
p.add_argument("--init", default="will-2m-synergy-v2")
p.add_argument("--out", default="will-2m-synergy-v3")
p.add_argument("--stage-steps", type=int, default=1000)
p.add_argument("--replay-steps", type=int, default=300)
p.add_argument("--batch", type=int, default=48)
p.add_argument("--depth", type=int, default=2)
p.add_argument("--stage-lr", type=float, default=8e-4)
p.add_argument("--replay-lr", type=float, default=3e-4)
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
model, MCFG, _ = load_model_from_checkpoint(path, device=DEVICE)
model.memory_logit_alpha.requires_grad_(False)


def pair() -> tuple[int, int]:
    return random.randint(10, 99), random.randint(10, 99)


def columns(a: int, b: int) -> dict[str, int]:
    s1 = a % 10 + b % 10
    c1 = s1 // 10
    s2 = a // 10 + b // 10 + c1
    return {"s1": s1, "s2": s2, "answer": a + b}


STAGE_NAMES = ("s1", "s2", "answer")
STAGE_WEIGHTS = (0.35, 0.35, 0.30)


def stage_row(a: int, b: int, stage: str) -> dict:
    c = columns(a, b)
    base = f"calc {a} + {b} ="
    if stage == "s1":
        prompt, target = base, str(c["s1"])
    elif stage == "s2":
        prompt, target = f"{base} ones {c['s1']}", str(c["s2"])
    else:
        prompt = f"{base} ones {c['s1']} tens {c['s2']}"
        target = str(c["answer"])
    ids, labels, mask, prompt_ids, _ = build_supervised_tokens(
        TOK, prompt, target, MCFG.max_seq_len
    )
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "prompt_ids": prompt_ids,
        "answer": str(c["answer"]),
        "columns": c,
    }


def easy_row() -> dict:
    a, b = random.randint(0, 9), random.randint(0, 9)
    prompt = f"calc {a} + {b} ="
    ids, labels, mask, prompt_ids, _ = build_supervised_tokens(
        TOK, prompt, str(a + b), MCFG.max_seq_len
    )
    return {
        "input_ids": ids, "labels": labels, "attention_mask": mask,
        "prompt_ids": prompt_ids, "answer": str(a + b),
    }


def mem_row() -> dict:
    a, b = pair()
    prompt = f"calc {a} + {b} = recall"
    ids, labels, mask, prompt_ids, _ = build_supervised_tokens(
        TOK, prompt, str(a + b), MCFG.max_seq_len
    )
    return {
        "input_ids": ids, "labels": labels, "attention_mask": mask,
        "prompt_ids": prompt_ids, "answer": str(a + b),
    }


val = make_mixed_dataset(
    "arith_add", "val", 400, SEED, TOK, MCFG.max_seq_len, easy_ratio=0.5
)
val_easy = [x for x in val.items if x["complexity"] == 1][:64]
val_hard = [x for x in val.items if x["complexity"] == 2][:64]
hard_prompts = [x["prompt"] for x in val_hard]
print(
    f"init={args.init} val_easy={len(val_easy)} val_hard={len(val_hard)}",
    flush=True,
)


def batch_loss(rows: list[dict], depth: int, memory: bool = False):
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
    keep = labels != -100
    ce = F.cross_entropy(
        logits.transpose(1, 2), labels, reduction="none", ignore_index=-100
    )
    first = keep.to(torch.int64).argmax(dim=1).unsqueeze(1)
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights.scatter_(1, first, 3.0)
    loss = (ce * weights * keep).sum() / weights.mul(keep).sum().clamp_min(1.0)
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
def evaluate_easy() -> float:
    model.eval()
    prompts = [
        [BOS_ID] + TOK.encode(x["prompt"]) + [SEP_ID] for x in val_easy
    ]
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
def evaluate_memory() -> float:
    model.eval()
    rows = val_hard[:32]
    prompts = [
        [BOS_ID] + TOK.encode(x["prompt"] + " recall") + [SEP_ID]
        for x in rows
    ]
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


@torch.no_grad()
def evaluate_hard() -> dict:
    model.eval()
    solved = solve_hard_batch(
        model, TOK, hard_prompts, depth=args.depth, max_seq_len=MCFG.max_seq_len
    )
    correct = 0
    field_correct = {name: 0 for name in ("s1", "s2")}
    for index, row in enumerate(val_hard):
        answer = solved["answers"][index]
        correct += int(answer == row["answer_text"])
        a, b = map(int, re.findall(r"\d+", row["prompt"]))
        expected = columns(a, b)
        for name in field_correct:
            field_correct[name] += int(solved["fields"][name][index] == expected[name])
    model.train()
    return {
        "accuracy": correct / len(val_hard),
        "field_accuracy": {
            name: value / len(val_hard) for name, value in field_correct.items()
        },
        "example_trace": [trace[:3] for trace in solved["stages"]],
    }


def save(phase: str, history: list[dict], opt, extra: dict):
    tcfg = TrainingConfig(
        task="arith_add", mixed_easy_ratio=0.4, min_complexity=1,
        max_complexity=2, mode="fixed_depth", precision="fp32", seed=SEED,
        max_steps=MCFG.max_steps, train_size=1600, val_size=len(val),
        test_size=200, batch_size=args.batch, checkpoint_name=args.out,
        profile="custom", device="cpu",
    )
    metrics = {
        "run_id": args.out,
        "phase": phase,
        "history": history,
        "curriculum": {
            "init_from": args.init,
            "stage_steps": args.stage_steps,
            "replay_steps": args.replay_steps,
            "stages": list(STAGE_NAMES),
            "stage_weights": list(STAGE_WEIGHTS),
            "final_mix": {"easy": 0.4, "hard": 0.4, "memory": 0.2},
            "hard_depth": args.depth,
            "fresh_operands_per_batch": True,
        },
        **extra,
    }
    save_checkpoint(
        CFG.server.checkpoint_dir, args.out, model, MCFG, tcfg, opt,
        metrics, history, extra=metrics,
    )


def pick_stage() -> str:
    return random.choices(STAGE_NAMES, weights=STAGE_WEIGHTS, k=1)[0]


def sample_task(weights: tuple[float, float, float]) -> str:
    return random.choices(("easy", "hard", "memory"), weights=weights, k=1)[0]


history: list[dict] = []
start = time.perf_counter()

for phase, steps, lr, weights in (
    ("stage", args.stage_steps, args.stage_lr, (0.20, 0.70, 0.10)),
    ("replay", args.replay_steps, args.replay_lr, (0.40, 0.40, 0.20)),
):
    print(f"[{phase}] {steps} steps lr={lr}", flush=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.15
    )
    for step in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        task = sample_task(weights)
        if task == "easy":
            rows = [easy_row() for _ in range(args.batch)]
            loss, slot = batch_loss(rows, 1)
        elif task == "memory":
            rows = [mem_row() for _ in range(args.batch)]
            loss, slot = batch_loss(rows, 1, memory=True)
        else:
            stage = pick_stage()
            rows = [
                stage_row(*pair(), stage) for _ in range(args.batch)
            ]
            loss, slot = batch_loss(rows, args.depth)
        total = loss + 0.5 * slot
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(
                f"{phase} {step}/{steps} {task} {float(total.detach()):.4f}",
                flush=True,
            )
        if step and step % 300 == 0:
            hard_metrics = evaluate_hard()
            easy_acc, mem_acc = evaluate_easy(), evaluate_memory()
            print(
                f"{phase} eval {step} hard={hard_metrics['accuracy']:.3f} "
                f"easy={easy_acc:.3f} mem={mem_acc:.3f}",
                flush=True,
            )
            history.append(
                {
                    "phase": phase, "step": step,
                    "metrics": {
                        "hard": hard_metrics, "easy": easy_acc,
                        "memory": mem_acc,
                    },
                }
            )
    hard_metrics = evaluate_hard()
    easy_acc, mem_acc = evaluate_easy(), evaluate_memory()
    final = {
        "hard": hard_metrics, "easy": {"accuracy": easy_acc},
        "memory": {"accuracy": mem_acc},
    }
    history.append({"phase": phase, "step": steps, "metrics": final})
    print(
        f"[{phase}] done hard={hard_metrics['accuracy']:.3f} "
        f"easy={easy_acc:.3f} mem={mem_acc:.3f} "
        f"fields={hard_metrics['field_accuracy']}",
        flush=True,
    )
    save(phase, history, opt, final)

print(
    f"SAVED {args.out} params={model.parameter_counts()[0]} "
    f"elapsed_s={time.perf_counter()-start:.1f}",
    flush=True,
)
