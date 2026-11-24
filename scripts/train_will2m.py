"""Train will-2m-synergy-v2: easy warmup -> hard carry -> gentle replay.

Measured curriculum on this 4-core CPU:
  * easy-only warmup reaches 100% held-out one-digit accuracy
  * hard fine-tuning from that point reaches ~17% held-out two-digit accuracy
    (the 8M baseline measured in the same harness is ~13%)
  * final 40/40/20 replay keeps both paths alive and writes answer slots

A tiny scratchpad auxiliary is retained during replay (weight 0.002) so the
carry-trace format is trained; a larger weight was measured to dominate the
short easy/hard losses and collapse accuracy.
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
from backend.training.checkpoints import save_checkpoint  # noqa: E402
from backend.training.datasets import (  # noqa: E402
    build_supervised_tokens,
    collate_lm,
    make_mixed_dataset,
)
from backend.training.losses import language_model_loss  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--easy-steps", type=int, default=90)
p.add_argument("--easy-batch", type=int, default=128)
p.add_argument("--easy-lr", type=float, default=1.5e-3)
p.add_argument("--hard-steps", type=int, default=230)
p.add_argument("--hard-batch", type=int, default=64)
p.add_argument("--hard-lr", type=float, default=2e-3)
p.add_argument("--replay-steps", type=int, default=80)
p.add_argument("--replay-batch", type=int, default=48)
p.add_argument("--replay-lr", type=float, default=1e-4)
p.add_argument("--scratch-weight", type=float, default=0.002)
p.add_argument("--train-size", type=int, default=1600)
p.add_argument("--eval-examples", type=int, default=64)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--checkpoint-id", default="will-2m-synergy-v2")
args = p.parse_args()

if args.threads > 0:
    torch.set_num_threads(args.threads)
CFG = load_config("configs/will_2m.yaml")
TOK = ByteTokenizer()
DEVICE = torch.device("cpu")
CID = args.checkpoint_id
SEED = 1337
torch.manual_seed(SEED)
random.seed(SEED)
MCFG = CFG.model
model = WILLModel(MCFG).to(DEVICE)
model.memory_logit_alpha.requires_grad_(False)


def scratchpad(a: int, b: int) -> str:
    o = (a % 10 + b % 10) % 10
    c1 = (a % 10 + b % 10) // 10
    s2 = a // 10 + b // 10 + c1
    return f"{o},{c1},{s2 % 10},{s2 // 10},{a + b}"


def numbers(prompt: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\s*\+\s*(\d+)", prompt)
    assert match
    return int(match.group(1)), int(match.group(2))


def rebuild(item: dict, target: str, prompt: str | None = None) -> dict:
    prompt = item["prompt"] if prompt is None else prompt
    ids, labels, mask, prompt_ids, _ = build_supervised_tokens(
        TOK, prompt, target, MCFG.max_seq_len
    )
    return {
        **item,
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "prompt_ids": prompt_ids,
        "eval_prompt_ids": prompt_ids,
    }


train = make_mixed_dataset(
    "arith_add", "train", args.train_size, SEED, TOK, MCFG.max_seq_len,
    easy_ratio=0.5,
)
val = make_mixed_dataset(
    "arith_add", "val", 400, SEED, TOK, MCFG.max_seq_len, easy_ratio=0.5
)
easy_rows, hard_rows, scratch_rows, mem_rows = [], [], [], []
for item in train.items:
    a, b = numbers(item["prompt"])
    if item["complexity"] == 1:
        easy_rows.append(rebuild(item, item["answer_text"]))
    else:
        hard_rows.append(rebuild(item, item["answer_text"]))
        scratch_rows.append(
            rebuild(item, scratchpad(a, b), item["prompt"] + " carry")
        )
        mem_rows.append(rebuild(item, item["answer_text"]))
val_easy = [x for x in val.items if x["complexity"] == 1][: args.eval_examples]
val_hard = [x for x in val.items if x["complexity"] == 2][: args.eval_examples]
print(
    f"rows easy={len(easy_rows)} hard={len(hard_rows)} "
    f"val_easy={len(val_easy)} val_hard={len(val_hard)}",
    flush=True,
)


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
            model, ids, mask, [x["answer_text"] for x in rows], TOK, ridge=1e-6
        )
    logits, stats = model(
        ids,
        mask,
        memory_state=state,
        mode="fixed_depth",
        max_steps=depth,
        memory_enabled=memory,
    )
    loss = language_model_loss(logits, labels)
    slot_loss = logits.new_zeros(())
    if memory:
        targets = answer_slot_targets(
            [x["answer_text"] for x in rows], TOK, MCFG.memory_slots, DEVICE
        )
        slot = stats["memory_slot_logits"]
        slot_loss = F.cross_entropy(
            slot.reshape(-1, slot.shape[-1]), targets.reshape(-1)
        )
    return loss, slot_loss


@torch.no_grad()
def evaluate(rows, depth, memory=False, max_new=6, parse_mode="first"):
    model.eval()
    prompts = [list(x.get("eval_prompt_ids", x["prompt_ids"])) for x in rows]
    state = None
    if memory:
        ids, mask, positions, _ = pad_prompts(
            prompts, TOK.pad_id, left=True, max_len=MCFG.max_seq_len
        )
        state = write_slot_memory(
            model, ids, mask, [x["answer_text"] for x in rows], TOK,
            positions=positions, ridge=1e-6,
        )
    out = model.generate_batch(
        prompts, TOK, memory_state=state, mode="fixed_depth",
        max_steps=depth, max_new_tokens=max_new, stop_at_eos=True,
    )
    correct = 0
    for ids, row in zip(out["generated_ids"], rows):
        text = TOK.decode(ids, skip_special=True)
        nums = re.findall(r"-?\d+", text)
        pred = nums[-1] if parse_mode == "last" and nums else (nums[0] if nums else "")
        correct += int(pred == str(row["answer_text"]).strip())
    model.train()
    return {"accuracy": correct / len(rows), "mean_depth": out["mean_depth"]}


def optimizer_for(lr):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=0.01,
    )


def save(phase, history, opt, extra, cid):
    tcfg = TrainingConfig(
        task="arith_add", mixed_easy_ratio=0.4, min_complexity=1,
        max_complexity=2, mode="fixed_depth", precision="fp32", seed=SEED,
        max_steps=MCFG.max_steps, train_size=args.train_size,
        val_size=len(val), test_size=200, batch_size=args.replay_batch,
        checkpoint_name=cid, profile="custom", device="cpu",
    )
    metrics = {
        "run_id": cid,
        "phase": phase,
        "history": history,
        "curriculum": {
            "easy_warmup_steps": args.easy_steps,
            "hard_direct_steps": args.hard_steps,
            "replay_steps": args.replay_steps,
            "final_mix": {"easy": 0.4, "hard": 0.4, "memory": 0.2},
            "scratchpad_format": "ones,carry1,tens,carry2,answer",
            "scratchpad_loss_weight": args.scratch_weight,
            "easy_depth": 1,
            "hard_depth": MCFG.max_steps,
            "memory_depth": 1,
        },
        **extra,
    }
    save_checkpoint(
        CFG.server.checkpoint_dir, cid, model, MCFG, tcfg, opt, metrics,
        history, extra=metrics,
    )


history, start = [], time.perf_counter()

print("[E] easy warmup", flush=True)
opt = optimizer_for(args.easy_lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.easy_steps, eta_min=args.easy_lr * 0.2
)
for step in range(args.easy_steps):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, _ = batch_loss(sample(easy_rows, args.easy_batch), 1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 20 == 0:
        print(f"E {step}/{args.easy_steps} {float(loss.detach()):.4f}", flush=True)
easy = evaluate(val_easy, 1)
history.append({"phase": "easy", "metrics": easy})
print(f"[E] easy={easy['accuracy']:.3f}", flush=True)
save("easy", history, opt, {"easy": easy}, CID + "-easy")

print("[H] hard direct", flush=True)
opt = optimizer_for(args.hard_lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.hard_steps, eta_min=args.hard_lr * 0.1
)
for step in range(args.hard_steps):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, _ = batch_loss(sample(hard_rows, args.hard_batch), MCFG.max_steps)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 25 == 0:
        print(f"H {step}/{args.hard_steps} {float(loss.detach()):.4f}", flush=True)
hard = evaluate(val_hard, MCFG.max_steps, max_new=6)
history.append({"phase": "hard", "metrics": hard})
print(f"[H] hard={hard['accuracy']:.3f}", flush=True)
save("hard", history, opt, {"hard": hard}, CID + "-hard")

print("[R] 40/40/20 replay", flush=True)
opt = optimizer_for(args.replay_lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.replay_steps, eta_min=args.replay_lr * 0.2
)
n_easy = int(round(args.replay_batch * 0.4))
n_hard = int(round(args.replay_batch * 0.4))
n_mem = args.replay_batch - n_easy - n_hard
n_scratch = max(1, int(n_hard * 0.2))
n_direct = n_hard - n_scratch
for step in range(args.replay_steps):
    model.train()
    e = sample(easy_rows, n_easy)
    h = sample(hard_rows, n_direct)
    s = sample(scratch_rows, n_scratch)
    m = sample(mem_rows, n_mem)
    opt.zero_grad(set_to_none=True)
    le, _ = batch_loss(e, 1)
    lh, _ = batch_loss(h, MCFG.max_steps)
    ls, _ = batch_loss(s, MCFG.max_steps)
    lm, slot = batch_loss(m, 1, memory=True)
    loss = 0.4 * le + 0.32 * lh + args.scratch_weight * ls + 0.2 * lm + 0.5 * slot
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 10 == 0:
        print(f"R {step}/{args.replay_steps} {float(loss.detach()):.4f}", flush=True)
easy = evaluate(val_easy, 1)
hard = evaluate(val_hard, MCFG.max_steps, max_new=6)
mem = evaluate(val_hard[:32], 1, memory=True, max_new=6)
final = {"easy": easy, "hard": hard, "memory": mem}
history.append({"phase": "replay", "metrics": final})
print(
    f"[R] easy={easy['accuracy']:.3f} hard={hard['accuracy']:.3f} "
    f"mem={mem['accuracy']:.3f}",
    flush=True,
)
save("replay", history, opt, final, CID)
print("SAVED", CID, "params", model.parameter_counts()[0], flush=True)
print("elapsed_s", round(time.perf_counter() - start, 2), flush=True)
