"""Train will-8m-synergy-v1 with the target-driven 3-phase curriculum.

Phase A: warm-start reasoning (fixed depth 4) until easy>98% and hard>85%.
Phase B: accuracy-gated ponder fine-tune until easy depth<1.3, hard>2.8.
Phase C: memory-conditioned 1-step training using the Test 5 key/value writer.

Resumable: --resume <checkpoint_id> loads a saved working checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import TrainingConfig, load_config  # noqa: E402
from backend.engine.model import WILLModel  # noqa: E402
from backend.engine.tokenizer import BOS_ID, ByteTokenizer  # noqa: E402
from backend.memory.fast_memory import FastWeightMemory  # noqa: E402
from backend.memory.memory_encoder import pool_memory_tokens  # noqa: E402
from backend.memory.memory_manager import MemoryManager  # noqa: E402
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
from backend.training.losses import language_model_loss, total_loss  # noqa: E402
from backend.training.trainer import evaluate_generation  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--resume", default=None)
p.add_argument("--train-size", type=int, default=1600)
p.add_argument("--phase-a-epochs", type=int, default=10)
p.add_argument("--phase-b-epochs", type=int, default=8)
p.add_argument("--phase-c-epochs", type=int, default=2)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--phase", choices=["all", "A", "B", "C"], default="all")
p.add_argument("--focus", choices=["mixed", "hard", "easy"], default="mixed")
args = p.parse_args()

CFG = load_config("configs/will_8m.yaml")
TOK = ByteTokenizer()
DEVICE = torch.device("cpu")
CID = "will-8m-synergy-v1"
torch.manual_seed(1337)
random.seed(1337)

if args.resume:
    model, mcfg, payload = load_model_from_checkpoint(
        resolve_checkpoint(CFG.server.checkpoint_dir, args.resume), device=DEVICE
    )
    history = list(payload.get("history") or [])
    metrics = dict(payload.get("metrics") or {})
    metrics.setdefault("run_id", CID)
    metrics.setdefault("phases", {})
    print("resumed", args.resume, "history", len(history), flush=True)
else:
    mcfg = CFG.model
    model = WILLModel(mcfg).to(DEVICE)
    history = []
    metrics = {"run_id": CID, "phases": {}}

seed = 1337
val_ds = make_mixed_dataset("arith_add", "test", 200, seed, TOK,
                            mcfg.max_seq_len, easy_ratio=0.5)
easy_val = [i for i, x in enumerate(val_ds.items) if x["complexity"] == 1]
hard_val = [i for i, x in enumerate(val_ds.items) if x["complexity"] == 2]

class IndexSubset:
    def __init__(self, ds, idxs):
        self.ds, self.idxs = ds, idxs
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        return self.ds[self.idxs[i]]

def strat_metrics(mode, depth, batch=32):
    out = {}
    for label, idxs in (("easy", easy_val), ("hard", hard_val)):
        g = evaluate_generation(model, IndexSubset(val_ds, idxs), TOK, DEVICE,
                                mode, depth, batch_size=batch,
                                max_new_tokens=6, max_examples=0)
        out[label] = {"acc": g["accuracy"], "depth": g["mean_depth"]}
    return out

def train_epoch(rows, mode, depth, objective, lr, accuracy_gated=False, ponder=0.0):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    random.shuffle(rows)
    total, seen = 0.0, 0
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        batch = collate_lm([{
            "input_ids": x["input_ids"], "labels": x["labels"],
            "attention_mask": x["attention_mask"],
        } for x in chunk])
        ids, labels, mask = (batch[k].to(DEVICE) for k in
                             ("input_ids", "labels", "attention_mask"))
        opt.zero_grad(set_to_none=True)
        logits, stats = model(ids, mask, mode=mode, max_steps=depth,
                              memory_enabled=False)
        loss, _parts = total_loss(logits, labels, stats,
                                  ponder_lambda=ponder,
                                  accuracy_gated=accuracy_gated)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.detach().item()) * len(chunk)
        seen += len(chunk)
    return total / max(1, seen)

def save(metrics, history, opt=None):
    tcfg = TrainingConfig(
        task="arith_add", mixed_easy_ratio=0.5, min_complexity=1,
        max_complexity=2, mode="adaptive_soft", precision="fp32",
        ponder_lambda=0.02, accuracy_gated_ponder=True, seed=seed,
        max_steps=mcfg.max_steps, train_size=args.train_size,
        val_size=200, test_size=200, batch_size=args.batch,
        checkpoint_name=CID, profile="custom", device="cpu",
    )
    return save_checkpoint(CFG.server.checkpoint_dir, CID, model, mcfg,
                           tcfg, opt, metrics, history, extra=metrics)

train_ds = make_mixed_dataset("arith_add", "train", args.train_size, seed,
                              TOK, mcfg.max_seq_len, easy_ratio=0.5)
rows = [train_ds[i] for i in range(len(train_ds))]
if args.focus == "hard":
    rows = [x for x in rows if x["complexity"] == 2]
elif args.focus == "easy":
    rows = [x for x in rows if x["complexity"] == 1]
print("focus", args.focus, "rows", len(rows), flush=True)

# ------------------------------------------------------------- Phase A
if args.phase in ("all", "A"):
    print("[A] warm-start reasoning", flush=True)
    a_start = sum(1 for x in history if x.get("phase") == "A")
    for epoch in range(a_start, a_start + args.phase_a_epochs):
        t0 = time.perf_counter()
        loss = train_epoch(rows, "fixed_depth", mcfg.max_steps, "lm", 2e-4)
        m = strat_metrics("fixed_depth", mcfg.max_steps)
        history.append({"phase": "A", "epoch": epoch, "loss": loss, **m})
        print(f"A ep{epoch} loss={loss:.4f} easy={m['easy']['acc']:.3f} "
              f"hard={m['hard']['acc']:.3f} {time.perf_counter()-t0:.0f}s", flush=True)
        metrics["phases"]["A"] = m
        save(metrics, history)
        if m["easy"]["acc"] >= 0.98 and m["hard"]["acc"] >= 0.85:
            print("A targets met", flush=True)
            break

# ------------------------------------------------------------- Phase B
if args.phase in ("all", "B"):
    print("[B] accuracy-gated ponder fine-tune", flush=True)
    b_start = sum(1 for x in history if x.get("phase") == "B")
    for epoch in range(b_start, b_start + args.phase_b_epochs):
        t0 = time.perf_counter()
        lam = 0.02 * min(1.0, (epoch + 1) / max(1, args.phase_b_epochs // 3))
        loss = train_epoch(rows, "adaptive_soft", mcfg.max_steps, "ponder",
                           1.5e-4, accuracy_gated=True, ponder=lam)
        m = strat_metrics("adaptive_hard", mcfg.resolved_adaptive_steps())
        history.append({"phase": "B", "epoch": epoch, "loss": loss, **m})
        print(f"B ep{epoch} loss={loss:.4f} easy_d={m['easy']['depth']:.3f} "
              f"hard_d={m['hard']['depth']:.3f} easy_a={m['easy']['acc']:.3f} "
              f"hard_a={m['hard']['acc']:.3f} {time.perf_counter()-t0:.0f}s", flush=True)
        metrics["phases"]["B"] = m
        save(metrics, history)
        if m["easy"]["depth"] < 1.3 and m["hard"]["depth"] > 2.8:
            print("B targets met", flush=True)
            break

# ------------------------------------------------------------- Phase C
if args.phase in ("all", "C"):
    print("[C] memory-conditioned 1-step training", flush=True)
    hard_rows = [x for x in train_ds.items if x["complexity"] == 2]
    mem_dir = Path(CFG.server.experiment_dir) / "will8m_mem"
    mem_dir.mkdir(parents=True, exist_ok=True)
    manager = MemoryManager(FastWeightMemory(mcfg), CFG.memory, mem_dir, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    def encode(texts, max_len):
        seqs = [[BOS_ID] + TOK.encode(t, max_len=max_len) for t in texts]
        width = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), width), dtype=torch.long, device=DEVICE)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=DEVICE)
            mask[i, :len(s)] = 1
        with torch.no_grad():
            _, k_all, v_all = model.encode_memory(ids, mask)
        k = torch.stack([pool_memory_tokens(k_all[i, 1:int(mask[i].sum().item())])
                         for i in range(len(seqs))])
        v = torch.stack([pool_memory_tokens(v_all[i, 1:int(mask[i].sum().item())])
                         for i in range(len(seqs))])
        return k, v

    c_start = sum(1 for x in history if x.get("phase") == "C")
    for epoch in range(c_start, c_start + args.phase_c_epochs):
        random.shuffle(hard_rows)
        total, seen = 0.0, 0
        model.train()
        for start in range(0, len(hard_rows), args.batch):
            chunk = hard_rows[start:start + args.batch]
            kp, _ = encode([x["prompt"] for x in chunk], 64)
            _, va = encode([x["answer_text"] for x in chunk], 32)
            sess = manager.create_session(f"c-{epoch}-{start}")
            state = manager.fast_memory.consolidate(
                manager.get_session(sess.session_id).state, kp, va
            )
            manager.get_session(sess.session_id).state = state
            rows_b = [build_supervised_tokens(TOK, x["prompt"],
                                              x["answer_text"], mcfg.max_seq_len)
                      for x in chunk]
            b = collate_lm([{"input_ids": r[0], "labels": r[1],
                             "attention_mask": r[2]} for r in rows_b])
            ids, labels, mask = (b[k].to(DEVICE) for k in
                                 ("input_ids", "labels", "attention_mask"))
            opt.zero_grad(set_to_none=True)
            logits, _stats = model(ids, mask, memory_state=state,
                                   mode="fixed_depth", max_steps=1,
                                   memory_enabled=True)
            loss = language_model_loss(logits, labels)
            # Align the bridge query with the exact key vector written into the
            # fast state.  Without this the bridge has no supervised target and
            # the retrieved value is effectively noise.
            q = _stats.get("memory_query")
            if q is not None:
                align = 1.0 - F.cosine_similarity(
                    F.normalize(q, dim=-1), F.normalize(kp, dim=-1), dim=-1
                ).mean()
                loss = loss + 0.5 * align
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().item()) * len(chunk)
            seen += len(chunk)
            manager.delete_session(sess.session_id)
        # memory-conditioned accuracy on held-out hard prompts
        model.eval()
        eval_rows = hard_val[:32]
        correct = 0
        for start in range(0, len(eval_rows), 16):
            chunk = [val_ds[i] for i in eval_rows[start:start + 16]]
            kp, _ = encode([x["prompt"] for x in chunk], 64)
            _, va = encode([x["answer_text"] for x in chunk], 32)
            sess = manager.create_session(f"ce-{epoch}-{start}")
            state = manager.fast_memory.consolidate(
                manager.get_session(sess.session_id).state, kp, va
            )
            manager.get_session(sess.session_id).state = state
            prompts = [list(x["prompt_ids"]) for x in chunk]
            out = model.generate_batch(prompts, TOK, mode="fixed_depth",
                                       max_steps=1, max_new_tokens=6,
                                       memory_state=state)
            for g, x in zip(out["generated_ids"], chunk):
                correct += int(TOK.decode(g, skip_special=True).strip() ==
                               x["answer_text"].strip())
            manager.delete_session(sess.session_id)
        acc = correct / max(1, len(eval_rows))
        print(f"C ep{epoch} loss={total/max(1,seen):.4f} mem_acc={acc:.3f}",
              flush=True)
        history.append({"phase": "C", "epoch": epoch,
                        "loss": total / max(1, seen), "mem_acc": acc})
        metrics["phases"]["C"] = {"mem_acc": acc}
        save(metrics, history, opt)
        if acc >= 0.90:
            print("C target met", flush=True)
            break

meta = save(metrics, history)
print("SAVED", meta["checkpoint_id"], "params",
      sum(p.numel() for p in model.parameters()), flush=True)
