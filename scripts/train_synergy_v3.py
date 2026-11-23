"""Train add-synergy-v3 with the 3-part curriculum.

Part 1+2: mixed 50/50 easy/hard data with ponder-budget loss (Trainer).
Part 3: joint memory-conditioned training of the main network + mem_gate:
facts are written to fast weights, the prompt context carries only the
question, and the model is trained to answer in ONE recurrent step.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import ModelConfig, TrainingConfig, load_config  # noqa: E402
from backend.engine.model import WILLModel  # noqa: E402
from backend.engine.tokenizer import BOS_ID, ByteTokenizer  # noqa: E402
from backend.memory.memory_encoder import pool_memory_tokens  # noqa: E402
from backend.training.checkpoints import save_checkpoint  # noqa: E402
from backend.training.datasets import (  # noqa: E402
    build_supervised_tokens,
    collate_lm,
    make_dataset,
)
from backend.training.losses import language_model_loss  # noqa: E402
from backend.training.trainer import Trainer  # noqa: E402

CFG = load_config()
TOK = ByteTokenizer()
DEVICE = torch.device("cpu")
torch.manual_seed(1337)
random.seed(1337)

# ---------------------------------------------------------------- model cfg
mcfg = ModelConfig(**CFG.model.model_dump())
mcfg.min_steps = 1
mcfg.max_steps = 4
mcfg.adaptive_inference_steps = 4
mcfg.dropout = 0.0
mcfg.memory_dim = 64

def make_tcfg(**overrides):
    base = dict(
        task="arith_add", train_size=1200, val_size=150, test_size=150,
        epochs=12, batch_size=16, lr=5e-4, weight_decay=0.01, grad_clip=1.0,
        grad_accum=1, optimizer="adamw", scheduler="cosine", warmup_steps=20,
        ponder_lambda=0.0, ponder_warmup_epochs=0, memory_lambda=0.0,
        mode="fixed_depth", seed=1337, eval_every=1, early_stop_patience=0,
        max_steps=4, min_complexity=1, max_complexity=2,
        mixed_easy_ratio=0.5, precision="fp32", torch_compile=False,
        compile_mode="reduce-overhead", profile="tiny",
        checkpoint_name="add-synergy-v3", device="cpu",
    )
    base.update(overrides)
    return TrainingConfig(**base)


tcfg = make_tcfg()

# ------------- Part 1: mixed 50/50 data, learn reasoning at full depth
print("[1/3] mixed 50/50 data, fixed-depth reasoning", flush=True)
model = WILLModel(mcfg).to(DEVICE)
trainer_a = Trainer(
    model=model, tokenizer=TOK, model_cfg=mcfg, training_cfg=tcfg,
    checkpoint_dir=CFG.server.checkpoint_dir, device="cpu",
    run_name="add-synergy-v3-phaseA",
)
part_a = trainer_a.train()
print("part A:", {k: part_a.get(k) for k in ("best_val_loss", "parameters_total")}, flush=True)

# ------------- Part 2: ponder-budget fine-tune (0.02 * mean_steps)
print("[2/3] ponder-budget halting-gate fine-tune", flush=True)
tcfg_b = make_tcfg(
    mode="adaptive_soft", epochs=8, lr=2e-4,
    ponder_lambda=0.02, ponder_warmup_epochs=6, warmup_steps=10,
)
trainer_b = Trainer(
    model=model, tokenizer=TOK, model_cfg=mcfg, training_cfg=tcfg_b,
    checkpoint_dir=CFG.server.checkpoint_dir, device="cpu",
    run_name="add-synergy-v3-phaseB",
)
part_b = trainer_b.train()
print("part B:", {k: part_b.get(k) for k in ("best_val_loss", "parameters_total")}, flush=True)

# -------------------------------------------- Part 3: memory-conditioned 1-step
print("[3/4] joint memory-conditioned 1-step training", flush=True)
mem_ds = make_dataset("arith_add", "train", 320, 2024, TOK, mcfg.max_seq_len, complexity=2)
examples = [(x["prompt"], x["answer_text"]) for x in mem_ds.items]
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
model.train()
history: list[dict] = []
MEM_EPOCHS, BATCH = 2, 8
for epoch in range(MEM_EPOCHS):
    random.shuffle(examples)
    total_loss, seen = 0.0, 0
    for start in range(0, len(examples), BATCH):
        chunk = examples[start:start + BATCH]
        prompts = [p for p, _ in chunk]
        answers = [a for _, a in chunk]

        def encode_texts(texts, max_len):
            seqs = [[BOS_ID] + TOK.encode(t, max_len=max_len) for t in texts]
            width = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), width), dtype=torch.long, device=DEVICE)
            mask = torch.zeros_like(ids)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s, device=DEVICE)
                mask[i, :len(s)] = 1
            with torch.no_grad():
                _, k_all, v_all = model.encode_memory(ids, mask)
            k = torch.stack([
                pool_memory_tokens(k_all[i, 1:int(mask[i].sum().item())])
                for i in range(len(seqs))
            ])
            v = torch.stack([
                pool_memory_tokens(v_all[i, 1:int(mask[i].sum().item())])
                for i in range(len(seqs))
            ])
            return k, v

        kp, _ = encode_texts(prompts, 64)
        ka, va = encode_texts(answers, 32)
        # Supporting facts live only in fast weights; the prompt keeps just
        # the question (no facts concatenated into it).
        state = model.memory.init_state(DEVICE)
        model.memory.adapt(state, kp, va)

        rows = [build_supervised_tokens(TOK, p, a, mcfg.max_seq_len)
                for p, a in chunk]
        batch = collate_lm([{
            "input_ids": r[0], "labels": r[1], "attention_mask": r[2],
        } for r in rows])
        ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits, _stats = model(
            ids, mask, memory_state=state, mode="fixed_depth",
            max_steps=1, memory_enabled=True,
        )
        loss = language_model_loss(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(chunk)
        seen += len(chunk)
    history.append({"epoch": epoch, "memory_step_loss": total_loss / max(1, seen)})
    print(f"  memory epoch {epoch}: loss={total_loss / max(1, seen):.4f}", flush=True)

# --------------------------------------------------------------- save v3
print("[4/4] saving add-synergy-v3", flush=True)
metrics = {
    "part_a_best_val_loss": part_a.get("best_val_loss"),
    "part_a_test": part_a.get("test"),
    "part_b_best_val_loss": part_b.get("best_val_loss"),
    "part_b_test": part_b.get("test"),
    "part_c_final_loss": history[-1]["memory_step_loss"],
    "parameters_total": sum(p.numel() for p in model.parameters()),
    "min_steps": mcfg.min_steps,
    "adaptive_inference_steps": mcfg.adaptive_inference_steps,
    "mem_gate_init": 1.0,
    "mem_gate_trained": round(float(torch.sigmoid(model.mem_gate).detach().item()), 6),
}
meta = save_checkpoint(
    CFG.server.checkpoint_dir, "add-synergy-v3", model, mcfg, tcfg,
    optimizer, metrics, history,
    extra={
        "curriculum": [
            "mixed_50_50_easy_hard_fixed_depth_reasoning",
            "adaptive_soft_ponder_0.02_with_2_epoch_warmup",
            "memory_conditioned_one_step_fast_weights",
        ],
        "architecture_changes": [
            "InputEmbedding forward has no .item()",
            "min_steps=1",
            "mem_gate scalar init +1.0, fast-weight residual before recurrent block",
        ],
    },
)
print("SAVED", meta["checkpoint_id"], "params", metrics["parameters_total"],
      "mem_gate", metrics["mem_gate_trained"], flush=True)
