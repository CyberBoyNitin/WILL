"""Controlled A/B: fixed@4 vs adaptive_hard@3, identical threads/priority."""
import json, math, statistics, sys
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.benchmarks.runner import _eval_dataset
from backend.config import load_config
from backend.engine.tokenizer import ByteTokenizer
from backend.training.checkpoints import load_model_from_checkpoint, resolve_checkpoint

torch.set_num_threads(1)
try:
    psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
except Exception:
    pass

cfg = load_config()
model, mcfg, payload = load_model_from_checkpoint(
    resolve_checkpoint(cfg.server.checkpoint_dir, "add-single-act-final"), device="cpu"
)
ds, _, _ = _eval_dataset(payload, ByteTokenizer(), {"split": "test", "limit": 200})
prompts = [list(x["prompt_ids"]) for x in ds.items]
tok = ByteTokenizer()

def p95(samples):
    s = sorted(samples)
    return s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]

rows = []
for mode, depth in (("fixed_depth", 4), ("adaptive_hard", 3)):
    for _ in range(5):
        model.generate_batch(prompts[:16], tok, mode=mode, max_steps=depth, max_new_tokens=12)
    samples, last = [], None
    for _ in range(50):
        t0 = __import__("time").perf_counter()
        last = model.generate_batch(prompts, tok, mode=mode, max_steps=depth, max_new_tokens=12)
        samples.append((__import__("time").perf_counter() - t0) * 1000.0)
    texts = [tok.decode(g, skip_special=True) for g in last["generated_ids"]]
    from backend.training.datasets import normalize_answer
    correct = sum(normalize_answer(t) == normalize_answer(x["answer_text"])
                  for t, x in zip(texts, ds.items))
    rows.append({
        "config": f"{mode}@{depth}",
        "accuracy_pct": round(100 * correct / len(ds), 4),
        "mean_depth": last["mean_depth"],
        "token_evaluations": last["token_evaluations"],
        "p50_ms": round(statistics.median(samples), 4),
        "p95_ms": round(p95(samples), 4),
        "runs": len(samples),
    })

f4 = rows[0]["p50_ms"]
for r in rows:
    r["speedup_vs_fixed4_p50"] = round(f4 / r["p50_ms"], 4)
out = {"threads": torch.get_num_threads(), "priority": "high", "warmup": 5, "runs": 50,
       "slow_parameters": sum(p.numel() for p in model.parameters()), "results": rows}
p = Path(cfg.server.experiment_dir) / "lead_eval" / "lead_eval-9e1899be630c" / "test2_controlled.json"
p.write_text(json.dumps(out, indent=2), encoding="utf-8")
for r in rows:
    print(r)
