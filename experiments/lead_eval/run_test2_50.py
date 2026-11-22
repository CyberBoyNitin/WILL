"""Re-run Test 2 with 50 timed generation repeats (5 warmup inside the suite)."""
import json, math, statistics, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.benchmarks.runner import run_benchmark_suite
from backend.config import load_config
from backend.engine.metrics import count_parameters
from backend.engine.tokenizer import ByteTokenizer
from backend.training.checkpoints import load_model_from_checkpoint, resolve_checkpoint

cfg = load_config()
model, mcfg, payload = load_model_from_checkpoint(
    resolve_checkpoint(cfg.server.checkpoint_dir, "add-single-act-final"), device="cpu"
)
rep = run_benchmark_suite(cfg, "adaptive_compute", {
    "checkpoint_id": "add-single-act-final", "split": "test", "limit": 200,
    "depths": [1, 2, 3, 4], "modes": ["adaptive_soft", "adaptive_hard"],
    "batch_size": 16, "max_new_tokens": 12, "generation_repeats": 50,
    "max_examples": 0, "compile_probe": False, "device": "cpu",
})

def pct(samples, q):
    s = sorted(samples)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]

def gen_tokens(mode, depth):
    ds = __import__("backend.benchmarks.runner", fromlist=["_eval_dataset"])._eval_dataset(
        payload, ByteTokenizer(), {"split": "test", "limit": 200}
    )[0]
    prompts = [list(x["prompt_ids"]) for x in ds.items]
    total = 0
    for i in range(0, len(prompts), 16):
        out = model.generate_batch(prompts[i:i+16], ByteTokenizer(), mode=mode,
                                   max_steps=depth, max_new_tokens=12)
        total += sum(len(g) for g in out["generated_ids"])
    return total

fixed4 = next(r for r in rep["results"] if r["mode"] == "fixed_depth" and r["max_steps"] == 4)
f4p50 = statistics.median(fixed4["generation_seconds_all"])
rows = []
for r in rep["results"]:
    samples = r["generation_seconds_all"]
    p50 = statistics.median(samples)
    p95 = pct(samples, 0.95)
    tok = gen_tokens(r["mode"], r["max_steps"])
    rows.append({
        "config": f"{r['mode']}@{r['max_steps']}",
        "accuracy_pct": round(100 * (r["accuracy"] or 0), 4),
        "mean_depth": r["generation_mean_depth"],
        "token_evaluations": r["generation_token_evaluations"],
        "estimated_macs": r["generation_estimated_macs"],
        "p50_seconds": round(p50, 4),
        "p95_seconds": round(p95, 4),
        "runs": len(samples),
        "speedup_vs_fixed4_p50": round(f4p50 / p50, 4),
        "generated_tokens": tok,
        "tokens_per_second_p50": round(tok / p50, 2),
    })

out = {
    "suite_report_id": rep["run_id"],
    "dataset": rep["dataset"],
    "slow_parameters": count_parameters(model),
    "warmup_policy": "suite performs one full warmup pass before the 50 timed passes",
    "results": rows,
}
target = Path(cfg.server.experiment_dir) / "lead_eval" / "lead_eval-9e1899be630c" / "test2_50runs.json"
target.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("REPORT_ID", rep["run_id"])
print("{:22s} {:>7s} {:>6s} {:>9s} {:>9s} {:>9s} {:>8s} {:>9s}".format(
    "config", "acc%", "depth", "p50_s", "p95_s", "speedup", "tok/s", "runs"))
for x in rows:
    print("{:22s} {:7.2f} {:6.3f} {:9.4f} {:9.4f} {:9.3f} {:8.1f} {:9d}".format(
        x["config"], x["accuracy_pct"], x["mean_depth"], x["p50_seconds"],
        x["p95_seconds"], x["speedup_vs_fixed4_p50"], x["tokens_per_second_p50"], x["runs"]))
