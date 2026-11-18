"""Evaluate a saved WILL checkpoint on its task's validation/test splits.

Usage:
    python -m scripts.evaluate_checkpoint checkpoints/longctx-d4-final.pt \
        --split test --modes fixed_depth adaptive_soft adaptive_hard --depths 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from backend.engine.tokenizer import ByteTokenizer
from backend.training.checkpoints import load_checkpoint, load_model_from_checkpoint
from backend.training.datasets import collate_lm, make_dataset
from backend.training.trainer import evaluate_generation, evaluate_loss, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to .pt checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--size", type=int, default=None, help="override split size")
    parser.add_argument("--modes", nargs="+", default=["fixed_depth", "adaptive_soft", "adaptive_hard"])
    parser.add_argument("--depths", nargs="+", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, payload = load_model_from_checkpoint(args.checkpoint, device=device)
    tokenizer = ByteTokenizer()
    tcfg = payload["training_config"]
    max_steps = min(tcfg.get("max_steps") or cfg.max_steps, cfg.max_steps)
    depths = args.depths or list(range(1, max_steps + 1))

    ds = make_dataset(
        task=tcfg["task"],
        split=args.split,
        size=args.size or tcfg.get(f"{args.split}_size", 150),
        seed=tcfg["seed"],
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        min_complexity=tcfg.get("min_complexity", 1),
        max_complexity=tcfg.get("max_complexity", 2),
    )
    results: dict[str, dict] = {}
    for mode in args.modes:
        mode_depths = depths if mode != "adaptive_hard" else [max_steps]
        for depth in mode_depths:
            key = f"{mode}@{depth}"
            loss = evaluate_loss(
                model,
                torch.utils.data.DataLoader(
                    ds, batch_size=args.batch_size, collate_fn=collate_lm
                ),
                device,
                mode,
                depth,
            )
            gen = evaluate_generation(
                model, ds, tokenizer, device, mode, depth,
                batch_size=args.batch_size, max_examples=args.max_examples,
            )
            results[key] = {
                "loss": loss,
                "generation": gen,
            }
            gen_acc = gen["accuracy"] if gen["accuracy"] is not None else 0.0
            print(
                f"{key:24s} loss={loss['loss']:.4f} ppl={loss['perplexity']:.3f} "
                f"depth={loss['mean_depth']:.2f} gen_acc={gen_acc:.3f} "
                f"({gen['correct']}/{gen['count']})"
            )
            for ex in gen["examples"][:3]:
                print(f"    gold={ex['gold']!r} pred={ex['prediction']!r}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
