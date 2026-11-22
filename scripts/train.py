"""Train WILL from the command line.

Examples:
    python scripts/train.py --task arith_add --min-complexity 1 --max-complexity 1 \
        --mode fixed_depth --max-steps 1 --epochs 30 --checkpoint-name add-d1

    python scripts/train.py --config configs/arith_add.yaml --mode adaptive_soft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SCALING_PRESETS, TrainingConfig, load_config, scaling_model_config
from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer
from backend.training.trainer import Trainer, resolve_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the WILL model")
    p.add_argument("--config", default=None, help="YAML AppConfig path")
    p.add_argument("--task", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--val-size", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--optimizer", default=None)
    p.add_argument("--scheduler", default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--ponder-lambda", type=float, default=None)
    p.add_argument("--ponder-warmup-epochs", type=int, default=None)
    p.add_argument("--mode", default=None, choices=["fixed_depth", "adaptive_soft"])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--min-steps", type=int, default=None)
    p.add_argument("--halt-epsilon", type=float, default=None)
    p.add_argument("--min-complexity", type=int, default=None)
    p.add_argument("--max-complexity", type=int, default=None)
    p.add_argument("--mixed-easy-ratio", type=float, default=None)
    p.add_argument("--scaling-preset", choices=list(SCALING_PRESETS), default=None)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16", "auto"], default=None)
    p.add_argument("--torch-compile", action="store_true", default=None)
    p.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--checkpoint-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json-out", default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    app_cfg = load_config(args.config)
    model_cfg = app_cfg.model
    training = app_cfg.training

    if args.profile:
        # Profiles are applied by load_config; an explicit override still
        # needs the matching model/training defaults.
        from backend.config import PROFILE_OVERRIDES, _deep_merge

        merged = _deep_merge(
            {"profile": args.profile}, PROFILE_OVERRIDES.get(args.profile, {})
        )
        model_cfg = type(model_cfg)(**{**model_cfg.model_dump(), **merged.get("model", {})})
        training = type(training)(**{**training.model_dump(), **merged.get("training", {})})
        app_cfg.profile = args.profile

    if args.scaling_preset:
        model_cfg = scaling_model_config(args.scaling_preset)
        app_cfg.profile = "custom"

    overrides = {
        "task": args.task,
        "epochs": args.epochs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "ponder_lambda": args.ponder_lambda,
        "ponder_warmup_epochs": args.ponder_warmup_epochs,
        "mode": args.mode,
        "max_steps": args.max_steps,
        "min_complexity": args.min_complexity,
        "max_complexity": args.max_complexity,
        "mixed_easy_ratio": args.mixed_easy_ratio,
        "precision": args.precision,
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "seed": args.seed,
        "eval_every": args.eval_every,
        "early_stop_patience": args.early_stop_patience,
        "checkpoint_name": args.checkpoint_name,
        "profile": args.profile,
    }
    data = training.model_dump()
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    training = TrainingConfig(**data)

    if args.max_steps and args.max_steps > model_cfg.max_steps:
        model_cfg = model_cfg.model_copy(update={"max_steps": args.max_steps})
    if args.min_steps is not None:
        model_cfg = model_cfg.model_copy(update={"min_steps": args.min_steps})
    if args.halt_epsilon is not None:
        model_cfg = model_cfg.model_copy(update={"halt_epsilon": args.halt_epsilon})
    if model_cfg.min_steps > model_cfg.max_steps:
        raise SystemExit("--min-steps cannot exceed --max-steps")

    tokenizer = ByteTokenizer()
    model = WILLModel(model_cfg)
    device = resolve_device(args.device)

    def progress(event: dict) -> None:
        if args.quiet:
            return
        if event.get("type") == "step":
            print(
                f"epoch {event['epoch']}/{event['epochs']} "
                f"step {event['step']}/{event['steps_per_epoch']} "
                f"loss={event['loss']:.4f} depth={event['mean_depth']:.2f} "
                f"lr={event['lr']:.2e}",
                flush=True,
            )
        elif event.get("type") == "eval":
            acc = event.get("val_accuracy")
            acc_text = f"{acc:.3f}" if isinstance(acc, float) else "n/a"
            print(
                f"  eval epoch {event['epoch']}: val_loss={event['val_loss']:.4f} "
                f"val_acc={acc_text} depth={event['val_mean_depth']:.2f}",
                flush=True,
            )
        elif event.get("type") == "checkpoint":
            print(f"  checkpoint: {event['checkpoint_id']}", flush=True)
        elif event.get("type") == "early_stop":
            print(f"  early stop at epoch {event['epoch']}", flush=True)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        training_cfg=training,
        checkpoint_dir=app_cfg.server.checkpoint_dir,
        device=str(device),
        progress_cb=progress,
        resume_from=args.resume,
        run_name=training.checkpoint_name or args.checkpoint_name,
    )
    metrics = trainer.train()
    print(
        json.dumps(
            {
                "checkpoint": metrics.get("checkpoint", {}).get("checkpoint_id"),
                "best_val_loss": metrics["best_val_loss"],
                "best_val_accuracy": metrics["best_val_accuracy"],
                "duration_seconds": round(metrics["duration_seconds"], 2),
                "parameters": metrics["parameters_total"],
                "device": metrics["device"],
                "precision": metrics.get("precision"),
                "compilation": metrics.get("compilation"),
            },
            indent=2,
        )
    )
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
