"""Training loop, evaluation and checkpoint orchestration."""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.config import ModelConfig, TrainingConfig
from backend.engine.compilation import maybe_compile
from backend.engine.metrics import count_parameters
from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer
from backend.training.precision import (
    PrecisionPlan,
    autocast_context,
    build_grad_scaler,
    resolve_precision,
)
from backend.training.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    safe_checkpoint_id,
)
from backend.training.datasets import (
    collate_lm,
    dataset_manifest,
    make_dataset,
    make_mixed_dataset,
    normalize_answer,
)
from backend.training.losses import language_model_loss, total_loss

ProgressCallback = Callable[[dict[str, Any]], None]


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if cfg.scheduler == "none":
        return None
    warmup = max(0, cfg.warmup_steps)
    total = max(1, total_steps)

    def fn(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        if cfg.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate_loss(
    model: WILLModel,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    max_steps: int,
    halt_epsilon: float | None = None,
    precision_plan: PrecisionPlan | None = None,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_ponder = 0.0
    total_depth = 0.0
    total_seq = 0
    total_evaluations = 0
    total_macs = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
        context = (
            autocast_context(precision_plan)
            if precision_plan is not None
            else nullcontext()
        )
        with context:
            logits, stats = model(
                ids,
                mask,
                mode=mode,
                max_steps=max_steps,
                halt_epsilon=halt_epsilon,
                memory_enabled=False,
            )
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n = int((labels != -100).sum().item())
        total_nll += float(nll.item())
        total_tokens += n
        total_ponder += float(stats["ponder_cost"].detach().item()) * ids.shape[0]
        total_depth += float(stats["mean_depth"]) * ids.shape[0]
        total_seq += ids.shape[0]
        total_evaluations += int(stats.get("total_token_evaluations", 0))
        total_macs += int(stats.get("estimated_macs", 0))
    loss = total_nll / max(1, total_tokens)
    return {
        "loss": loss,
        "perplexity": float(math.exp(min(loss, 50.0))),
        "ponder_cost": total_ponder / max(1, total_seq),
        "mean_depth": total_depth / max(1, total_seq),
        "tokens": total_tokens,
        "sequences": total_seq,
        "token_evaluations": total_evaluations,
        "estimated_macs": total_macs,
    }


@torch.no_grad()
def evaluate_generation(
    model: WILLModel,
    dataset,
    tokenizer: ByteTokenizer,
    device: torch.device,
    mode: str,
    max_steps: int,
    batch_size: int = 32,
    max_new_tokens: int = 24,
    max_examples: int = 12,
    halt_epsilon: float | None = None,
    precision_plan: PrecisionPlan | None = None,
) -> dict[str, Any]:
    """Generate every held-out answer with the incremental cache path.

    Prompts are left-padded into a batch, prefilled once, and then decoded with
    one row-compacted KV cache.  This keeps the cache exact while avoiding a
    separate Python generation loop per example.
    """
    if len(dataset) == 0:
        return {"accuracy": None, "count": 0, "examples": []}
    correct = 0
    total = 0
    examples: list[dict[str, Any]] = []
    model.eval()
    generation_start = time.perf_counter()
    total_evaluations = 0
    total_macs = 0
    depth_sum = 0.0
    depth_seqs = 0
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[i] for i in range(start, min(start + batch_size, len(dataset)))
        ]
        prompts = [
            list(item["prompt_ids"])[: model.cfg.max_seq_len]
            or [tokenizer.bos_id]
            for item in items
        ]
        context = (
            autocast_context(precision_plan)
            if precision_plan is not None
            else nullcontext()
        )
        with context:
            out = model.generate_batch(
                prompts,
                tokenizer,
                mode=mode,
                max_steps=max_steps,
                halt_epsilon=halt_epsilon,
                max_new_tokens=max_new_tokens,
                stop_at_eos=True,
            )
        total_evaluations += int(out.get("token_evaluations", 0))
        total_macs += int(out.get("estimated_macs", 0))
        batch_depth_count = int(out.get("depth_count", 0))
        if batch_depth_count:
            depth_sum += float(out.get("mean_depth", 0.0)) * batch_depth_count
            depth_seqs += batch_depth_count
        for local, item in enumerate(items):
            pred_ids = list(out["generated_ids"][local])
            if tokenizer.eos_id in pred_ids:
                pred_ids = pred_ids[: pred_ids.index(tokenizer.eos_id)]
            pred = tokenizer.decode(pred_ids, skip_special=True)
            gold = item["answer_text"]
            ok = normalize_answer(pred) == normalize_answer(gold)
            correct += int(ok)
            total += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "prompt": item["prompt"][:220],
                        "gold": gold,
                        "prediction": pred,
                        "correct": ok,
                        "complexity": item["complexity"],
                    }
                )
    return {
        "accuracy": correct / max(1, total),
        "correct": correct,
        "count": total,
        "examples": examples,
        "mean_depth": depth_sum / max(1, depth_seqs),
        "token_evaluations": total_evaluations,
        "estimated_macs": total_macs,
        "elapsed_seconds": time.perf_counter() - generation_start,
    }


class Trainer:
    def __init__(
        self,
        model: WILLModel,
        tokenizer: ByteTokenizer,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        checkpoint_dir: str | Path,
        device: str | None = None,
        progress_cb: ProgressCallback | None = None,
        stop_event=None,
        resume_from: str | None = None,
        run_name: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.cfg = training_cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.precision_plan = resolve_precision(training_cfg.precision, self.device)
        self.scaler = build_grad_scaler(self.precision_plan)
        self.train_model = self.model
        self.compilation_status: dict[str, Any] = {
            "requested": bool(training_cfg.torch_compile),
            "compiled": False,
            "used": "eager",
            "mode": training_cfg.compile_mode,
            "checked": False,
            "reason": "compilation not requested",
            "generation_path": "eager",
        }
        self.progress_cb = progress_cb
        self.stop_event = stop_event
        self.history: list[dict[str, Any]] = []
        self.run_name = safe_checkpoint_id(
            run_name or f"{training_cfg.task}-{training_cfg.profile}-{int(time.time())}"
        )
        self.resume_payload: dict[str, Any] | None = None
        self.start_epoch = 0

        torch.manual_seed(training_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_cfg.seed)

        max_steps = training_cfg.max_steps or model_cfg.max_steps
        self.max_steps = min(max_steps, model_cfg.max_steps)
        self.eval_mode = (
            "fixed_depth" if training_cfg.mode == "fixed_depth" else "adaptive_hard"
        )
        if training_cfg.torch_compile:
            self._setup_compilation()
        self.train_ds = self._make_split("train", training_cfg.train_size)
        self.val_ds = self._make_split("val", training_cfg.val_size)
        self.test_ds = self._make_split("test", training_cfg.test_size)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=training_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_lm,
            num_workers=0,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=max(8, training_cfg.batch_size),
            shuffle=False,
            collate_fn=collate_lm,
            num_workers=0,
        )
        self.optimizer = build_optimizer(self.model, training_cfg)
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / training_cfg.grad_accum))
        self.scheduler = build_scheduler(
            self.optimizer, training_cfg, steps_per_epoch * training_cfg.epochs
        )
        if resume_from:
            self._resume(resume_from)

    def _make_split(self, split: str, size: int):
        if self.cfg.mixed_easy_ratio is not None:
            return make_mixed_dataset(
                task=self.cfg.task,
                split=split,
                size=size,
                seed=self.cfg.seed,
                tokenizer=self.tokenizer,
                max_seq_len=self.model_cfg.max_seq_len,
                easy_ratio=self.cfg.mixed_easy_ratio,
                easy_complexity=self.cfg.min_complexity,
                hard_complexity=self.cfg.max_complexity,
            )
        return make_dataset(
            self.cfg.task,
            split,
            size,
            self.cfg.seed,
            self.tokenizer,
            self.model_cfg.max_seq_len,
            min_complexity=self.cfg.min_complexity,
            max_complexity=self.cfg.max_complexity,
        )

    def _setup_compilation(self) -> None:
        """Compile the full-sequence forward and verify it against eager.

        The optimizer and checkpoints keep using ``self.model`` so state-dict
        keys and saved checkpoints are unchanged.  Only the training forward
        call goes through the compiled handle.
        """
        probe_len = min(8, self.model_cfg.max_seq_len)
        ids = torch.randint(
            1, self.model_cfg.vocab_size, (1, probe_len), device=self.device
        )
        mask = torch.ones_like(ids)
        was_training = self.model.training
        self.model.eval()

        def probe(module):
            logits, _ = module(
                ids,
                mask,
                mode="fixed_depth",
                max_steps=self.max_steps,
                memory_enabled=False,
            )
            return logits

        try:
            compiled, status = maybe_compile(
                self.model,
                enabled=True,
                mode=self.cfg.compile_mode,
                probe=probe,
            )
        finally:
            self.model.train(was_training)
        self.compilation_status = status
        if status.get("used") == "compiled":
            self.train_model = compiled
        self._emit({"type": "compile", **status})

    def _train_forward(self, ids, mask):
        try:
            return self.train_model(
                ids,
                mask,
                mode=self.cfg.mode,
                max_steps=self.max_steps,
                memory_enabled=False,
            )
        except Exception as exc:
            if self.train_model is self.model:
                raise
            # A compiler can fail on the first real training shape even when
            # the probe passed.  Retry the batch eagerly; no optimizer step has
            # happened yet, so this is a clean fallback.
            self.compilation_status.update(
                {
                    "compiled": False,
                    "used": "eager",
                    "error": f"{type(exc).__name__}: {exc}",
                    "reason": "compiled training step failed; eager fallback",
                }
            )
            self._emit({"type": "compile", **self.compilation_status})
            self.train_model = self.model
            return self.model(
                ids,
                mask,
                mode=self.cfg.mode,
                max_steps=self.max_steps,
                memory_enabled=False,
            )

    def _resume(self, path: str) -> None:
        payload = load_checkpoint(path, map_location="cpu")
        ckpt_cfg = payload.get("model_config", {})
        if ckpt_cfg.get("hidden_dim") != self.model_cfg.hidden_dim:
            raise ValueError("checkpoint model config does not match current model")
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device)
        if payload.get("optimizer_state"):
            try:
                self.optimizer.load_state_dict(payload["optimizer_state"])
            except Exception:
                pass
        self.history = list(payload.get("history", []))
        self.start_epoch = len(self.history)
        self.resume_payload = payload
        self.run_name = payload.get("checkpoint_id", self.run_name) + "-resume"

    def _emit(self, event: dict[str, Any]) -> None:
        if self.progress_cb:
            try:
                self.progress_cb(event)
            except Exception:
                pass

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _stopped(self) -> bool:
        return bool(self.stop_event and self.stop_event.is_set())

    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        best_val = float("inf")
        best_accuracy: float | None = None
        best_epoch = -1
        patience = 0
        start = time.perf_counter()
        global_step = 0
        final_metrics: dict[str, Any] = {}
        last_val: dict[str, Any] = {}

        for epoch in range(self.start_epoch, cfg.epochs):
            if self._stopped():
                break
            self.model.train()
            epoch_loss = 0.0
            epoch_lm = 0.0
            epoch_ponder = 0.0
            epoch_depth = 0.0
            n_batches = 0
            tokens_seen = 0
            epoch_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            ponder_lambda = cfg.ponder_lambda
            if cfg.ponder_warmup_epochs > 0:
                warm = min(1.0, (epoch + 1) / cfg.ponder_warmup_epochs)
                ponder_lambda = cfg.ponder_lambda * warm
            for step, batch in enumerate(self.train_loader):
                if self._stopped():
                    break
                ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                with autocast_context(self.precision_plan):
                    logits, stats = self._train_forward(ids, mask)
                    loss, parts = total_loss(
                        logits, labels, stats, ponder_lambda=ponder_lambda,
                        accuracy_gated=self.cfg.accuracy_gated_ponder,
                    )
                scaled_loss = loss / cfg.grad_accum
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if (step + 1) % cfg.grad_accum == 0:
                    self._optimizer_step()
                    global_step += 1
                n_batches += 1
                tokens_seen += int(mask.sum().item())
                epoch_loss += parts["total_loss"]
                epoch_lm += parts["lm_loss"]
                epoch_ponder += parts["ponder_cost"]
                epoch_depth += float(stats["mean_depth"])
                if step % 10 == 0 or step == len(self.train_loader) - 1:
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.perf_counter() - epoch_start
                    self._emit(
                        {
                            "type": "step",
                            "epoch": epoch + 1,
                            "epochs": cfg.epochs,
                            "step": step + 1,
                            "steps_per_epoch": len(self.train_loader),
                            "global_step": global_step,
                            "loss": round(parts["total_loss"], 6),
                            "lm_loss": round(parts["lm_loss"], 6),
                            "ponder_cost": round(parts["ponder_cost"], 6),
                            "mean_depth": round(float(stats["mean_depth"]), 4),
                            "lr": lr,
                            "tokens_per_second": round(tokens_seen / max(elapsed, 1e-9), 2),
                            "elapsed_seconds": round(time.perf_counter() - start, 2),
                        }
                    )

            if n_batches and n_batches % cfg.grad_accum != 0:
                self._optimizer_step()
                global_step += 1

            train_metrics = {
                "loss": epoch_loss / max(1, n_batches),
                "lm_loss": epoch_lm / max(1, n_batches),
                "ponder_cost": epoch_ponder / max(1, n_batches),
                "mean_depth": epoch_depth / max(1, n_batches),
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            val = evaluate_loss(
                self.model,
                self.val_loader,
                self.device,
                cfg.mode,
                self.max_steps,
                precision_plan=self.precision_plan,
            )
            val_acc = None
            gen = None
            if cfg.task != "lm":
                gen = evaluate_generation(
                    self.model,
                    self.val_ds,
                    self.tokenizer,
                    self.device,
                    mode=self.eval_mode,
                    max_steps=self.max_steps,
                    batch_size=max(16, cfg.batch_size),
                    precision_plan=self.precision_plan,
                )
                val_acc = gen["accuracy"]
            entry = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss": train_metrics["loss"],
                "train_lm_loss": train_metrics["lm_loss"],
                "train_ponder_cost": train_metrics["ponder_cost"],
                "train_mean_depth": train_metrics["mean_depth"],
                "val_loss": val["loss"],
                "val_perplexity": val["perplexity"],
                "val_ponder_cost": val["ponder_cost"],
                "val_mean_depth": val["mean_depth"],
                "val_accuracy": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": train_metrics["epoch_seconds"],
                "elapsed_seconds": time.perf_counter() - start,
            }
            self.history.append(entry)
            last_val = entry
            self._emit({"type": "eval", **entry})
            improved = val["loss"] < best_val - 1e-6
            if improved:
                best_val = val["loss"]
                best_accuracy = val_acc
                best_epoch = epoch + 1
                patience = 0
                self._save("best", entry, final=False)
            else:
                patience += 1
            if cfg.early_stop_patience and patience >= cfg.early_stop_patience:
                self._emit(
                    {"type": "early_stop", "epoch": epoch + 1, "patience": patience}
                )
                break
            if self._stopped():
                break

        final_metrics = {
            "best_val_loss": best_val if math.isfinite(best_val) else None,
            "best_val_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "last": last_val,
            "history": self.history,
            "stopped": self._stopped(),
            "duration_seconds": time.perf_counter() - start,
            "parameters_total": count_parameters(self.model),
            "parameters_trainable": count_parameters(self.model, trainable_only=True),
            "device": str(self.device),
            "precision": self.precision_plan.as_dict(),
            "compilation": dict(self.compilation_status),
            "model_config": self.model_cfg.model_dump(),
            "training_config": self.cfg.model_dump(),
            "datasets": {
                "train": dataset_manifest(self.train_ds),
                "val": dataset_manifest(self.val_ds),
                "test": dataset_manifest(self.test_ds),
            },
            "max_steps_used": self.max_steps,
        }
        meta = self._save("final", final_metrics, final=True)
        final_metrics["checkpoint"] = {
            key: value for key, value in meta.items() if key != "metrics"
        }
        self._emit({"type": "done", **{k: v for k, v in final_metrics.items() if k != "history"}})
        return final_metrics

    def _save(self, suffix: str, metrics: dict[str, Any], final: bool) -> dict[str, Any]:
        cid = f"{self.run_name}-{suffix}"
        meta = save_checkpoint(
            self.checkpoint_dir,
            cid,
            self.model,
            self.model_cfg,
            self.cfg,
            self.optimizer,
            metrics,
            self.history,
            extra={"final": final, "device": str(self.device)},
        )
        self._emit({"type": "checkpoint", "checkpoint_id": cid, "path": meta["path"]})
        return meta

    def evaluate_test(self) -> dict[str, Any]:
        val = evaluate_loss(
            self.model,
            self.val_loader,
            self.device,
            self.cfg.mode,
            self.max_steps,
            precision_plan=self.precision_plan,
        )
        result: dict[str, Any] = {"loss": val, "task": self.cfg.task}
        if self.cfg.task != "lm":
            gen = evaluate_generation(
                self.model,
                self.test_ds,
                self.tokenizer,
                self.device,
                mode=self.eval_mode,
                max_steps=self.max_steps,
                batch_size=max(16, self.cfg.batch_size),
                precision_plan=self.precision_plan,
            )
            result["generation"] = gen
        return result
