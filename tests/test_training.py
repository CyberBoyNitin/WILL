"""Training loop, checkpoint and evaluation tests."""
from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer
from backend.training.checkpoints import load_model_from_checkpoint
from backend.training.datasets import collate_lm, make_dataset
from backend.training.trainer import Trainer, evaluate_loss


def test_tiny_training_run_writes_loadable_checkpoint(tmp_path, tiny_cfg, tiny_training_cfg):
    torch.manual_seed(11)
    tokenizer = ByteTokenizer()
    model = WILLModel(tiny_cfg)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        model_cfg=tiny_cfg,
        training_cfg=tiny_training_cfg,
        checkpoint_dir=tmp_path / "checkpoints",
        device="cpu",
        run_name="pytest-run",
    )
    metrics = trainer.train()
    assert len(metrics["history"]) == tiny_training_cfg.epochs
    assert metrics["history"][-1]["train_loss"] < metrics["history"][0]["train_loss"]
    assert metrics["best_val_loss"] < float("inf")
    assert metrics["parameters_total"] > 0

    checkpoint_id = metrics["checkpoint"]["checkpoint_id"]
    path = tmp_path / "checkpoints" / f"{checkpoint_id}.pt"
    assert path.exists()
    assert path.with_suffix(".json").exists()
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["checkpoint_id"] == checkpoint_id
    assert meta["metrics"]["best_val_loss"] == metrics["best_val_loss"]

    restored, restored_cfg, payload = load_model_from_checkpoint(path, device="cpu")
    assert restored_cfg.hidden_dim == tiny_cfg.hidden_dim
    assert payload["checkpoint_id"] == checkpoint_id
    assert sum(p.numel() for p in restored.parameters()) == metrics["parameters_total"]


def test_evaluate_loss_reports_compute(tiny_model, tiny_cfg, tokenizer):
    ds = make_dataset(
        task="arith_add",
        split="val",
        size=12,
        seed=3,
        tokenizer=tokenizer,
        max_seq_len=tiny_cfg.max_seq_len,
        min_complexity=1,
        max_complexity=1,
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_lm)
    result = evaluate_loss(tiny_model, loader, torch.device("cpu"), "fixed_depth", 2)
    assert result["loss"] > 0
    assert result["sequences"] == 12
    assert result["token_evaluations"] > 0
    assert result["estimated_macs"] > 0
    assert result["mean_depth"] == 2.0


def test_early_stopping_and_cancellation(
    tmp_path, tiny_cfg, tiny_training_cfg, monkeypatch
):
    import threading

    tiny_training_cfg.early_stop_patience = 1
    tiny_training_cfg.epochs = 5
    calls = {"n": 0}

    def fake_eval(*_args, **_kwargs):
        calls["n"] += 1
        return {
            "loss": 0.5 + 0.1 * calls["n"],
            "perplexity": 2.0,
            "ponder_cost": 0.0,
            "mean_depth": 2.0,
            "tokens": 1,
            "sequences": 1,
            "token_evaluations": 1,
            "estimated_macs": 1,
        }

    monkeypatch.setattr("backend.training.trainer.evaluate_loss", fake_eval)
    model = WILLModel(tiny_cfg)
    trainer = Trainer(
        model=model,
        tokenizer=ByteTokenizer(),
        model_cfg=tiny_cfg,
        training_cfg=tiny_training_cfg,
        checkpoint_dir=tmp_path / "checkpoints",
        device="cpu",
        run_name="pytest-early",
    )
    metrics = trainer.train()
    assert len(metrics["history"]) == 2
    assert metrics["stopped"] is False

    stop = threading.Event()
    stop.set()
    model2 = WILLModel(tiny_cfg)
    trainer2 = Trainer(
        model=model2,
        tokenizer=ByteTokenizer(),
        model_cfg=tiny_cfg,
        training_cfg=tiny_training_cfg,
        checkpoint_dir=tmp_path / "checkpoints2",
        device="cpu",
        stop_event=stop,
        run_name="pytest-stop",
    )
    metrics2 = trainer2.train()
    assert metrics2["history"] == []
