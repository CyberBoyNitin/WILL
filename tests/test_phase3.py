"""Phase 3 tests: mixed difficulty, precision, scaling and compilation."""
from __future__ import annotations

import math

import pytest
import torch

from backend.benchmarks.runner import run_benchmark_suite
from backend.config import SCALING_PRESETS, ModelConfig, scaling_model_config
from backend.engine.compilation import maybe_compile
from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer
from backend.training.datasets import make_mixed_dataset
from backend.training.precision import resolve_precision
from backend.training.trainer import Trainer


def test_mixed_dataset_has_exact_70_30_composition(tokenizer):
    ds = make_mixed_dataset(
        "arith_add",
        "test",
        size=100,
        seed=1337,
        tokenizer=tokenizer,
        max_seq_len=64,
        easy_ratio=0.7,
    )
    assert ds.complexity_counts == {1: 70, 2: 30}
    assert len(ds) == 100
    assert sum(1 for item in ds.items if item["complexity"] == 1) == 70
    assert sum(1 for item in ds.items if item["complexity"] == 2) == 30
    # Deterministic for a fixed seed.
    again = make_mixed_dataset(
        "arith_add", "test", 100, 1337, tokenizer, 64, easy_ratio=0.7
    )
    assert [item["prompt"] for item in ds.items] == [
        item["prompt"] for item in again.items
    ]


def test_precision_resolution_is_explicit():
    fp32 = resolve_precision("fp32", torch.device("cpu"))
    assert fp32.enabled is False and fp32.effective == "fp32"
    bf16 = resolve_precision("bf16", torch.device("cpu"))
    assert bf16.enabled is True
    assert bf16.effective == "bf16"
    assert bf16.dtype == torch.bfloat16
    fp16 = resolve_precision("fp16", torch.device("cpu"))
    assert fp16.enabled is True and fp16.effective == "fp16"
    auto = resolve_precision("auto", torch.device("cpu"))
    assert auto.effective == "fp32"
    with pytest.raises(ValueError):
        resolve_precision("int8", torch.device("cpu"))


def test_compile_falls_back_when_torch_compile_raises(monkeypatch, tiny_model):
    def boom(*_args, **_kwargs):
        raise RuntimeError("no C++ compiler")

    monkeypatch.setattr(torch, "compile", boom)
    ids = torch.randint(1, 20, (1, 8))
    model, status = maybe_compile(
        tiny_model,
        enabled=True,
        mode="reduce-overhead",
        probe=lambda module: module(ids, torch.ones_like(ids), mode="fixed_depth")[0],
    )
    assert model is tiny_model
    assert status["used"] == "eager"
    assert status["compiled"] is False
    assert "no C++ compiler" in status["error"]


def test_compile_verifies_numerical_equivalence(monkeypatch, tiny_model):
    # Identity compile stand-in: exercises the probe and comparison path
    # without requiring a local compiler.
    monkeypatch.setattr(torch, "compile", lambda module, mode=None: module)
    ids = torch.randint(1, 20, (1, 8))
    compiled, status = maybe_compile(
        tiny_model,
        enabled=True,
        probe=lambda module: module(ids, torch.ones_like(ids), mode="fixed_depth")[0],
    )
    assert status["used"] == "compiled"
    assert status["checked"] is True
    assert status["max_abs_diff"] == 0.0
    assert compiled is tiny_model


def test_scaling_presets_target_parameter_budgets():
    assert set(SCALING_PRESETS) == {"15m", "50m", "125m"}
    for name, preset in SCALING_PRESETS.items():
        cfg = scaling_model_config(name)
        assert isinstance(cfg, ModelConfig)
        assert cfg.hidden_dim % cfg.n_heads == 0
        assert preset["target_parameters"] > 0


def test_scaling_benchmark_runs_real_steps(temp_app_cfg):
    report = run_benchmark_suite(
        temp_app_cfg,
        "scaling",
        {
            "presets": ["15m"],
            "precisions": ["fp32"],
            "scaling_seq_len": 8,
            "scaling_batch_size": 1,
            "scaling_steps": 1,
            "scaling_warmup": 0,
            "scaling_repeats": 1,
            "device": "cpu",
        },
    )
    assert report["suite"] == "scaling"
    assert len(report["results"]) == 1
    row = report["results"][0]
    assert row["supported"] is True
    assert row["finite"] is True
    assert row["step_seconds"] > 0
    assert row["parameters"] == report["summary"]["presets"]["15m"]["parameters"]
    assert abs(row["parameters"] - 15_000_000) / 15_000_000 < 0.05
    assert math.isfinite(row["loss"])
    assert isinstance(report["summary"]["cuda_available"], bool)


def test_mixed_difficulty_benchmark_reports_strata(temp_app_cfg):
    report = run_benchmark_suite(
        temp_app_cfg,
        "mixed_difficulty",
        {
            "limit": 10,
            "depths": [1],
            "modes": ["adaptive_hard"],
            "batch_size": 5,
            "max_new_tokens": 2,
            "max_examples": 0,
            "generation_repeats": 1,
            "device": "cpu",
            "compile_probe": False,
        },
    )
    assert report["suite"] == "mixed_difficulty"
    assert report["dataset"]["complexity_counts"] == {"1": 7, "2": 3}
    assert report["summary"]["composition_ok"] is True
    assert report["summary"]["strata"] == {"trivial": 7, "hard": 3}
    for row in report["results"]:
        assert set(row["strata"]) == {"trivial", "hard"}
        assert row["strata"]["trivial"]["count"] == 7
        assert row["strata"]["hard"]["count"] == 3
        assert 0.0 <= row["accuracy"] <= 1.0


def test_trainer_honours_bf16_precision(tmp_path, tiny_cfg, tiny_training_cfg):
    cfg = tiny_training_cfg.model_copy(
        update={"precision": "bf16", "epochs": 1, "max_steps": 1}
    )
    model = WILLModel(tiny_cfg)
    trainer = Trainer(
        model=model,
        tokenizer=ByteTokenizer(),
        model_cfg=tiny_cfg,
        training_cfg=cfg,
        checkpoint_dir=tmp_path / "checkpoints",
        device="cpu",
        run_name="pytest-bf16",
    )
    metrics = trainer.train()
    assert metrics["precision"]["effective"] == "bf16"
    assert metrics["precision"]["enabled"] is True
    assert math.isfinite(metrics["best_val_loss"])
    assert metrics["history"][-1]["train_loss"] > 0
