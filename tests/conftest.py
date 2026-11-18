"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import ModelConfig, TrainingConfig, load_config
from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer


@pytest.fixture()
def tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


@pytest.fixture()
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        hidden_dim=64,
        n_heads=4,
        context_layers=1,
        context_ff_mult=2,
        max_seq_len=64,
        max_steps=4,
        min_steps=1,
        dropout=0.0,
        ff_mult=2,
        memory_dim=32,
    )


@pytest.fixture()
def tiny_model(tiny_cfg: ModelConfig) -> WILLModel:
    torch.manual_seed(7)
    model = WILLModel(tiny_cfg)
    model.eval()
    return model


@pytest.fixture()
def tiny_training_cfg() -> TrainingConfig:
    return TrainingConfig(
        task="arith_add",
        train_size=32,
        val_size=16,
        test_size=16,
        epochs=3,
        batch_size=8,
        lr=1e-3,
        warmup_steps=2,
        mode="fixed_depth",
        max_steps=2,
        min_complexity=1,
        max_complexity=1,
        seed=7,
    )


@pytest.fixture()
def temp_app_cfg(tmp_path: Path):
    cfg = load_config()
    cfg.model = ModelConfig(
        hidden_dim=64,
        n_heads=4,
        context_layers=1,
        context_ff_mult=2,
        max_seq_len=64,
        max_steps=4,
        min_steps=1,
        dropout=0.0,
        ff_mult=2,
        memory_dim=32,
    )
    cfg.training = TrainingConfig(
        task="arith_add",
        train_size=16,
        val_size=8,
        test_size=8,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        warmup_steps=1,
        mode="fixed_depth",
        max_steps=2,
        min_complexity=1,
        max_complexity=1,
        seed=7,
    )
    root = tmp_path / "will"
    cfg.server.data_dir = str(root / "data")
    cfg.server.checkpoint_dir = str(root / "checkpoints")
    cfg.server.experiment_dir = str(root / "experiments")
    cfg.server.upload_dir = str(root / "data" / "uploads")
    cfg.server.memory_dir = str(root / "data" / "memory_sessions")
    cfg.server.db_path = str(root / "data" / "will.db")
    cfg.ensure_dirs()
    return cfg


@pytest.fixture()
def client(temp_app_cfg):
    from fastapi.testclient import TestClient

    from backend.server.app import create_app

    app = create_app(temp_app_cfg)
    with TestClient(app) as test_client:
        yield test_client
