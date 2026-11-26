"""Lightweight v4 compatibility checks (no GPU or network required)."""
from __future__ import annotations

import torch

from backend.config import ModelConfig
from backend.engine.device import amp_dtype, resolve_device
from backend.engine.model import WILLModel
from backend.engine.tokenizer import build_tokenizer
from backend.memory.slot_memory import absorb_documents


def tiny_v4_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=260,
        hidden_dim=64,
        n_heads=4,
        n_kv_heads=2,
        context_layers=2,
        context_ff_mult=2,
        max_seq_len=64,
        tokenizer_type="bpe",
        bpe_name="gpt2",
        use_rope=True,
        memory_dim=64,
        memory_token_dim=32,
        memory_slots=4,
        num_slots=4,
        memory_kind="ttt_linear",
        memory_logit_skip=True,
        memory_halt_trigger=True,
    )


def test_gqa_rope_forward_and_bpe_fallback():
    cfg = tiny_v4_config()
    tok = build_tokenizer(cfg)
    model = WILLModel(cfg).eval()
    assert cfg.num_slots == 4 and cfg.slot_dim == 64
    assert tok.type == "bpe"
    ids = torch.tensor([tok.encode("hello world", add_bos=True)])
    with torch.no_grad():
        logits, stats = model(ids, mode="fixed_depth", max_steps=2)
    assert logits.shape == (1, ids.shape[1], cfg.vocab_size)
    assert stats["mean_depth"] == 2.0
    assert model.context_encoder.layers[0].attn.n_kv_heads == 2


def test_unstructured_absorption_is_bounded():
    cfg = tiny_v4_config()
    tok = build_tokenizer(cfg)
    model = WILLModel(cfg).eval()
    state = absorb_documents(
        model,
        tok,
        ["Ada -> the answer is seven. Ben -> the answer is nine."],
        max_facts=4,
        slots_per_fact=2,
    )
    expected = cfg.memory_dim * (cfg.memory_dim + 1) * 4
    assert state.state_bytes() == expected
    assert int(state.metadata["absorbed_facts"]) == 2


def test_device_fallback():
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")
    if device.type == "cpu":
        assert amp_dtype(device) is None
    else:
        assert amp_dtype(device) in (torch.float16, torch.bfloat16)
