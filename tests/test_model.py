"""Model, ACT and hard-skipping tests."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from backend.engine.adaptive_controller import serializable_stats
from backend.engine.model import WILLModel


def _batch(tokenizer, size=3, length=12):
    ids = torch.randint(4, 40, (size, length))
    ids[:, 0] = tokenizer.bos_id
    mask = torch.ones_like(ids)
    return ids, mask


def test_fixed_depth_shapes_and_compute(tiny_model, tokenizer):
    ids, mask = _batch(tokenizer)
    logits, stats = tiny_model(ids, mask, mode="fixed_depth", max_steps=3)
    assert logits.shape == (3, 12, 260)
    assert stats["total_token_evaluations"] == 3 * 12 * 3
    assert stats["mean_depth"] == 3.0
    assert stats["estimated_macs"] > 0


def test_soft_act_gradients_flow_to_halting_gate(tiny_model, tokenizer):
    tiny_model.train()
    ids, mask = _batch(tokenizer)
    logits, stats = tiny_model(ids, mask, mode="adaptive_soft", max_steps=4)
    loss = F.cross_entropy(logits.reshape(-1, 260), torch.randint(4, 40, (36,)))
    loss.backward()
    assert tiny_model.halting_gate.net[-1].weight.grad is not None
    assert tiny_model.halting_gate.net[-1].weight.grad.abs().sum().item() > 0
    assert tiny_model.recurrent_block.ff_in.weight.grad is not None
    assert stats["total_token_evaluations"] == 3 * 12 * 4


def test_hard_mode_actually_skips_positions(tiny_model, tokenizer, monkeypatch):
    # Force an almost-certain halt at the first step.
    with torch.no_grad():
        tiny_model.halting_gate.net[-1].bias.fill_(12.0)
    ids, mask = _batch(tokenizer, size=2, length=10)
    seen_rows = []
    original = tiny_model.recurrent_block.forward

    def wrapped(hidden, context, step_emb):
        seen_rows.append(
            hidden.shape[0] * hidden.shape[1] if hidden.ndim == 3 else hidden.shape[0]
        )
        return original(hidden, context, step_emb)

    monkeypatch.setattr(tiny_model.recurrent_block, "forward", wrapped)
    logits, stats = tiny_model(ids, mask, mode="adaptive_hard", max_steps=4)
    assert logits.shape == (2, 10, 260)
    assert stats["total_token_evaluations"] == 2 * 10
    assert sum(seen_rows) == 2 * 10
    assert stats["mean_depth"] == 1.0
    assert stats["mode"] == "adaptive_hard"


def test_soft_mode_runs_all_positions(tiny_model, tokenizer, monkeypatch):
    ids, mask = _batch(tokenizer, size=2, length=10)
    seen_rows = []
    original = tiny_model.recurrent_block.forward

    def wrapped(hidden, context, step_emb):
        seen_rows.append(hidden.shape[0] * hidden.shape[1])
        return original(hidden, context, step_emb)

    monkeypatch.setattr(tiny_model.recurrent_block, "forward", wrapped)
    _logits, stats = tiny_model(ids, mask, mode="adaptive_soft", max_steps=4)
    assert sum(seen_rows) == 2 * 10 * 4
    assert stats["total_token_evaluations"] == 2 * 10 * 4
    assert stats["mode"] == "adaptive_soft"


def test_memory_disabled_matches_plain_forward(tiny_model, tokenizer):
    ids, mask = _batch(tokenizer)
    with torch.no_grad():
        a, sa = tiny_model(ids, mask, mode="fixed_depth", max_steps=2, memory_enabled=False)
        b, sb = tiny_model(ids, mask, mode="fixed_depth", max_steps=2, memory_enabled=False)
    assert torch.allclose(a, b)
    assert sa["memory"]["enabled"] is False
    assert sb["memory"]["enabled"] is False


def test_stats_are_json_serializable(tiny_model, tokenizer):
    ids, mask = _batch(tokenizer)
    _logits, stats = tiny_model(ids, mask, mode="adaptive_soft", max_steps=3)
    payload = serializable_stats(stats)
    text = json.dumps(payload)
    assert "token_depth" in text
    assert "estimated_macs" in payload


def test_forward_rejects_bad_mode(tiny_model, tokenizer):
    ids, mask = _batch(tokenizer)
    try:
        tiny_model(ids, mask, mode="nope")
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("bad mode should raise")
