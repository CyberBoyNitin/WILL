"""Fast-weight memory, isolation and persistence tests."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from backend.config import MemoryConfig
from backend.memory.fast_memory import (
    AssociativeMemory,
    FastMemoryState,
    FastWeightMemory,
    TTTLinear,
)
from backend.memory.memory_manager import MemoryManager


def test_ttt_linear_fits_associative_recall():
    torch.manual_seed(3)
    d = 16
    ttt = TTTLinear(d, d, lr=0.05, steps=4, inner_optimizer="adam")
    state = ttt.init_state()
    keys = F.normalize(torch.randn(12, d), dim=-1)
    target = torch.randn(d, d)
    values = F.normalize(keys @ target.T, dim=-1)
    before = F.mse_loss(ttt.read(state, keys), values).item()
    for _ in range(20):
        ttt.adapt(state, keys, values)
    after = F.mse_loss(ttt.read(state, keys), values).item()
    assert after < before * 0.8, (before, after)
    assert state.updates == 20
    assert state.last_loss is not None


def test_fast_states_are_independent():
    torch.manual_seed(4)
    d = 12
    ttt = TTTLinear(d, d, lr=0.05, steps=2)
    a = ttt.init_state()
    b = ttt.init_state()
    original_b = {k: v.clone() for k, v in b.params.items()}
    keys = F.normalize(torch.randn(8, d), dim=-1)
    values = F.normalize(torch.randn(8, d), dim=-1)
    ttt.adapt(a, keys, values)
    assert a.updates == 1
    assert b.updates == 0
    for key in original_b:
        assert torch.allclose(b.params[key], original_b[key])
    assert not torch.allclose(a.params["weight"], b.params["weight"])


def test_fast_state_save_restore(tmp_path):
    d = 12
    ttt = TTTLinear(d, d, lr=0.05, steps=2)
    state = ttt.init_state()
    keys = F.normalize(torch.randn(6, d), dim=-1)
    values = F.normalize(torch.randn(6, d), dim=-1)
    ttt.adapt(state, keys, values)
    path = tmp_path / "state.pt"
    state.save(str(path))
    loaded = FastMemoryState.load(str(path))
    assert loaded.kind == state.kind
    assert loaded.updates == state.updates
    for key in state.params:
        assert torch.allclose(loaded.params[key], state.params[key])


def test_adapting_memory_does_not_mutate_slow_weights(tiny_model):
    before = {name: p.detach().clone() for name, p in tiny_model.named_parameters()}
    state = tiny_model.memory.init_state()
    keys = F.normalize(torch.randn(6, tiny_model.cfg.memory_dim), dim=-1)
    values = F.normalize(torch.randn(6, tiny_model.cfg.memory_dim), dim=-1)
    tiny_model.memory.adapt(state, keys, values)
    assert state.updates == 1
    for name, param in tiny_model.named_parameters():
        assert torch.allclose(param, before[name]), name


def test_memory_manager_session_isolation_and_persistence(tmp_path, tiny_model):
    cfg = MemoryConfig(max_chunks=50)
    manager = MemoryManager(
        FastWeightMemory(tiny_model.cfg), cfg, tmp_path / "sessions", device="cpu"
    )
    s1 = manager.create_session("one")
    s2 = manager.create_session("two")
    keys = F.normalize(torch.randn(5, tiny_model.cfg.memory_dim), dim=-1)
    values = F.normalize(torch.randn(5, tiny_model.cfg.memory_dim), dim=-1)
    result = manager.adapt_pairs(s1.session_id, keys, values, persist=True)
    assert result["updates"] == 1
    assert manager.get_session(s1.session_id).state.updates == 1
    assert manager.get_session(s2.session_id).state.updates == 0

    # Restore in a fresh manager instance from the on-disk state.
    manager2 = MemoryManager(
        FastWeightMemory(tiny_model.cfg), cfg, tmp_path / "sessions", device="cpu"
    )
    restored = manager2.get_session(s1.session_id)
    assert restored.state.updates == 1
    assert restored.name == "one"
    summaries = manager2.list_sessions()
    assert {s["session_id"] for s in summaries} == {s1.session_id, s2.session_id}

    manager2.delete_session(s2.session_id)
    assert all(s["session_id"] != s2.session_id for s in manager2.list_sessions())


def test_memory_encoder_shapes(tiny_model, tokenizer):
    ids = torch.randint(4, 30, (2, 9))
    ids[:, 0] = tokenizer.bos_id
    mask = torch.ones_like(ids)
    context, keys, values = tiny_model.encode_memory(ids, mask)
    assert context.shape == (2, 9, tiny_model.cfg.hidden_dim)
    assert keys.shape == (2, 9, tiny_model.cfg.memory_dim)
    assert values.shape == (2, 9, tiny_model.cfg.memory_dim)
    assert torch.isfinite(keys).all()
    assert torch.allclose(keys.norm(dim=-1), torch.ones(2, 9), atol=1e-5)


def test_associative_memory_delta_rule():
    torch.manual_seed(9)
    d = 10
    memory = AssociativeMemory(d, lr=0.5)
    state = memory.init_state()
    keys = F.normalize(torch.randn(4, d), dim=-1)
    values = F.normalize(torch.randn(4, d), dim=-1)
    before = F.mse_loss(memory.read(state, keys), values).item()
    for _ in range(5):
        memory.adapt(state, keys, values)
    after = F.mse_loss(memory.read(state, keys), values).item()
    assert after < before
    assert state.kind == "associative"
