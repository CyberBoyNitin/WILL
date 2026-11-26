"""Public SDK surface used in the README quickstart."""
from __future__ import annotations

import pytest


def test_public_sdk_aliases():
    from will_engine import WILLConfig, WILLForCausalLM, WILLModel

    assert WILLModel.__name__ == "WILLModel"
    assert WILLConfig.__name__ == "ModelConfig"
    assert WILLForCausalLM.__name__ == "WILLForCausalLM"


def test_will_config_builds_slot_model():
    from will_engine import WILLConfig, WILLModel

    cfg = WILLConfig(
        hidden_dim=32,
        n_heads=4,
        context_layers=1,
        max_seq_len=32,
        memory_dim=32,
        memory_slots=4,
        memory_logit_skip=True,
        memory_logit_scale=20.0,
    )
    model = WILLModel(cfg)
    assert model.cfg.memory_slots == 4
    assert model.mem_to_vocab.weight is model.embedding.token_embedding.weight
    assert pytest.approx(float(model.memory_logit_alpha.item())) == 20.0
