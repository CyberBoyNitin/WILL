"""Regression tests for sequential answer-slot fast memory."""
from __future__ import annotations

import torch

from backend.config import ModelConfig
from backend.engine.model import WILLModel
from backend.engine.tokenizer import BOS_ID, SEP_ID, ByteTokenizer
from backend.memory.slot_memory import pad_prompts, write_slot_memory


def slot_model() -> WILLModel:
    cfg = ModelConfig(
        hidden_dim=64,
        n_heads=4,
        context_layers=1,
        max_seq_len=64,
        min_steps=1,
        max_steps=2,
        memory_dim=64,
        memory_slots=4,
        memory_logit_skip=True,
        memory_logit_scale=20.0,
        memory_lexical_scale=1.0,
    )
    torch.manual_seed(7)
    return WILLModel(cfg).eval()


def prompts_and_answers(tokenizer: ByteTokenizer):
    pairs = [(12, 34), (7, 68), (55, 46)]
    prompts = [
        [BOS_ID] + tokenizer.encode(f"calc {a} + {b} =") + [SEP_ID]
        for a, b in pairs
    ]
    answers = [str(a + b) for a, b in pairs]
    return prompts, answers


def test_slot_memory_retrieves_answer_tokens_exactly():
    model = slot_model()
    tokenizer = ByteTokenizer()
    prompts, answers = prompts_and_answers(tokenizer)
    ids, mask, positions, _ = pad_prompts(
        prompts, tokenizer.pad_id, left=True
    )
    with torch.no_grad():
        state = write_slot_memory(
            model,
            ids,
            mask,
            answers,
            tokenizer,
            positions=positions,
            ridge=1e-6,
        )
        _, stats, _ = model.prefill(
            ids,
            attention_mask=mask,
            memory_state=state,
            positions=positions,
            mode="fixed_depth",
            max_steps=1,
        )
        slot_ids = stats["memory_slot_logits"].argmax(dim=-1).tolist()
        generated = model.generate_batch(
            prompts,
            tokenizer,
            memory_state=state,
            mode="fixed_depth",
            max_steps=1,
            max_new_tokens=5,
        )
    for row, answer in enumerate(answers):
        expected = tokenizer.encode(answer, add_eos=True)
        assert slot_ids[row][: len(expected)] == expected
    assert stats["memory"]["slot_confidence_min"] > 0.85
    assert [
        tokenizer.decode(ids, skip_special=True)
        for ids in generated["generated_ids"]
    ] == answers


def test_memory_to_vocab_tied_when_widths_match():
    model = slot_model()
    assert model.mem_to_vocab.weight is model.embedding.token_embedding.weight


def test_legacy_config_has_no_slot_logits():
    cfg = ModelConfig(
        hidden_dim=32,
        n_heads=4,
        context_layers=1,
        max_seq_len=32,
        min_steps=1,
        max_steps=1,
        memory_dim=16,
        memory_logit_skip=False,
    )
    model = WILLModel(cfg).eval()
    tokenizer = ByteTokenizer()
    ids = torch.tensor([[BOS_ID] + tokenizer.encode("1 + 2") + [SEP_ID]])
    with torch.no_grad():
        _, stats, _ = model.prefill(
            ids, mode="fixed_depth", max_steps=1
        )
    assert "memory_slot_logits" not in stats
    assert float(model.memory_logit_alpha.item()) == 0.0
