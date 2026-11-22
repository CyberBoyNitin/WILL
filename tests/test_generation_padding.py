"""Regression test for batched generation with unequal prompt lengths.

The original implementation read ``logits[:, -1]`` from a right-padded batch.
Only the longest prompt had a real token in that column; shorter prompts were
evaluated at a PAD position.  This test stubs the cache path so the next-token
logits are only correct at each row's true last token, then checks that
``evaluate_generation`` still decodes every row correctly.
"""
from __future__ import annotations

import torch

from backend.training.trainer import evaluate_generation


class _ListDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _FakeCache:
    def select_rows(self, keep):
        return self

    def memory_bytes(self) -> int:
        return 0


def test_batched_generation_uses_each_rows_last_token(tiny_model, tokenizer):
    seven = tokenizer.encode("7")[0]
    eos = tokenizer.eos_id
    items = []
    for n in (4, 7, 11, 5, 9, 13):
        prompt_ids = [tokenizer.bos_id] + [tokenizer.encode("a")[0]] * n + [tokenizer.sep_id]
        items.append(
            {
                "prompt_ids": prompt_ids,
                "answer_text": "7",
                "prompt": "a" * n,
                "complexity": 1,
            }
        )

    def fake_stats(b: int, t: int) -> dict:
        return {
            "mean_depth": 1.0,
            "token_depth": torch.ones((b, t)),
            "total_token_evaluations": int(b * t),
            "estimated_macs": 0,
            "ponder_cost": torch.tensor(0.0),
        }

    def fake_prefill(
        input_ids,
        attention_mask=None,
        positions=None,
        **_kwargs,
    ):
        b, t = input_ids.shape
        lengths = attention_mask.sum(dim=1).long() - 1
        rows = torch.arange(b)
        last_tokens = input_ids[rows, lengths]
        next_tokens = torch.where(
            last_tokens == seven,
            torch.full_like(last_tokens, eos),
            torch.full_like(last_tokens, seven),
        )
        logits = torch.full((b, t, 260), -20.0)
        logits[rows, lengths, next_tokens] = 20.0
        return logits, fake_stats(b, t), _FakeCache()

    def fake_decode_step(input_ids, attention_mask, cache, **_kwargs):
        b, n = input_ids.shape
        rows = torch.arange(b)
        last_tokens = input_ids[:, -1]
        next_tokens = torch.where(
            last_tokens == seven,
            torch.full_like(last_tokens, eos),
            torch.full_like(last_tokens, seven),
        )
        logits = torch.full((b, n, 260), -20.0)
        logits[rows, n - 1, next_tokens] = 20.0
        return logits, fake_stats(b, n), cache

    tiny_model.prefill = fake_prefill  # type: ignore[assignment]
    tiny_model.decode_step = fake_decode_step  # type: ignore[assignment]
    result = evaluate_generation(
        tiny_model,
        _ListDataset(items),
        tokenizer,
        torch.device("cpu"),
        mode="fixed_depth",
        max_steps=1,
        batch_size=6,
        max_new_tokens=4,
    )
    assert result["accuracy"] == 1.0, result["examples"]
    assert result["correct"] == len(items)
    assert result["count"] == len(items)
    assert all(ex["prediction"] == "7" for ex in result["examples"])
