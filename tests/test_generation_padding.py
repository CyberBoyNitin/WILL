"""Regression test for batched generation with unequal prompt lengths.

The original implementation read ``logits[:, -1]`` from a right-padded batch.
Only the longest prompt had a real token in that column; shorter prompts were
evaluated at a PAD position.  This test builds a model whose next-token logits
are only correct at each row's true last token.
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

    def fake_forward(input_ids, attention_mask=None, **_kwargs):
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
        return logits, {
            "mean_depth": 1.0,
            "total_token_evaluations": int(b * t),
            "estimated_macs": 0,
            "ponder_cost": torch.tensor(0.0),
        }

    tiny_model.forward = fake_forward  # type: ignore[assignment]
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
