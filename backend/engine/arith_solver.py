"""Chained carry-scratchpad decoder for two-digit addition.

The answer is produced one carry field at a time.  Each stage is a very short
generation whose first token is the supervised value, so the training loss and
the decoding decision are the same quantity (unlike a flat 11-token scratchpad
where teacher forcing hides first-token errors).

    calc 12 + 34 =              -> 6        (ones digit)
    calc 12 + 34 = 6            -> 0        (ones carry)
    calc 12 + 34 = 6,0          -> 4        (tens digit)
    calc 12 + 34 = 6,0,4        -> 1        (tens carry)
    calc 12 + 34 = 6,0,4,1      -> 46       (final answer)

Only model-generated fields are passed to the next stage; no carry is computed
by Python at inference.
"""
from __future__ import annotations

import re
from typing import Any

import torch

from backend.engine.tokenizer import BOS_ID, SEP_ID


def parse_first_int(text: str) -> int | None:
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[0]) if numbers else None


def prompt_ids(tokenizer, text: str, max_len: int) -> list[int]:
    return [BOS_ID] + tokenizer.encode(text, max_len=max_len) + [SEP_ID]


@torch.no_grad()
def solve_hard_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    depth: int = 2,
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    """Generate final answers with a five-stage carry scratchpad."""
    if not prompts:
        return {"answers": [], "fields": [], "stages": []}
    limit = max_seq_len or int(model.cfg.max_seq_len)
    matches = [re.search(r"(\d+)\s*\+\s*(\d+)", p) for p in prompts]
    if any(match is None for match in matches):
        raise ValueError("every prompt must be a 'calc A + B =' question")
    bases = [
        f"calc {match.group(1)} + {match.group(2)} ="  # type: ignore[union-attr]
        for match in matches
    ]

    fields: list[list[int]] = [[], [], [], []]
    stage_texts: list[list[str]] = []
    depth_sum = 0.0
    prompts_now = bases
    for stage in range(5):
        out = model.generate_batch(
            [prompt_ids(tokenizer, text, limit) for text in prompts_now],
            tokenizer,
            mode="fixed_depth",
            max_steps=depth,
            max_new_tokens=4 if stage < 4 else 6,
            stop_at_eos=True,
        )
        texts = [
            tokenizer.decode(ids, skip_special=True) for ids in out["generated_ids"]
        ]
        stage_texts.append(texts)
        depth_sum += float(out["mean_depth"])
        parsed = [parse_first_int(text) for text in texts]
        if stage < 4:
            for row in range(len(prompts)):
                fields[stage].append(
                    parsed[row] if parsed[row] is not None else -1
                )
            prompts_now = [
                f"{bases[i]} "
                + ",".join(str(fields[s][i]) for s in range(stage + 1))
                for i in range(len(prompts))
            ]
        else:
            answers = [str(value) if value is not None else "" for value in parsed]

    return {
        "answers": answers,
        "fields": fields,
        "stages": stage_texts,
        "mean_depth": depth_sum / 5.0,
        "bases": bases,
    }
