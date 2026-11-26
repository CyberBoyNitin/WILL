"""Right-to-left column-scratchpad decoder for two-digit addition.

The model emits the two column sums from right to left, then reads the answer
off those sums.  No explicit ``digit = sum mod 10`` stage is needed: mod-10
extraction is a copy of the final character of the emitted ones-column string,
a primitive the model already masters.

    calc 47 + 38 =                 -> 15      ones column: 7 + 8
    calc 47 + 38 = ones 15         -> 8       tens column: 4 + 3 + carry(15)
    calc 47 + 38 = ones 15 tens 8  -> 85      concatenate tens sum + last ones digit

For any two-digit operands, ``answer = str(tens_sum) + str(ones_sum)[-1]``:
e.g. ones 10 / tens 19 -> 190, ones 9 / tens 10 -> 109.  The carry is therefore
resolved inside the right-to-left column chain, not guessed before it.

Only model-generated sums are fed to the next stage; Python does not compute
carries at inference.
"""
from __future__ import annotations

import re
from typing import Any

import torch

from backend.engine.tokenizer import BOS_ID, SEP_ID


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"-?\d+", text)]


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
    """Generate final answers with a three-stage right-to-left column trace."""
    if not prompts:
        return {"answers": [], "fields": {}, "stages": []}
    limit = max_seq_len or int(model.cfg.max_seq_len)
    matches = [re.search(r"(\d+)\s*\+\s*(\d+)", p) for p in prompts]
    if any(match is None for match in matches):
        raise ValueError("every prompt must be a 'calc A + B =' question")
    bases = [
        f"calc {match.group(1)} + {match.group(2)} ="  # type: ignore[union-attr]
        for match in matches
    ]

    def generate(texts: list[str], new_tokens: int) -> tuple[list[str], dict]:
        out = model.generate_batch(
            [prompt_ids(tokenizer, text, limit) for text in texts],
            tokenizer,
            mode="fixed_depth",
            max_steps=depth,
            max_new_tokens=new_tokens,
            stop_at_eos=True,
        )
        return [
            tokenizer.decode(ids, skip_special=True)
            for ids in out["generated_ids"]
        ], out

    stage_texts: list[list[str]] = []
    depth_sum = 0.0

    # Rightmost column first: raw ones sum.
    texts, out = generate(bases, 4)
    stage_texts.append(texts)
    depth_sum += float(out["mean_depth"])
    s1 = [(parse_ints(text) or [-1])[0] for text in texts]

    # Next column left: raw tens sum, including the carry from the ones column.
    texts, out = generate(
        [f"{bases[i]} ones {s1[i]}" for i in range(len(bases))], 4
    )
    stage_texts.append(texts)
    depth_sum += float(out["mean_depth"])
    s2 = [(parse_ints(text) or [-1])[0] for text in texts]

    # Read the answer off the emitted columns.
    texts, out = generate(
        [f"{bases[i]} ones {s1[i]} tens {s2[i]}" for i in range(len(bases))],
        6,
    )
    stage_texts.append(texts)
    depth_sum += float(out["mean_depth"])
    answers = [(parse_ints(text) or [""])[0] for text in texts]

    return {
        "answers": [str(value) for value in answers],
        "fields": {"s1": s1, "s2": s2},
        "stages": stage_texts,
        "mean_depth": depth_sum / 3.0,
        "bases": bases,
    }
