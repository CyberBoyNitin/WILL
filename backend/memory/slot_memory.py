"""Sequential answer-slot writer for the fast-weight memory.

The v4 memory writer stored one pooled answer vector per record.  A pooled
vector cannot preserve the order of a multi-token answer, so the decoder had
to guess the digits from a single point in representation space.  This writer
stores one key/value pair per answer-token position instead:

* key   = normalized(record key + learned slot address)
* value = the language model's token embedding for that answer token

Retrieval is therefore a direct nearest-neighbour lookup in the same space as
``mem_to_vocab`` / the token embedding table.  The runtime logit skip in
``WILLModel._apply_memory`` turns each retrieved embedding into exact output
token scores at the corresponding decoding step.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from backend.engine.tokenizer import BOS_ID, EOS_ID


def pad_prompts(
    prompts: Sequence[Sequence[int]],
    pad_id: int,
    *,
    left: bool,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Pad prompts and return ``(ids, mask, positions, lengths)``.

    ``positions`` must be passed to the writer as well as generation: a
    left-padded prompt is embedded with positions ``0..n-1``, not with the
    physical column index.  Without this the writer and reader would encode
    different keys for the same question.
    """
    lengths = [max(1, len(p)) for p in prompts]
    width = max(lengths)
    if max_len is not None:
        width = min(width, max_len)
    ids = torch.full((len(prompts), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    positions = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        seq = list(prompt)[:width]
        if not seq:
            seq = [BOS_ID]
        n = len(seq)
        if left:
            ids[row, width - n :] = torch.tensor(seq, dtype=torch.long)
            mask[row, width - n :] = 1
            positions[row, width - n :] = torch.arange(n, dtype=torch.long)
        else:
            ids[row, :n] = torch.tensor(seq, dtype=torch.long)
            mask[row, :n] = 1
            positions[row, :n] = torch.arange(n, dtype=torch.long)
    return ids, mask, positions, lengths


@torch.no_grad()
def write_slot_memory(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answers: Sequence[str],
    tokenizer,
    *,
    positions: torch.Tensor | None = None,
    ridge: float | None = None,
):
    """Consolidate one answer-slot pair per position into a fresh fast state.

    ``input_ids`` and ``attention_mask`` must be the exact padded tensors that
    will be passed to generation.  The pooled record key depends on padding
    width when the batch is left-padded, so writer and reader must share it.
    """
    device = input_ids.device
    if len(answers) != input_ids.shape[0]:
        raise ValueError("answers and prompts must have the same length")
    _, keys, _ = model.encode_memory(
        input_ids, attention_mask, positions=positions
    )
    record_keys = model.pool_memory_key(
        keys, attention_mask, input_ids=input_ids
    )
    slot_keys: list[torch.Tensor] = []
    slot_values: list[torch.Tensor] = []
    for row, answer in enumerate(answers):
        answer_ids = tokenizer.encode(answer, add_eos=True)
        if len(answer_ids) > model.cfg.memory_slots:
            # Keep the final EOS by replacing the last stored position.
            answer_ids = answer_ids[: model.cfg.memory_slots]
            answer_ids[-1] = EOS_ID
        for slot in range(model.cfg.memory_slots):
            token_id = (
                answer_ids[slot] if slot < len(answer_ids) else int(EOS_ID)
            )
            slot_keys.append(
                model.make_slot_keys(record_keys[row], slot)
            )
            slot_values.append(
                model.embedding.token_embedding.weight[token_id]
            )
    state = model.memory.init_state(device)
    return model.memory.consolidate(
        state,
        torch.stack(slot_keys),
        torch.stack(slot_values),
        ridge=ridge,
    )


def answer_slot_targets(
    answers: Iterable[str], tokenizer, slots: int, device: torch.device | str
) -> torch.Tensor:
    """Return ``[N, slots]`` answer-token ids padded with EOS.

    Used by the Phase C auxiliary readout loss.  Slot ``j`` is the j-th answer
    token; later slots are EOS so the language model learns the stop point.
    """
    rows = []
    for answer in answers:
        ids = tokenizer.encode(answer, add_eos=True)
        if len(ids) > slots:
            ids = ids[:slots]
            ids[-1] = EOS_ID
        ids = ids + [EOS_ID] * (slots - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long, device=device)
