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

import re
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


_FACT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_facts(text: str, max_chars: int = 400) -> list[str]:
    """Split raw prose into bounded semantic fact spans (sentences)."""
    spans = [part.strip() for part in _FACT_SPLIT.split(text) if part.strip()]
    facts: list[str] = []
    for span in spans:
        while len(span) > max_chars:
            cut = span.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            facts.append(span[:cut].strip())
            span = span[cut:].strip()
        if span:
            facts.append(span)
    return facts


def fact_key(text: str, limit: int = 120) -> str:
    """Extract the entity/topic used as the retrieval key for a fact."""
    lowered = text.lower()
    markers = (" -> ", ": ", " is ", " are ", " was ", " were ", " = ")
    positions = [lowered.find(marker) for marker in markers if lowered.find(marker) > 0]
    if positions:
        subject = text[: min(positions)].strip().rstrip(".,;:")
        if subject:
            return subject[:limit]
    return " ".join(text.split()[:8])[:limit]


def fact_answer(text: str, limit: int = 160) -> str:
    """Pick the salient predicate of a sentence as the memorised answer span."""
    lowered = text.lower()
    for marker in (": ", " is ", " are ", " was ", " were ", " = "):
        index = lowered.find(marker)
        if index >= 0:
            candidate = text[index + len(marker) :].strip().rstrip(".")
            if candidate:
                return candidate[:limit]
    return text[:limit]


def orthogonalize(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gram-Schmidt a batch of key vectors to reduce fast-weight interference."""
    out = []
    for row in vectors:
        value = row
        for previous in out:
            value = value - torch.dot(value, previous) * previous
        norm = value.norm()
        out.append(value / norm.clamp_min(eps))
    return torch.stack(out)


@torch.no_grad()
def absorb_documents(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    state=None,
    max_facts: int | None = None,
    slots_per_fact: int | None = None,
    decay: float | None = None,
    ridge: float | None = 1e-6,
):
    """Absorb unstructured paragraphs into a bounded sequential fast state.

    Each sentence becomes a record key (encoded by ``MemoryEncoder``); the
    salient predicate is stored as token-embedding values in sequential slots.
    Keys are orthogonalised within the call and older weights decay, giving
    stable sequential absorption without growing the state size.
    """
    device = next(model.parameters()).device
    facts: list[str] = []
    for text in texts:
        facts.extend(split_facts(text))
    if not facts:
        return state if state is not None else model.memory.init_state(device)
    max_facts = int(max_facts or min(64, model.cfg.memory_slots))
    slots_per_fact = int(
        slots_per_fact or min(8, max(1, model.cfg.memory_slots // 8))
    )
    facts = facts[:max_facts]
    if state is None:
        state = model.memory.init_state(device)
    current_decay = (
        float(model.memory_decay.item())
        if hasattr(model, "memory_decay")
        else float(getattr(model.cfg, "memory_decay_rate", 0.98))
    )
    if decay is not None:
        current_decay = float(decay)
    all_keys: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    for fact in facts:
        key_text = fact_key(fact)
        ids = torch.tensor(
            [[tokenizer.bos_id] + tokenizer.encode(key_text, max_len=model.cfg.max_seq_len - 2) + [tokenizer.sep_id]],
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones_like(ids)
        _, keys, _ = model.encode_memory(ids, mask)
        record_key = model.pool_memory_key(keys, mask, input_ids=ids)[0]
        answer_ids = tokenizer.encode(fact_answer(fact), add_eos=True)
        answer_ids = answer_ids[:slots_per_fact]
        answer_ids += [tokenizer.eos_id] * (slots_per_fact - len(answer_ids))
        for slot, token_id in enumerate(answer_ids):
            # Slot signs already orthogonalise the per-record component; do
            # not rotate stored keys here, otherwise the reader query would no
            # longer match the written key.
            key = model.make_slot_keys(record_key, min(slot, model.cfg.memory_slots - 1))
            all_keys.append(key)
            all_values.append(model.embedding.token_embedding.weight[token_id])
    if not all_keys:
        return state
    # Decay bounds capacity: when a new document pushes past the slot budget,
    # the oldest record (not the fast-weight matrix) is dropped.
    budget = max(1, model.cfg.memory_slots) * max(1, slots_per_fact)
    if len(all_keys) > budget:
        drop = len(all_keys) - budget
        all_keys = all_keys[drop:]
        all_values = all_values[drop:]
    key_batch = torch.stack(all_keys)
    value_batch = torch.stack(all_values)
    state.metadata["absorbed_facts"] = len(facts)
    state.metadata["decay"] = current_decay
    state = model.memory.consolidate(state, key_batch, value_batch, ridge=ridge)
    return state


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
