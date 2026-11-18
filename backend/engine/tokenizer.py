"""Deterministic byte-level tokenizer.

No pretrained tokenizer downloads are required.  Every byte value maps to one
token, with four reserved control tokens at the start of the vocabulary:

    0 PAD | 1 BOS | 2 EOS | 3 SEP | 4..259 byte values

This makes the vocabulary exactly reproducible across machines and means the
model can represent any UTF-8 document.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260

SPECIAL_TOKENS = {
    "<pad>": PAD_ID,
    "<bos>": BOS_ID,
    "<eos>": EOS_ID,
    "<sep>": SEP_ID,
}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}


class ByteTokenizer:
    def __init__(self) -> None:
        self.vocab_size = VOCAB_SIZE
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.sep_id = SEP_ID

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> list[int]:
        ids = [b + BYTE_OFFSET for b in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        data = bytearray()
        for i in ids:
            i = int(i)
            if i < BYTE_OFFSET:
                if not skip_special:
                    data.extend(ID_TO_SPECIAL.get(i, "<?>").encode())
                continue
            data.append(i - BYTE_OFFSET)
        return data.decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str:
        """Human-readable label for a single token (used by diagnostics)."""
        if token_id < BYTE_OFFSET:
            return ID_TO_SPECIAL.get(token_id, f"<{token_id}>")
        b = bytes([token_id - BYTE_OFFSET])
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            return f"<0x{token_id - BYTE_OFFSET:02x}>"
        return {" ": "␠", "\n": "↵", "\t": "⇥"}.get(s, s)

    def token_bytes(self, token_id: int) -> bytes | None:
        if token_id < BYTE_OFFSET:
            return None
        return bytes([token_id - BYTE_OFFSET])

    # ------------------------------------------------------------------
    # Batched helpers
    # ------------------------------------------------------------------
    def encode_batch(
        self,
        texts: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> list[list[int]]:
        return [self.encode(t, add_bos, add_eos, max_len) for t in texts]

    def pad_batch(
        self,
        sequences: Sequence[Sequence[int]],
        pad_to: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("empty batch")
        max_len = max(len(s) for s in sequences)
        if pad_to is not None:
            max_len = max(max_len, pad_to)
        ids = torch.full((len(sequences), max_len), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            if len(seq) > max_len:
                seq = seq[:max_len]
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
        if device is not None:
            ids = ids.to(device)
            mask = mask.to(device)
        return ids, mask

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "type": "byte",
                    "vocab_size": self.vocab_size,
                    "specials": SPECIAL_TOKENS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_tokenizer(path: str | Path | None = None) -> ByteTokenizer:
    if path is not None and Path(path).exists():
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        if meta.get("type") != "byte":
            raise ValueError("unsupported tokenizer type")
    return ByteTokenizer()
