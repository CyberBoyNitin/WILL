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
    def __init__(self, vocab_size: int | None = None) -> None:
        # ``vocab_size`` may be enlarged by a v4 config while the byte alphabet
        # stays at 260 ids; unused embedding rows are harmless and keep every
        # legacy checkpoint/test byte-identical when the argument is omitted.
        self.vocab_size = int(vocab_size) if vocab_size else VOCAB_SIZE
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.sep_id = SEP_ID
        self.type = "char"
        self.name = "byte"

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


class BPETokenizer:
    """Subword BPE tokenizer with reserved WILL control ids 0..3.

    ``tiktoken`` is used when installed and its encoding is cached locally.  If
    it is unavailable or offline, the instance cleanly falls back to the byte
    tokenizer while still reporting the configured vocabulary size, so a v4
    model can be loaded and exercised without network access.
    """

    def __init__(
        self,
        name: str = "cl100k_base",
        vocab_size: int | None = None,
    ) -> None:
        self.type = "bpe"
        self.name = name
        self.pad_id, self.bos_id, self.eos_id, self.sep_id = (
            PAD_ID, BOS_ID, EOS_ID, SEP_ID
        )
        self._enc = None
        self._fallback = None
        try:
            import tiktoken  # type: ignore[import-not-found]

            self._enc = tiktoken.get_encoding(name)
        except Exception:
            # Offline / dependency-free fallback; no network call is retried.
            self._fallback = ByteTokenizer(vocab_size=vocab_size)
        native = getattr(self._enc, "n_vocab", 0)
        self.vocab_size = int(vocab_size) if vocab_size else int(native + 4 or VOCAB_SIZE)
        self._max_native = max(0, self.vocab_size - 4)
        if self._enc is not None and native > self._max_native:
            # Configured embedding table is too small for this BPE alphabet;
            # fall back rather than silently dropping every real token.
            self._enc = None
        if self._enc is None:
            self._fallback = ByteTokenizer(vocab_size=self.vocab_size)

    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> list[int]:
        if self._enc is None:
            ids = self._fallback.encode(text)  # type: ignore[union-attr]
        else:
            ids = [value + 4 for value in self._enc.encode(text) if value < self._max_native]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        if self._enc is None:
            return self._fallback.decode(ids, skip_special=skip_special)  # type: ignore[union-attr]
        native = []
        for value in ids:
            value = int(value)
            if value < 4:
                if not skip_special:
                    native.append(value)
                continue
            native.append(value - 4)
        try:
            return self._enc.decode(
                [value for value in native if value < self._max_native],
                errors="replace",
            )
        except Exception:
            return ""

    def token_text(self, token_id: int) -> str:
        if token_id < 4:
            return ID_TO_SPECIAL.get(int(token_id), f"<{int(token_id)}>")
        if self._enc is None:
            return self._fallback.token_text(int(token_id))  # type: ignore[union-attr]
        try:
            return self._enc.decode_single_token_bytes(int(token_id) - 4).decode(
                "utf-8", errors="replace"
            )
        except Exception:
            return f"<0x{int(token_id) - 4:02x}>"

    def token_bytes(self, token_id: int) -> bytes | None:
        if token_id < 4:
            return None
        if self._enc is None:
            return self._fallback.token_bytes(int(token_id))  # type: ignore[union-attr]
        try:
            return self._enc.decode_single_token_bytes(int(token_id) - 4)
        except Exception:
            return None

    def encode_batch(self, texts: Iterable[str], **kwargs) -> list[list[int]]:
        return [self.encode(text, **kwargs) for text in texts]

    def pad_batch(
        self,
        sequences: Sequence[Sequence[int]],
        pad_to: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = list(sequences)
        if not sequences:
            raise ValueError("empty batch")
        width = max(len(seq) for seq in sequences)
        if pad_to is not None:
            width = max(width, pad_to)
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, : len(seq)] = 1
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return ids, mask

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "type": "bpe",
                    "name": self.name,
                    "vocab_size": self.vocab_size,
                    "specials": SPECIAL_TOKENS,
                    "offline_fallback": self._enc is None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def build_tokenizer(config) -> ByteTokenizer | BPETokenizer:
    """Construct the tokenizer selected by a ModelConfig (or any duck type)."""
    if getattr(config, "tokenizer_type", "char") == "bpe":
        return BPETokenizer(
            name=getattr(config, "bpe_name", "cl100k_base"),
            vocab_size=getattr(config, "vocab_size", None),
        )
    return ByteTokenizer(vocab_size=getattr(config, "vocab_size", None))


def load_tokenizer(path: str | Path | None = None) -> ByteTokenizer | BPETokenizer:
    if path is not None and Path(path).exists():
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = meta.get("type", "byte")
        if kind == "byte":
            return ByteTokenizer(vocab_size=meta.get("vocab_size"))
        if kind == "bpe":
            return BPETokenizer(
                name=meta.get("name", "cl100k_base"),
                vocab_size=meta.get("vocab_size"),
            )
        raise ValueError("unsupported tokenizer type")
    return ByteTokenizer()
