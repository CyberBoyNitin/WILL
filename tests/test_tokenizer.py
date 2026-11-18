"""Byte tokenizer tests."""
from __future__ import annotations

import torch

from backend.engine.tokenizer import BOS_ID, EOS_ID, PAD_ID, SEP_ID


def test_roundtrip_unicode(tokenizer):
    text = "héllo 世界 — adaptive computation"
    ids = tokenizer.encode(text)
    assert all(4 <= i < 260 for i in ids)
    assert tokenizer.decode(ids) == text


def test_special_tokens_and_max_len(tokenizer):
    ids = tokenizer.encode("abc", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert tokenizer.encode("abcdef", max_len=3) == tokenizer.encode("abc")
    assert tokenizer.decode([BOS_ID, SEP_ID, EOS_ID], skip_special=True) == ""


def test_pad_batch(tokenizer):
    ids, mask = tokenizer.pad_batch([[BOS_ID, 10, 11], [BOS_ID, 12]])
    assert ids.shape == (2, 3)
    assert ids[0, 0].item() == BOS_ID
    assert ids[1, 2].item() == PAD_ID
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert isinstance(ids, torch.Tensor)
