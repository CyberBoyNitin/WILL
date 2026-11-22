"""Deterministic synthetic datasets for every research task family.

All generators are seeded.  Train/validation/test splits are produced with
different seeds and, for the OOD benchmarks, a higher complexity level.  No
test example is ever used for training.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from backend.engine.tokenizer import BOS_ID, EOS_ID, SEP_ID, ByteTokenizer

TASK_NAMES = (
    "lm", "arithmetic", "arith_add", "multihop", "sequence", "longctx", "mixture"
)

WORDS = [
    "amber", "basil", "cedar", "delta", "ember", "fable", "gamma", "harbor",
    "ivory", "juniper", "kestrel", "lumen", "maple", "nectar", "onyx", "pearl",
    "quartz", "raven", "saffron", "tundra", "umber", "violet", "willow", "xenon",
]


@dataclass
class Example:
    task: str
    prompt: str
    answer: str
    complexity: int = 1
    meta: dict[str, Any] = field(default_factory=dict)


def _example(
    task: str,
    prompt: str,
    answer: str,
    complexity: int,
    meta: dict | None = None,
) -> Example:
    return Example(task, prompt, answer, complexity, meta or {})


# ----------------------------------------------------------------------
# Task generators
# ----------------------------------------------------------------------
def gen_arithmetic(rng: random.Random, complexity: int) -> Example:
    if complexity <= 1:
        lo, hi = 1, 9
    elif complexity == 2:
        lo, hi = 10, 99
    else:
        lo, hi = 100, 999
    op = rng.choice(["+", "-", "*", "cmp", "chain"])
    if op == "*" and complexity > 2:
        op = "+"
    if op == "+":
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        return _example("arithmetic", f"calc {a} + {b} =", str(a + b), complexity)
    if op == "-":
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        a, b = max(a, b), min(a, b)
        return _example("arithmetic", f"calc {a} - {b} =", str(a - b), complexity)
    if op == "*":
        a, b = rng.randint(1, 12), rng.randint(1, 12)
        return _example("arithmetic", f"calc {a} * {b} =", str(a * b), complexity)
    if op == "cmp":
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if a == b:
            ans = "equal"
        elif a > b:
            ans = "greater"
        else:
            ans = "less"
        return _example("arithmetic", f"compare {a} and {b}", ans, complexity)
    # parenthesised two-step expression
    a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    if rng.random() < 0.5:
        prompt = f"calc ({a} + {b}) * {c} ="
        ans = (a + b) * c
    else:
        prompt = f"calc {a} + {b} * {c} ="
        ans = a + b * c
    return _example("arithmetic", prompt, str(ans), complexity, {"multi_step": True})


def gen_arith_add(rng: random.Random, complexity: int) -> Example:
    """Focused addition task with a controlled digit-width curriculum."""
    if complexity <= 1:
        lo, hi = 0, 9
    elif complexity == 2:
        lo, hi = 10, 99
    else:
        lo, hi = 100, 999
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    return _example("arith_add", f"calc {a} + {b} =", str(a + b), complexity)


def gen_multihop(rng: random.Random, complexity: int) -> Example:
    chain_len = min(3, max(1, complexity))
    n_entities = chain_len + rng.randint(2, 4)
    entities = [f"e{rng.randint(0, 99)}" for _ in range(n_entities)]
    while len(set(entities)) < len(entities):
        entities = [f"e{rng.randint(0, 99)}" for _ in range(n_entities)]
    start = entities[0]
    path = [start]
    remaining = entities[1:]
    rng.shuffle(remaining)
    for i in range(chain_len):
        path.append(remaining[i])
    item = f"item{rng.randint(1, 999)}"
    facts = []
    for a, b in zip(path[:-1], path[1:]):
        facts.append(f"{a} links {b}")
    facts.append(f"{path[-1]} owns {item}")
    # Distractors never contradict the answer path.
    used = set(path)
    for e in remaining[chain_len:]:
        if e in used:
            continue
        other = rng.choice(entities)
        if other != e:
            facts.append(f"{e} links {other}")
    rng.shuffle(facts)
    prompt = "facts: " + ". ".join(facts) + f". question: who owns {item}?"
    return _example(
        "multihop", prompt, path[-1], complexity, {"chain": path, "item": item}
    )


def gen_sequence(rng: random.Random, complexity: int) -> Example:
    length = {1: 3, 2: 4, 3: 5, 4: 6}.get(complexity, 6)
    kind = rng.choice(["sort", "parity", "max", "next", "logic"])
    if kind == "sort":
        nums = [rng.randint(0, 9) for _ in range(length)]
        return _example(
            "sequence", "sort " + " ".join(map(str, nums)),
            " ".join(map(str, sorted(nums))), complexity,
        )
    if kind == "parity":
        bits = [rng.randint(0, 1) for _ in range(length)]
        ans = "odd" if sum(bits) % 2 else "even"
        return _example("sequence", "parity " + " ".join(map(str, bits)), ans, complexity)
    if kind == "max":
        nums = [rng.randint(0, 9) for _ in range(length)]
        return _example("sequence", "max " + " ".join(map(str, nums)), str(max(nums)), complexity)
    if kind == "next":
        start = rng.randint(0, 5)
        step = rng.randint(2, 5)
        seq = [start + step * i for i in range(length)]
        return _example(
            "sequence", "next " + " ".join(map(str, seq)), str(seq[-1] + step), complexity
        )
    op = rng.choice(["and", "or", "xor", "not"])
    if op == "not":
        bit = rng.randint(0, 1)
        return _example("sequence", f"logic not {bit}", str(1 - bit), complexity)
    a, b = rng.randint(0, 1), rng.randint(0, 1)
    ans = {"and": a & b, "or": a | b, "xor": a ^ b}[op]
    return _example("sequence", f"logic {op} {a} {b}", str(ans), complexity)


def gen_longctx(rng: random.Random, complexity: int) -> Example:
    n_records = {1: 2, 2: 4, 3: 8, 4: 12}.get(complexity, 4)
    code_pos = rng.randrange(n_records + 1)
    code = f"{rng.choice(['alpha', 'beta', 'gamma'])}_{rng.randint(1000, 9999)}"
    parts: list[str] = []
    record_words: dict[str, str] = {}
    for i in range(n_records + 1):
        if i == code_pos:
            parts.append(f"access code {code}.")
        if i < n_records:
            word = rng.choice(WORDS)
            record_words[f"r{i}"] = word
            parts.append(f"r{i} {word}.")
    if code_pos > n_records:
        parts.append(f"access code {code}.")
    question_type = rng.choice(["code", "record"])
    if question_type == "code":
        question = "what is the access code?"
        answer = code
    else:
        key = rng.choice(list(record_words.keys()))
        question = f"what is the value of {key}?"
        answer = record_words[key]
    prompt = "records: " + " ".join(parts) + " question: " + question
    return _example(
        "longctx", prompt, answer, complexity,
        {"code": code, "n_records": n_records, "question_type": question_type},
    )


def gen_lm(rng: random.Random, complexity: int) -> Example:
    nouns = ["model", "memory", "signal", "token", "layer", "gradient", "system", "matrix"]
    verbs = ["processes", "stores", "updates", "reads", "routes", "refines", "tracks", "aligns"]
    adjs = ["small", "local", "adaptive", "recurrent", "sparse", "stable", "learned", "bounded"]
    templates = [
        "the {adj} {noun} {verb} the {noun}.",
        "a {noun} {verb} a {adj} {noun}.",
        "the {noun} is {adj} and the {noun} {verb}.",
        "when the {noun} {verb}, the {adj} {noun} updates.",
        "the {adj} {noun} {verb} a {adj} {noun}.",
    ]
    sentences = []
    for _ in range(rng.randint(2, 5)):
        t = rng.choice(templates)
        sentences.append(
            t.format(
                noun=rng.choice(nouns),
                verb=rng.choice(verbs),
                adj=rng.choice(adjs),
            )
        )
    text = " ".join(sentences)
    return _example("lm", text, "", complexity)


GENERATORS: dict[str, Callable[[random.Random, int], Example]] = {
    "arithmetic": gen_arithmetic,
    "arith_add": gen_arith_add,
    "multihop": gen_multihop,
    "sequence": gen_sequence,
    "longctx": gen_longctx,
    "lm": gen_lm,
}


# ----------------------------------------------------------------------
# Tokenised dataset
# ----------------------------------------------------------------------
def build_supervised_tokens(
    tokenizer: ByteTokenizer,
    prompt: str,
    answer: str,
    max_seq_len: int,
) -> tuple[list[int], list[int], list[int], list[int], int]:
    """Return input_ids, labels, attention_mask, prompt_ids, answer_start.

    The full sequence is [BOS] + prompt + [SEP] + answer + [EOS].  The model
    input drops the final token and labels are shifted by one.  Loss is masked
    to answer tokens (and EOS).
    """
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer, add_eos=True)
    # Leave room for BOS, SEP and the answer.
    available = max_seq_len - 2 - len(answer_ids)
    if available < 4:
        answer_ids = answer_ids[: max(1, max_seq_len - 8)]
        answer_ids[-1] = EOS_ID
        available = max_seq_len - 2 - len(answer_ids)
    prompt_ids = prompt_ids[-max(4, available):]
    full = [BOS_ID] + prompt_ids + [SEP_ID] + answer_ids
    full = full[:max_seq_len]
    if full[-1] != EOS_ID and len(full) < max_seq_len:
        full.append(EOS_ID)
    answer_start = 1 + len(prompt_ids) + 1
    input_ids = full[:-1]
    labels = full[1:]
    # Label at input position i predicts full[i+1].  Keep labels from the
    # position immediately before the first answer token.
    keep_from = answer_start - 1
    labels = [lab if i >= keep_from else -100 for i, lab in enumerate(labels)]
    mask = [1] * len(input_ids)
    prompt_with_sep = [BOS_ID] + prompt_ids + [SEP_ID]
    return input_ids, labels, mask, prompt_with_sep, answer_start


def build_lm_tokens(
    tokenizer: ByteTokenizer, text: str, max_seq_len: int
) -> tuple[list[int], list[int], list[int], list[int], int]:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_len=max_seq_len)
    input_ids = ids[:-1]
    labels = ids[1:]
    mask = [1] * len(input_ids)
    return input_ids, labels, mask, ids, 1


class SyntheticDataset(Dataset):
    def __init__(
        self,
        task: str,
        split: str,
        size: int,
        seed: int,
        tokenizer: ByteTokenizer,
        max_seq_len: int,
        complexity: int | None = None,
        min_complexity: int = 1,
        max_complexity: int = 2,
        complexity_mix: list[tuple[int, float]] | None = None,
    ) -> None:
        if task not in TASK_NAMES:
            raise ValueError(f"unknown task {task!r}")
        if complexity is not None and complexity_mix is not None:
            raise ValueError("pass either complexity or complexity_mix, not both")
        if complexity_mix is not None:
            if not complexity_mix:
                raise ValueError("complexity_mix must not be empty")
            total = sum(weight for _, weight in complexity_mix)
            if total <= 0 or any(weight <= 0 for _, weight in complexity_mix):
                raise ValueError("complexity_mix weights must be positive")
            if any(level < 1 for level, _ in complexity_mix):
                raise ValueError("complexity_mix levels must be >= 1")
            complexity_mix = [
                (int(level), float(weight) / total) for level, weight in complexity_mix
            ]
        self.task = task
        self.split = split
        self.size = size
        self.seed = seed
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.complexity = complexity
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.complexity_mix = complexity_mix
        self.complexity_counts: dict[int, int] = {}
        self.items: list[dict[str, Any]] = []
        self._build()

    def _complexity_schedule(self, rng: random.Random) -> list[int] | None:
        """Exact-count mixed schedule, shuffled with the dataset seed.

        Flooring the weighted counts leaves a few slots; those go to the
        largest fractional parts so a 70/30 request yields 70/30 exactly
        whenever ``size`` is divisible by ten.
        """
        if self.complexity_mix is None:
            return None
        raw = [self.size * weight for _, weight in self.complexity_mix]
        counts = [int(math.floor(value)) for value in raw]
        remaining = self.size - sum(counts)
        order = sorted(
            range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True
        )
        for i in order[:remaining]:
            counts[i] += 1
        schedule = [
            level
            for (level, _), count in zip(self.complexity_mix, counts)
            for _ in range(count)
        ]
        rng.shuffle(schedule)
        return schedule

    def _build(self) -> None:
        rng = random.Random(self.seed + {"train": 0, "val": 100_003, "test": 200_003}[self.split])
        tasks = (
            ["arith_add", "multihop", "sequence", "longctx"]
            if self.task == "mixture"
            else [self.task]
        )
        schedule = self._complexity_schedule(rng)
        for i in range(self.size):
            task = tasks[i % len(tasks)]
            if schedule is not None:
                complexity = schedule[i]
            else:
                complexity = (
                    self.complexity
                    if self.complexity is not None
                    else rng.randint(self.min_complexity, self.max_complexity)
                )
            self.complexity_counts[complexity] = (
                self.complexity_counts.get(complexity, 0) + 1
            )
            ex = GENERATORS[task](rng, complexity)
            if task == "lm":
                ids, labels, mask, prompt_ids, start = build_lm_tokens(
                    self.tokenizer, ex.prompt, self.max_seq_len
                )
                answer_text = ""
            else:
                ids, labels, mask, prompt_ids, start = build_supervised_tokens(
                    self.tokenizer, ex.prompt, ex.answer, self.max_seq_len
                )
                answer_text = ex.answer
            self.items.append(
                {
                    "input_ids": ids,
                    "labels": labels,
                    "attention_mask": mask,
                    "prompt_ids": prompt_ids,
                    "answer_text": answer_text,
                    "answer_start": start,
                    "task": task,
                    "complexity": complexity,
                    "prompt": ex.prompt,
                    "meta": ex.meta,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def collate_lm(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in batch)
    b = len(batch)
    ids = torch.zeros((b, max_len), dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    mask = torch.zeros((b, max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(x["labels"], dtype=torch.long)
        mask[i, :n] = 1
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "items": batch,
    }


def collate_generation(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, Any]:
    max_len = max(len(x["prompt_ids"]) for x in batch)
    b = len(batch)
    ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((b, max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["prompt_ids"])
        ids[i, :n] = torch.tensor(x["prompt_ids"], dtype=torch.long)
        mask[i, :n] = 1
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "items": batch,
    }


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def make_dataset(
    task: str,
    split: str,
    size: int,
    seed: int,
    tokenizer: ByteTokenizer,
    max_seq_len: int,
    complexity: int | None = None,
    min_complexity: int = 1,
    max_complexity: int = 2,
    complexity_mix: list[tuple[int, float]] | None = None,
) -> SyntheticDataset:
    return SyntheticDataset(
        task=task,
        split=split,
        size=size,
        seed=seed,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        complexity=complexity,
        min_complexity=min_complexity,
        max_complexity=max_complexity,
        complexity_mix=complexity_mix,
    )


def make_mixed_dataset(
    task: str,
    split: str,
    size: int,
    seed: int,
    tokenizer: ByteTokenizer,
    max_seq_len: int,
    easy_ratio: float = 0.7,
    easy_complexity: int = 1,
    hard_complexity: int = 2,
) -> SyntheticDataset:
    """70/30-style mixed-difficulty split with exact stratum counts.

    ``easy_ratio`` is the fraction of examples at ``easy_complexity``; the
    remainder use ``hard_complexity``.  The composition is exact up to the
    largest-remainder rounding implemented by :class:`SyntheticDataset`.
    """
    if not 0.0 < easy_ratio < 1.0:
        raise ValueError("easy_ratio must be in (0, 1)")
    if easy_complexity == hard_complexity:
        raise ValueError("easy_complexity and hard_complexity must differ")
    return make_dataset(
        task=task,
        split=split,
        size=size,
        seed=seed,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        complexity_mix=[
            (easy_complexity, easy_ratio),
            (hard_complexity, 1.0 - easy_ratio),
        ],
        min_complexity=min(easy_complexity, hard_complexity),
        max_complexity=max(easy_complexity, hard_complexity),
    )


def dataset_manifest(ds: SyntheticDataset) -> dict[str, Any]:
    return {
        "task": ds.task,
        "split": ds.split,
        "size": len(ds),
        "seed": ds.seed,
        "complexity": ds.complexity,
        "min_complexity": ds.min_complexity,
        "max_complexity": ds.max_complexity,
        "complexity_mix": ds.complexity_mix,
        "complexity_counts": {
            str(level): count for level, count in sorted(ds.complexity_counts.items())
        },
        "max_seq_len": ds.max_seq_len,
    }
