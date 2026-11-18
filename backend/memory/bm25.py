"""Local BM25 lexical retrieval.  No API keys, no external index."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    docs: list[list[str]] = field(default_factory=list)
    df: dict[str, int] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    avgdl: float = 0.0

    def fit(self, texts: list[str]) -> "BM25Index":
        self.docs = [tokenize(t) for t in texts]
        n = len(self.docs)
        self.df = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }
        self.avgdl = sum(len(d) for d in self.docs) / max(1, n)
        return self

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out: list[float] = []
        for doc in self.docs:
            dl = len(doc)
            score = 0.0
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            for term in q:
                if term not in counts:
                    continue
                tf = counts[term]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1.0) / max(denom, 1e-9)
            out.append(score)
        return out

    def top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scored = list(enumerate(self.scores(query)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
