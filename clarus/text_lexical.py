"""Zero-dependency hashing TF-IDF encoder.

The encoder is the lexical counterpart to :mod:`clarus.text_topology`: it
projects token n-grams into a fixed-size bucket vector via the signed
hashing trick, then weights buckets by inverse document frequency learned
on the training corpus. The output is L2-normalised so Euclidean distance
between two vectors equals ``2 - 2 * cosine_similarity`` and can be mixed
directly with topology features inside a centroid classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass
class LexicalEncoder:
    """Hashing TF-IDF over unigrams and bigrams.

    Parameters
    ----------
    dim:
        Number of hash buckets (output dimension). Larger reduces collisions
        at a linear memory cost.
    ngram_range:
        Inclusive (low, high) n-gram orders to emit, e.g. (1, 2) for
        unigrams + bigrams.
    sublinear_tf:
        If True, replace raw term counts with ``1 + log(count)`` so a single
        very frequent token cannot dominate the vector.
    """

    dim: int = 1024
    ngram_range: tuple[int, int] = (1, 2)
    sublinear_tf: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        low, high = self.ngram_range
        if low < 1 or high < low:
            raise ValueError("ngram_range must satisfy 1 <= low <= high")
        self._idf: list[float] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._idf is not None

    @property
    def idf(self) -> list[float]:
        if self._idf is None:
            raise RuntimeError("encoder is not fitted")
        return self._idf

    def _hash(self, term: str) -> tuple[int, int]:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:6], "little") % self.dim
        sign = 1 if (digest[6] & 1) else -1
        return bucket, sign

    def _ngrams(self, tokens: Sequence[str]) -> Iterable[str]:
        low, high = self.ngram_range
        for n in range(low, high + 1):
            if n <= 0 or n > len(tokens):
                continue
            for i in range(len(tokens) - n + 1):
                yield " ".join(tokens[i: i + n])

    def _signed_counts(self, text: str) -> list[float]:
        counts = [0.0] * self.dim
        tokens = _tokenize(text)
        if not tokens:
            return counts
        for term in self._ngrams(tokens):
            bucket, sign = self._hash(term)
            counts[bucket] += float(sign)
        if self.sublinear_tf:
            counts = [
                math.copysign(1.0 + math.log1p(abs(c) - 1.0), c) if abs(c) > 1.0 else c
                for c in counts
            ]
        return counts

    def fit(self, corpus: Iterable[str]) -> "LexicalEncoder":
        """Estimate inverse document frequency from a corpus."""
        document_frequency = [0] * self.dim
        n_docs = 0
        for text in corpus:
            n_docs += 1
            tokens = _tokenize(text)
            seen: set[int] = set()
            for term in self._ngrams(tokens):
                bucket, _ = self._hash(term)
                if bucket in seen:
                    continue
                seen.add(bucket)
                document_frequency[bucket] += 1
        if n_docs == 0:
            self._idf = [1.0] * self.dim
            return self
        self._idf = [
            math.log((1.0 + n_docs) / (1.0 + df)) + 1.0 for df in document_frequency
        ]
        return self

    def encode(self, text: str) -> list[float]:
        """Project text into the unit-norm hashing TF-IDF space."""
        idf = self.idf
        counts = self._signed_counts(text)
        weighted = [c * idf[i] for i, c in enumerate(counts)]
        norm = math.sqrt(sum(x * x for x in weighted))
        if norm <= 1e-12:
            return weighted
        inv = 1.0 / norm
        return [x * inv for x in weighted]


__all__ = ["LexicalEncoder"]
