"""Pluggable embedding interface for tag discovery (Stage C). Embedding proximity is ONE factor
among three in DISCOVER — never the mechanism (the owner's scoped ambivalence on vectors). The
default production embedder is a small local CPU model (sentence-transformers MiniLM, an OPTIONAL
extra: `pip install localharness[embeddings]`); when it is unavailable the factory falls back to a
dependency-free deterministic HashingEmbedder so discovery still has an embedding leg rather than
blocking the idle cycle. A CHANGED embedder needs a re-embed pass — vectors from different models
are not comparable. Tests inject their own fake — the INTERFACE is the point, not the model.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Dependency-free deterministic bag-of-words hashing to a fixed dim, L2-normalized. NOT a
    learned semantic model — a cheap, always-available fallback (and the offline/CI default) so
    discovery's embedding leg works without the optional ML dep. Texts sharing vocabulary land
    close (higher cosine); disjoint vocabularies land far. Same box, no GPU, no network."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            if len(tok) < 4:  # drop short/stopword-ish tokens (matches the subsystem's >=4 floor)
                continue
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class SentenceTransformerEmbedder:
    """The default production embedder: a small local CPU model (MiniLM). Lazy-imports the optional
    dep at construction (raises if the extra is not installed), so importing this module is free."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # optional extra: [embeddings]

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in
                self._model.encode(texts, normalize_embeddings=True)]


def default_embedder(model_name: str | None = None) -> Embedder:
    """MiniLM if the `embeddings` extra is installed, else the dep-free HashingEmbedder. Never
    None — the disabled path (fall back to a stricter temporal+trace 2-factor rule) is a config
    choice that discovery handles by being passed embedder=None explicitly."""
    try:
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return HashingEmbedder()
