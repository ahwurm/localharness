"""Test-double embedding utilities.

The PRODUCTION similarity engine lives in memory/resonance.py (subject-family model,
no fallback). This module keeps two things: `_quiet_ml_output` (the stream-silencing
guard resonance.py loads the real model under) and `HashingEmbedder` — a dependency-free
deterministic bag-of-words embedder the unit tests inject as their engine double
(the INTERFACE is the point, not the model). Nothing in src/ falls back to it.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@contextlib.contextmanager
def _quiet_ml_output():
    """Silence third-party ML loader chatter around embedder load/encode (#76).

    HF/torch/tqdm write progress ("loading weights…", bars) DIRECTLY to stdout/stderr —
    bypassing the logging system that the #20 fix routes to memory.log — and in an
    interactive session that lands on top of the input box. Env flags turn off what
    honors them; the redirect catches the rest. Errors still surface: exceptions
    propagate unchanged, only stream writes are swallowed."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield


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


