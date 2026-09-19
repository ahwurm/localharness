"""Resonance — the model as the similarity engine (memory spec, role 2).

This module is the ONE place the memory system judges what-resembles-what. Every
similarity call — search ranking, dreaming's replay integration, write-time
re-sighting candidates — comes here, and the judgment is made in a LEARNED
representation space: a subject-family embedding model (Qwen3-Embedding, the same
model family as the served subject), never a hand-built discretization. Words,
tokenizers, tag graphs, and overlap rules are banned as mechanism (the no-units /
no-magic-words laws); embeddings are the lane available today, true activations
replace them when vLLM exposes them.

NO FALLBACK, by owner order: if the model cannot load, every resonance judgment
raises `ResonanceUnavailable` and the caller surfaces the error. There is no
hashing, lexical, or statistical stand-in behind this interface.

Vectors are L2-normalized float32, stored as raw bytes; dot product = cosine.
Vectors from different models are not comparable — the store records which model
embedded it (meta key `embed_model`) and dreaming re-embeds everything when the
configured model changes.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from localharness.memory.errors import MemoryError

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class ResonanceUnavailable(MemoryError):
    """The similarity engine cannot run. Raised loudly; never degraded around."""

    def __init__(self, model_name: str, underlying: Exception | str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Resonance engine unavailable (model {model_name!r}): {underlying}. "
            "Memory retrieval runs in the model's representation space and has no "
            "fallback path — install the embeddings extra "
            "(`uv sync --extra embeddings`) and ensure the model weights are "
            "downloadable/cached, then retry."
        )


class ResonanceEngine:
    """Lazy-loading wrapper around the subject-family sentence-embedding model.

    Loading and encoding are CPU-bound and blocking — call through
    `asyncio.to_thread` from async code. A process-wide lock serializes encode
    calls (the underlying model is not safely reentrant); memory traffic is low
    enough that serialization is invisible.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        try:
            from localharness.memory.embeddings import _quiet_ml_output

            with _quiet_ml_output():
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as exc:  # loud, typed, actionable — never swallowed
            raise ResonanceUnavailable(self.model_name, exc) from exc
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- encoding ----------------------------------------------------------

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        """Encode stored-trace / stream-window text. Returns (n, d) float32, L2-normalized."""
        with self._lock:
            model = self._ensure_loaded()
            from localharness.memory.embeddings import _quiet_ml_output

            with _quiet_ml_output():
                vecs = model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False,
                )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a retrieval probe. Uses the model's own query prompt when it
        defines one (Qwen3-Embedding is instruction-aware and ships a `query`
        prompt in its config — the model's metadata, not a hand rule)."""
        with self._lock:
            model = self._ensure_loaded()
            from localharness.memory.embeddings import _quiet_ml_output

            kwargs = {}
            if "query" in (getattr(model, "prompts", None) or {}):
                kwargs["prompt_name"] = "query"
            with _quiet_ml_output():
                vec = model.encode(
                    [text], normalize_embeddings=True, show_progress_bar=False, **kwargs,
                )
        return np.asarray(vec, dtype=np.float32)[0]


# -- vector packing ---------------------------------------------------------

def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _matrix(id_blobs: list[tuple[int, bytes]]) -> tuple[list[int], np.ndarray]:
    ids = [i for i, _ in id_blobs]
    mat = np.stack([unpack(b) for _, b in id_blobs]) if id_blobs else np.zeros((0, 1), np.float32)
    return ids, mat


def rank(query_vec: np.ndarray, id_blobs: list[tuple[int, bytes]]) -> list[tuple[int, float]]:
    """All candidates ordered by resonance with the probe (cosine, descending).
    Ties break by id (deterministic)."""
    if not id_blobs:
        return []
    ids, mat = _matrix(id_blobs)
    sims = mat @ np.asarray(query_vec, dtype=np.float32)
    order = sorted(range(len(ids)), key=lambda i: (-float(sims[i]), ids[i]))
    return [(ids[i], float(sims[i])) for i in order]


def shares(window_vec: np.ndarray, id_blobs: list[tuple[int, bytes]]) -> dict[int, float]:
    """One stream moment's unit of attention, distributed over traces by resonance.

    The zero-sum of the spec lives here: shares are non-negative resonances
    normalized to sum to 1 — a moment has exactly one unit of attention and the
    traces compete for it. Anti-resonant traces (cosine <= 0) receive nothing.
    A moment that resonates with nothing distributes nothing (empty dict).
    """
    if not id_blobs:
        return {}
    ids, mat = _matrix(id_blobs)
    sims = np.maximum(mat @ np.asarray(window_vec, dtype=np.float32), 0.0)
    total = float(sims.sum())
    if total <= 0.0:
        return {}
    return {ids[i]: float(sims[i]) / total for i in range(len(ids)) if sims[i] > 0.0}
