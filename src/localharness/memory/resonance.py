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

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from localharness.memory.errors import MemoryError

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# R24: the one line for a model that names code of its own, and the one for a relative folder.
SHIPS_CODE_LINE = ("memory: the embedding model {model} ships its own code; LocalHarness does not run "
                   "it — choose a model without custom modules")
RELATIVE_FOLDER_LINE = ("memory: the embedding model {model} is a relative path, read from the config "
                        "folder as {folder} (never from the current folder)")

EMBED_MAX_TOKENS = 1024
"""Where the model's input is cut, in tokens, for every encode (facts, stream windows, queries).

The model accepts 32k tokens and runs on CPU, where attention is quadratic in length: one 32k-token
turn window is minutes of every core. The probe's question — what was this moment about, which
stored trace does it resemble — is answered by the head of the text. Set on the loaded model, so the
cut holds for every caller, not only the one that remembered it."""

EMBED_THREADS = min(4, os.cpu_count() or 4)
"""CPU threads torch may use for the embedding model: a background job's share of the box, not the
box (and never more than the box has). Torch's default is every core, and the dreaming pass runs
exactly when nobody is at the keyboard to see it (2026-10-04: a phone session idled into a pass that
held 20 cores for hours)."""

log = logging.getLogger(__name__)


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


class EmbeddingModelRefused(ResonanceUnavailable):
    """The embedding model names code of its own (R24): refused in one line, and memory's embedding
    leg stays off for the session, as for a model that cannot load."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        MemoryError.__init__(self, SHIPS_CODE_LINE.format(model=model_name))


class ResonanceEngine:
    """Lazy-loading wrapper around the subject-family sentence-embedding model.

    Loading and encoding are CPU-bound and blocking — call through
    `asyncio.to_thread` from async code. A process-wide lock serializes encode
    calls (the underlying model is not safely reentrant); memory traffic is low
    enough that serialization is invisible.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL,
                 config_dir: str | os.PathLike | None = None) -> None:
        self.model_name = model_name
        self.config_dir = config_dir  # where a relative folder is read from (None: the machine's)
        self._model = None
        self._refused = False
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        if self._refused:
            raise EmbeddingModelRefused(self.model_name)
        try:
            from localharness.memory.embeddings import _quiet_ml_output

            with _quiet_ml_output():
                from sentence_transformers import SentenceTransformer

                self._model = _bound(_load(SentenceTransformer, self.model_name, self.config_dir))
        except EmbeddingModelRefused as exc:  # said once; every later use is refused as it stands
            self._refused = True
            log.warning("%s", exc)
            raise
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


def _bound(model):
    """`model` with its input cut at EMBED_MAX_TOKENS and torch's CPU threads capped at EMBED_THREADS.

    Neither cap may fail a load: a model object without the attribute, or a torch without thread
    control, keeps the model usable and says so once in the log."""
    try:
        model.max_seq_length = EMBED_MAX_TOKENS
    except Exception:  # noqa: BLE001
        log.warning("memory: could not cap the embedding model's input at %d tokens", EMBED_MAX_TOKENS)
    try:
        import torch

        torch.set_num_threads(EMBED_THREADS)
    except Exception:  # noqa: BLE001
        log.warning("memory: could not cap the embedding model at %d CPU threads", EMBED_THREADS)
    return model


def cached_copy(model: str) -> str | None:
    """The folder of `model`'s copy in the local Hugging Face cache, or None. Reads the cache only,
    as doctor and the start notice do; an id the cache cannot read (a malformed one) is None."""
    try:
        from huggingface_hub import try_to_load_from_cache

        for name in ("modules.json", "config.json"):
            found = try_to_load_from_cache(model, name)
            if isinstance(found, str):
                return os.path.dirname(found)
    except Exception:  # noqa: BLE001 — unreadable from the cache is "not cached", never an error here
        return None
    return None


def model_folder(model: str, config_dir: str | os.PathLike | None = None) -> Path | None:
    """The folder `model` names, as an absolute path, or None when it is a model id. `~` expands. A
    relative path is the config folder's (`config_dir`, else the machine's), and a folder only when
    that folder holds it; never the current folder's, where sentence-transformers would look for a
    name first, so a cloned project could ship one named like the model (R24)."""
    path = Path(model).expanduser()
    if not path.is_absolute():
        if config_dir is None:
            from localharness.config.paths import global_config_dir

            config_dir = global_config_dir()
        path = Path(config_dir).expanduser() / path
        if not path.is_dir():
            return None
    return path.absolute()


def _load(factory, model: str, config_dir: str | os.PathLike | None = None):
    """`model` loaded by `factory` (SentenceTransformer), only from a vetted folder and only by its
    absolute path (R24): the library is never handed a name, which it would resolve against the
    current folder first. A model id is its snapshot in the local Hugging Face cache, and
    huggingface.co is reached only on a real cache miss, after the start notice's line is logged (R8).

    Loaded by id, the model was checked against huggingface.co on every load, and even with
    `local_files_only=True` the hub library still fetched its list of "agent harnesses" for its
    user-agent header whenever its own copy of that list was missing or a day old (huggingface_hub
    1.x). HF_HUB_OFFLINE is no way round it: the library reads it once, at import, and setting it for
    the process would change it for everything else the session runs. A cached copy missing a file
    the model needs (an interrupted download) is a cache miss: the download completes it."""
    folder = model_folder(model, config_dir)
    if folder is not None:
        if not Path(model).expanduser().is_absolute():
            log.warning(RELATIVE_FOLDER_LINE.format(model=model, folder=folder))
        if not folder.is_dir():
            raise FileNotFoundError(f"no such folder: {folder}")
        return _vetted(factory, folder, model)
    repo = _repo(factory, model)
    cached = cached_copy(repo)
    if cached is not None:
        try:
            return _vetted(factory, cached, model)
        except OSError:
            log.debug("the cached copy of %s is incomplete — downloading the rest", model,
                      exc_info=True)
    from huggingface_hub import snapshot_download

    from localharness.memory.plugin import embedding_download_line

    log.warning(embedding_download_line(model))
    return _vetted(factory, snapshot_download(repo), model)


def _repo(factory, model: str) -> str:
    """The hub repo of the model id `model`, as `factory` itself resolves a name with no owner: its
    organisation's (`sentence-transformers/<name>`), except the original transformers models."""
    org = getattr(factory, "default_huggingface_organization", None)
    if not org or "/" in model:
        return model
    try:
        from sentence_transformers.util.misc import ORIGINAL_TRANSFORMER_MODELS as originals
    except ImportError:  # a library without the list keeps every name as given
        return model
    return model if model.lower() in originals else f"{org}/{model}"


def _vetted(factory, folder: str | os.PathLike, model: str):
    """`factory` over the model in `folder`, by its absolute path, once nothing in it names code."""
    folder = Path(folder).absolute()
    found = code_named(folder)
    if found is not None:
        log.debug("embedding model %s: %s", model, found)
        raise EmbeddingModelRefused(model)
    return factory(str(folder), device="cpu")


_OWN = "sentence_transformers."  # the library's own classes, imported from the installed package


def code_named(root: Path) -> str | None:
    """What in the model at `root` names code of its own, or None. Read from its files; nothing is
    imported. Every class reference sentence-transformers resolves while loading it must be the
    library's own, because for anything else it imports the module FROM the model's folder, with no
    `trust_remote_code` asked (5.6; deprecated, removed in v6): each module type in `modules.json` and
    in a Router's (or the older Asym's) config, recursively; a Dense module's activation function
    (torch's); a WordEmbeddings module's tokenizer class. And no config in the model's folders may map
    a class to code (`auto_map`, what transformers runs with trust) or turn `trust_remote_code` on."""
    modules = _json(root / "modules.json")
    todo = [(root, entry) for entry in modules] if isinstance(modules, list) else []
    folders, seen = [root], set()
    while todo:
        parent, entry = todo.pop()
        ref = entry.get("type") if isinstance(entry, dict) else None
        if not (isinstance(ref, str) and ref.startswith(_OWN)):
            return f"module {ref!r}"
        here = parent / str(entry.get("path") or "")
        if (os.path.realpath(here), ref) in seen:
            continue
        seen.add((os.path.realpath(here), ref))
        folders.append(here)
        kind = ref.rsplit(".", 1)[-1]
        if kind in ("Router", "Asym"):
            for name in ("router_config.json", "asym_config.json", "config.json"):
                types = _key(here / name, "types")
                if isinstance(types, dict):
                    todo += [(here, {"type": t, "path": str(k)}) for k, t in types.items()]
        elif kind == "Dense":
            fn = _key(here / "config.json", "activation_function")
            if fn is not None and not (isinstance(fn, str) and fn.startswith("torch.")):
                return f"activation function {fn!r}"
        elif kind == "WordEmbeddings":
            cls = _key(here / "wordembedding_config.json", "tokenizer_class")
            if cls is not None and not (isinstance(cls, str) and cls.startswith(_OWN)):
                return f"tokenizer {cls!r}"
    for here in folders:
        for config in sorted(here.glob("*config*.json")):
            if _asks_for_code(_json(config)):
                return f"custom code in {config}"
    return None


def _json(path: Path):
    """The JSON in `path`, or None when it is missing or unreadable: the library, reading it the same
    way, could not act on it either."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, RecursionError):
        return None


def _key(path: Path, key: str):
    data = _json(path)
    return data.get(key) if isinstance(data, dict) else None


def _asks_for_code(data) -> bool:
    """Does this config, at any depth, map a class to code (`auto_map`) or turn `trust_remote_code` on?"""
    todo = [data]
    while todo:
        item = todo.pop()
        if isinstance(item, dict):
            if "auto_map" in item or item.get("trust_remote_code"):
                return True
            todo += item.values()
        elif isinstance(item, list):
            todo += item
    return False


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
