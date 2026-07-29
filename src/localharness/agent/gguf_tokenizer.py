"""The harness's OWN tokenizer — exact token counting from the model's real GGUF vocab.

Ollama and LM Studio serve no `/tokenize` endpoint (unlike vLLM/llama.cpp), so token counting
there used to fall back to an APPROXIMATE cl100k estimate — wrong for Qwen-family models by ~15-20%
depending on content, which corrupts every context-budget / compaction / memory decision that rests
on the count. This module gives the harness its own token eyes instead: it loads the served model's
REAL tokenizer straight from the GGUF the server is already running (via `llama-cpp-python` in
`vocab_only` mode — just the vocab + metadata, no weights, no GPU), so counts are EXACT, offline,
and match the server to the token (verified live: local render+tokenize == LM Studio's own
`usage.prompt_tokens` across short/system/multi-turn/long-code cases). Nothing leaves the apparatus.

Two provider-specific resolvers find the GGUF on disk from the served model id:
- **LM Studio**: `lms ls --json` maps `modelKey` -> a path under `~/.lmstudio/models/`.
- **Ollama**: the manifest's `application/vnd.ollama.image.model` layer digest -> a blob under
  `~/.ollama/models/blobs/` (a raw GGUF, identified by magic bytes, no extension).

Requires `llama-cpp-python` (an optional extra; there is no prebuilt aarch64 wheel, so it is a
source build). When it or the GGUF is unreachable — e.g. a harness pointed at a REMOTE server with
no local model files — the caller falls back to the labeled-approximate estimator, never a silent
guess.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger("localharness.gguf_tokenizer")


# ---------------------------------------------------------------------------
# GGUF path resolution (served model id -> the GGUF file on disk)
# ---------------------------------------------------------------------------


def _lmstudio_models_root() -> Path:
    return Path(os.environ.get("LMSTUDIO_MODELS_DIR", "~/.lmstudio/models")).expanduser()


def _lms_binary() -> str | None:
    """The `lms` CLI, for the authoritative modelKey->path map. Not on PATH by default."""
    from shutil import which
    cand = Path("~/.lmstudio/bin/lms").expanduser()
    return str(cand) if cand.exists() else which("lms")


def _resolve_lmstudio_gguf(model: str) -> Path | None:
    """`lms ls --json` gives the authoritative modelKey -> relative-path map; join onto the models
    root. Falls back to a filename scan when the `lms` CLI is absent."""
    root = _lmstudio_models_root()
    lms = _lms_binary()
    if lms:
        try:
            out = subprocess.run([lms, "ls", "--json"], capture_output=True, timeout=15, text=True)
            if out.returncode == 0:
                for entry in json.loads(out.stdout or "[]"):
                    if entry.get("modelKey") == model and entry.get("path"):
                        p = root / entry["path"]
                        if p.exists():
                            return p
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    # Fallback: match the served id against a *.gguf FILENAME stem under the models root. A prefix
    # match on the filename (not a substring of the full path) so a short/common id can't false-match
    # a directory name (e.g. `lmstudio-community`) and return the WRONG model's tokenizer.
    if root.is_dir():
        norm = model.lower().replace(":", "-")
        for gguf in root.rglob("*.gguf"):
            if gguf.stem.lower().replace(":", "-").startswith(norm):
                return gguf
    return None


def _ollama_models_root() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS", "~/.ollama/models")).expanduser()


def _resolve_ollama_gguf(model: str) -> Path | None:
    """Parse the OCI manifest for the tag, take the `…image.model` layer digest, resolve to its blob.
    Handles `name:tag` (default registry `registry.ollama.ai/library`), a `latest` default tag, and
    an explicit-registry tag like `hf.co/user/repo:tag`."""
    root = _ollama_models_root()
    ref, _, tag = model.partition(":")
    tag = tag or "latest"
    parts = ref.split("/")
    if len(parts) == 1:            # bare name -> the default library namespace
        registry, namespace, name = "registry.ollama.ai", "library", parts[0]
    elif len(parts) == 2:          # user/name -> default registry
        registry, namespace, name = "registry.ollama.ai", parts[0], parts[1]
    else:                          # registry/user/name (e.g. hf.co/user/repo)
        registry, namespace, name = parts[0], "/".join(parts[1:-1]), parts[-1]
    manifest = root / "manifests" / registry / namespace / name / tag
    try:
        layers = json.loads(manifest.read_text()).get("layers", [])
    except (OSError, ValueError):
        return None
    for layer in layers:
        if str(layer.get("mediaType", "")).endswith(".model"):
            digest = str(layer.get("digest", ""))
            if digest.startswith("sha256:"):
                blob = root / "blobs" / ("sha256-" + digest.split(":", 1)[1])
                return blob if blob.exists() else None
    return None


def resolve_gguf_path(model: str | None, provider_type: str | None) -> Path | None:
    """The on-disk GGUF for a served model, or None if it can't be located (unknown provider, remote
    server with no local files, model not found). Only ollama/lmstudio need this — vLLM/llama.cpp
    already count exactly over `/tokenize`."""
    if not model:
        return None
    ptype = (provider_type or "").lower()
    if ptype == "lmstudio":
        return _resolve_lmstudio_gguf(model)
    if ptype == "ollama":
        return _resolve_ollama_gguf(model)
    return None


# ---------------------------------------------------------------------------
# The tokenizer itself (llama-cpp-python vocab-only over the GGUF)
# ---------------------------------------------------------------------------


class GgufTokenizer:
    """Exact counting from a GGUF's own vocab + embedded chat template. `count` tokenizes raw content
    (the dominant, previously-mis-estimated term); `count_messages` renders the conversation through
    the model's REAL chat template first, so message structure (role markers, the default system
    prompt, the generation-prompt suffix — ~25-30 Qwen tokens/turn the old `+4` guess missed) is
    counted exactly, matching what the server actually tokenizes."""

    def __init__(self, gguf_path: str) -> None:
        from llama_cpp import Llama  # optional dep — import lazily so absence degrades, not crashes
        self._llm = Llama(model_path=gguf_path, vocab_only=True, verbose=False)
        self._formatter: Any = None
        md = self._llm.metadata or {}
        template = md.get("tokenizer.chat_template")
        if template:
            try:
                from llama_cpp.llama_chat_format import Jinja2ChatFormatter
                self._formatter = Jinja2ChatFormatter(
                    template=template,
                    eos_token=self._special_text(self._llm.token_eos()),
                    bos_token=self._special_text(self._llm.token_bos()),
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001 — a template we can't render → fall back to summed count
                self._formatter = None

    def _special_text(self, token_id: int) -> str:
        """The text of a special token (for chat-template `eos_token`/`bos_token` vars). `special=True`
        so a real special token detokenizes to its literal (e.g. `<|im_end|>`), not empty."""
        if token_id is None or token_id < 0:
            return ""
        try:
            return self._llm.detokenize([token_id], special=True).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""

    def count(self, text: str) -> int:
        """Exact token count of raw content. `special=False`: literal special-token text inside user
        content is counted as ordinary text (never treated as a control token), the safe direction."""
        if not text:
            return 0
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def count_messages(self, messages: list[dict]) -> int:
        """Exact count of a full conversation. Renders the model's real chat template then tokenizes
        with `special=True` (so `<|im_start|>` et al. count as one token each) — this is what the
        server does, so the count matches `usage.prompt_tokens` to the token. Falls back to a summed
        content + fixed-overhead estimate only if the template can't be rendered."""
        if self._formatter is not None:
            try:
                prompt = self._formatter(messages=messages).prompt
                return len(self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
            except Exception:  # noqa: BLE001 — a message shape the template rejects → summed fallback
                pass
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content") or ""
            if isinstance(content, str):
                total += self.count(content)
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                total += self.count(fn.get("name", "")) + self.count(fn.get("arguments", ""))
        return total


@lru_cache(maxsize=8)
def _load_cached(gguf_path: str) -> GgufTokenizer:
    return GgufTokenizer(gguf_path)


def load_gguf_tokenizer(model: str | None, provider_type: str | None) -> GgufTokenizer | None:
    """Resolve the served model to its GGUF and load an exact tokenizer over it, or return None when
    that isn't possible (llama-cpp-python not installed, no local GGUF, unknown provider) so the
    caller falls back to the labeled-approximate estimator. Cached per GGUF path (vocab-only load is
    ~1s)."""
    path = resolve_gguf_path(model, provider_type)
    if path is None:
        return None
    try:
        return _load_cached(str(path))
    except ImportError:
        log.warning(
            "llama-cpp-python not installed — exact GGUF token counting unavailable for %s; falling "
            "back to approximate. Install the 'exact-tokenizer' extra for exact counts.", model,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — a bad/unreadable GGUF must not crash the counter
        log.warning("could not load GGUF tokenizer for %s at %s: %s — using approximate.",
                    model, path, exc)
        return None
