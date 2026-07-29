"""Tests for gguf_tokenizer's GGUF path RESOLVERS only.

Never constructs a real GgufTokenizer / imports llama_cpp — that's an optional, source-built
dependency (the 'exact-tokenizer' extra) not guaranteed present in every test environment.
Every resolver call here is pointed at a tmp_path fixture (via the OLLAMA_MODELS /
LMSTUDIO_MODELS_DIR env vars) so nothing touches the real ~/.ollama or ~/.lmstudio on a box
that has them installed.
"""
import json
import subprocess

from localharness.agent.gguf_tokenizer import (
    load_gguf_tokenizer,
    resolve_gguf_path,
)
from localharness.agent import gguf_tokenizer as gguf_mod


# ---------------------------------------------------------------------------
# Ollama: manifest -> .model layer digest -> blob
# ---------------------------------------------------------------------------


def _write_ollama_manifest(tmp_path, name: str, tag: str, layers: list[dict]) -> None:
    manifest_dir = tmp_path / "manifests" / "registry.ollama.ai" / "library" / name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / tag).write_text(json.dumps({"layers": layers}))


def test_resolve_ollama_gguf_from_manifest_digest(tmp_path, monkeypatch):
    """name:tag manifest with a `.model` layer among others resolves to that layer's blob."""
    _write_ollama_manifest(tmp_path, "qwen2.5", "7b", [
        {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:paramsdigest"},
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc123"},
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:tpldigest"},
    ])
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    blob = blobs_dir / "sha256-abc123"
    blob.write_bytes(b"fake gguf bytes")

    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert resolve_gguf_path("qwen2.5:7b", "ollama") == blob


def test_resolve_ollama_gguf_bare_name_defaults_to_latest_tag(tmp_path, monkeypatch):
    """A bare name (no ':tag') resolves against the 'latest' manifest."""
    _write_ollama_manifest(tmp_path, "qwen2.5", "latest", [
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:def456"},
    ])
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    blob = blobs_dir / "sha256-def456"
    blob.write_bytes(b"data")

    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert resolve_gguf_path("qwen2.5", "ollama") == blob


def test_resolve_ollama_gguf_missing_manifest_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert resolve_gguf_path("nonexistent:7b", "ollama") is None


def test_resolve_ollama_gguf_no_model_layer_returns_none(tmp_path, monkeypatch):
    """A manifest with layers but none of mediaType *.model -> None."""
    _write_ollama_manifest(tmp_path, "qwen2.5", "7b", [
        {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:paramsdigest"},
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:tpldigest"},
    ])
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert resolve_gguf_path("qwen2.5:7b", "ollama") is None


def test_resolve_ollama_gguf_missing_blob_file_returns_none(tmp_path, monkeypatch):
    """The manifest names a .model digest but the blob file was never pulled -> None, not a
    dangling path."""
    _write_ollama_manifest(tmp_path, "qwen2.5", "7b", [
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:missing"},
    ])
    # deliberately no blobs/ dir at all
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert resolve_gguf_path("qwen2.5:7b", "ollama") is None


# ---------------------------------------------------------------------------
# LM Studio: filename-scan fallback (no `lms` binary) + the `lms ls --json` path
# ---------------------------------------------------------------------------


def test_resolve_lmstudio_gguf_filename_scan_fallback(tmp_path, monkeypatch):
    """With no `lms` CLI on the box, resolution falls back to a best-effort *.gguf filename
    scan under the models root, matching the served id against the path (case/colon-insensitive)."""
    monkeypatch.setattr(gguf_mod, "_lms_binary", lambda: None)
    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(tmp_path))
    model_dir = tmp_path / "pub" / "Qwen2.5-0.5B-Instruct-GGUF"
    model_dir.mkdir(parents=True)
    gguf_file = model_dir / "Qwen2.5-0.5B-Instruct-Q8_0.gguf"
    gguf_file.write_bytes(b"fake gguf bytes")

    assert resolve_gguf_path("qwen2.5-0.5b-instruct", "lmstudio") == gguf_file


def test_resolve_lmstudio_gguf_via_lms_cli(tmp_path, monkeypatch):
    """When `lms` IS present, `lms ls --json`'s modelKey->path map is authoritative — the
    filename scan is never even consulted."""
    model_file = tmp_path / "pub" / "repo" / "model.gguf"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"data")

    class _FakeCompleted:
        returncode = 0
        stdout = '[{"modelKey":"qwen2.5-0.5b-instruct","path":"pub/repo/model.gguf"}]'

    def fake_run(cmd, capture_output=True, timeout=15, text=True):
        return _FakeCompleted()

    monkeypatch.setattr(gguf_mod, "_lms_binary", lambda: "/fake/lms")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(tmp_path))

    assert resolve_gguf_path("qwen2.5-0.5b-instruct", "lmstudio") == model_file


# ---------------------------------------------------------------------------
# resolve_gguf_path: provider/model edge cases
# ---------------------------------------------------------------------------


def test_resolve_gguf_path_unknown_provider_returns_none():
    """Only ollama/lmstudio need on-disk resolution — vllm/llamacpp count exactly over
    /tokenize and never call this."""
    assert resolve_gguf_path("qwen2.5:7b", "vllm") is None


def test_resolve_gguf_path_none_model_returns_none():
    assert resolve_gguf_path(None, "ollama") is None


# ---------------------------------------------------------------------------
# load_gguf_tokenizer: unresolvable model short-circuits before ever touching llama_cpp
# ---------------------------------------------------------------------------


def test_load_gguf_tokenizer_returns_none_when_unresolvable(tmp_path, monkeypatch):
    """No manifest for this model under an (empty) models root -> resolve_gguf_path returns
    None -> load_gguf_tokenizer returns None without ever importing llama_cpp."""
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert load_gguf_tokenizer("nonexistent", "ollama") is None
