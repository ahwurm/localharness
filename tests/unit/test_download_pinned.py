from __future__ import annotations

import huggingface_hub
import pytest

from localharness.provider import server
from localharness.provider.refarch import RTX_4060
from localharness.provider.server import HubFile

COMMIT = "a" * 40
DIGEST = "b" * 64


class _Lfs:
    def __init__(self, size=123, sha256=DIGEST):
        self.size, self.sha256 = size, sha256


class _Entry:
    def __init__(self, lfs):
        self.lfs = lfs


@pytest.fixture
def hub(monkeypatch):
    calls = {}
    state = {"sha": COMMIT, "lfs": _Lfs()}

    def model_info(self, repo_id, *, revision=None):
        calls["model_info"] = (repo_id, revision)
        if isinstance(state["sha"], Exception):
            raise state["sha"]
        return type("Info", (), {"sha": state["sha"]})()

    def get_paths_info(self, repo_id, paths, *, revision=None):
        calls["paths"] = (repo_id, list(paths), revision)
        return [_Entry(state["lfs"])]

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", get_paths_info)
    return calls, state


def test_download_file_passes_the_pinned_revision(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return "/cache/model.gguf"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    path = server.download_file(RTX_4060.repo, RTX_4060.file, revision=RTX_4060.revision)
    assert path == "/cache/model.gguf"
    assert seen == {"repo_id": RTX_4060.repo, "filename": RTX_4060.file, "revision": RTX_4060.revision}


def test_hub_file_returns_commit_size_and_hash_at_that_commit(hub):
    calls, _ = hub
    assert server.hub_file("org/repo", "m.gguf") == HubFile(COMMIT, 123, DIGEST)
    assert calls["model_info"] == ("org/repo", None)
    assert calls["paths"] == ("org/repo", ["m.gguf"], COMMIT)


def test_hub_file_resolves_the_requested_ref(hub):
    calls, _ = hub
    server.hub_file("org/repo", "m.gguf", ref="main")
    assert calls["model_info"] == ("org/repo", "main")


def test_hub_file_none_when_the_hub_call_raises(hub):
    _, state = hub
    state["sha"] = OSError("offline")
    assert server.hub_file("org/repo", "m.gguf") is None


@pytest.mark.parametrize("sha", ["main", "A" * 40, "a" * 39, None])
def test_hub_file_none_when_the_commit_is_not_40_lowercase_hex(hub, sha):
    hub[1]["sha"] = sha
    assert server.hub_file("org/repo", "m.gguf") is None


@pytest.mark.parametrize("lfs", [
    None, _Lfs(size=-1), _Lfs(size="9"), _Lfs(size=True), _Lfs(sha256="b" * 63), _Lfs(sha256="B" * 64), _Lfs(sha256=None),
])
def test_hub_file_none_when_the_file_has_no_valid_lfs_metadata(hub, lfs):
    hub[1]["lfs"] = lfs
    assert server.hub_file("org/repo", "m.gguf") is None


@pytest.mark.parametrize("call", [
    lambda: huggingface_hub.hf_hub_download(repo_id="r", filename="f"),
    lambda: huggingface_hub.HfApi().model_info("r"),
    lambda: huggingface_hub.HfApi().get_paths_info("r", ["f"]),
])
def test_the_suite_refuses_hub_calls_unless_a_test_patches_them(call):
    with pytest.raises(RuntimeError, match="Hugging Face hub"):
        call()
