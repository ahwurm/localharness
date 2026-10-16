"""`localharness model --download` records one `.gguf` only when the hub vouched for its bytes at a commit the hub named,
with a config and a person at the terminal; otherwise it writes nothing and says the next start finds it.
The hub and the download are stubs; the file is a real tiny GGUF."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli import library, workspace
from localharness.cli.app import app
from localharness.config import trust
from localharness.provider import server as managed_server
from localharness.provider.server import HubFile
from tests.unit import test_library_later_runs as runs
from tests.unit.test_init_core_setup import _hermetic  # noqa: F401  (autouse fixture)
from tests.unit.test_init_discovery_records import _model, _peers

box = runs.box
later = runs.later
COMMIT = "a" * 40
FAKE = "b" * 40


@pytest.fixture
def hub(later, monkeypatch, tmp_path):
    """The hub names COMMIT and the real size and hash of hf/m.gguf; download_file records its revision and returns that file."""
    path = _model(tmp_path / "hf" / "m.gguf")
    data = path.read_bytes()
    calls: list[dict] = []

    def download_file(repo, file, *, revision=None):
        calls.append({"repo": repo, "file": file, "revision": revision})
        return str(path)

    monkeypatch.setattr(managed_server, "hub_file", lambda *_a, **_k: HubFile(COMMIT, len(data), hashlib.sha256(data).hexdigest()))
    monkeypatch.setattr(managed_server, "download_file", download_file)
    later.file, later.calls, later.sha, later.size = path, calls, hashlib.sha256(data).hexdigest(), len(data)
    return later


def _download(box, *flags: str, config_dir: Path | None = None):
    args = ["model", "--download", "org/repo", *flags, "--config-dir", str(config_dir or box.cfg)]
    return CliRunner().invoke(app, args)


def _names(box) -> list[str]:
    return [p["name"] for p in _peers(box.cfg)]


def test_a_verified_gguf_is_recorded_as_a_peer_and_adopted(hub):
    result = _download(hub, "--file", "m.gguf")

    assert result.exit_code == 0, result.output
    assert hub.calls == [{"repo": "org/repo", "file": "m.gguf", "revision": COMMIT}]
    assert f"Verified m.gguf at {COMMIT} (sha256 {hub.sha})" in result.output
    assert library.record_line("m", "llamacpp", 8083, hub.bin, hub.file) in hub.printed.getvalue()
    assert _names(hub) == ["Swift-27B-Q8", "m"]
    entries = {e["name"]: e["shown"] for e in trust.recorded_machine(hub.cfg)["entries"]}
    assert entries["extra_endpoints.m.lifecycle.model"] == str(hub.file)


def test_a_planted_snapshot_directory_is_never_the_commit_or_the_hash(hub, monkeypatch, tmp_path):
    planted = _model(tmp_path / "cache" / "snapshots" / FAKE / "m.gguf", embedding=4096)
    planted.write_bytes(planted.read_bytes() + b"\0" * (hub.size - planted.stat().st_size))
    monkeypatch.setattr(managed_server, "download_file", lambda *_a, **_k: str(planted))

    result = _download(hub, "--file", "m.gguf")

    assert result.exit_code == 0 and "Not recorded (sha256 " in result.output and "differs" in result.output
    assert f"at {FAKE}" not in result.output and "Verified" not in result.output
    assert _names(hub) == ["Swift-27B-Q8"]


def test_a_file_of_another_size_is_named(hub, monkeypatch, tmp_path):
    short = tmp_path / "hf" / "short.gguf"
    short.write_bytes(hub.file.read_bytes()[:-1])
    monkeypatch.setattr(managed_server, "download_file", lambda *_a, **_k: str(short))

    result = _download(hub, "--file", "m.gguf")

    assert f"Not recorded (size is {hub.size - 1} bytes, expected {hub.size}) — the next localharness start finds it" in result.output
    assert _names(hub) == ["Swift-27B-Q8"]


def test_a_fifo_at_the_returned_path_is_refused(hub, monkeypatch, tmp_path):
    fifo = tmp_path / "hf" / "pipe.gguf"
    os.mkfifo(fifo)
    monkeypatch.setattr(managed_server, "download_file", lambda *_a, **_k: str(fifo))

    result = _download(hub, "--file", "m.gguf")

    assert "Not recorded (not a regular file)" in result.output and _names(hub) == ["Swift-27B-Q8"]


def test_without_a_config_nothing_is_recorded_and_start_will_find_it(hub, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _download(hub, "--file", "m.gguf", config_dir=empty)

    assert result.exit_code == 0
    assert "Not recorded (no config yet) — the next localharness start finds it" in result.output
    assert not (empty / "overrides.yaml").exists()


def test_without_a_person_nothing_is_recorded(hub, monkeypatch):
    monkeypatch.setattr(workspace, "_stdout_is_a_terminal", lambda: False)

    result = _download(hub, "--file", "m.gguf")

    assert result.exit_code == 0 and "Not recorded (no person at this terminal)" in result.output
    assert _names(hub) == ["Swift-27B-Q8"]


def test_without_hub_metadata_the_download_is_unpinned_and_nothing_is_recorded(hub, monkeypatch):
    monkeypatch.setattr(managed_server, "hub_file", lambda *_a, **_k: None)

    result = _download(hub, "--file", "m.gguf")

    assert hub.calls == [{"repo": "org/repo", "file": "m.gguf", "revision": None}]
    assert "Not recorded (the hub gave no file metadata)" in result.output and _names(hub) == ["Swift-27B-Q8"]


def test_a_file_already_recorded_adds_no_second_preset(hub):
    assert _download(hub, "--file", "m.gguf").exit_code == 0

    again = _download(hub, "--file", "m.gguf")

    assert "Not recorded (no preset was added)" in again.output and _names(hub) == ["Swift-27B-Q8", "m"]


def test_the_whole_repo_download_and_a_file_that_is_not_a_gguf_record_nothing(hub, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("recorded")

    monkeypatch.setattr(library, "record_path", boom)
    monkeypatch.setattr(managed_server, "hub_file", boom)
    monkeypatch.setattr(managed_server, "download_model", lambda repo: f"/cache/{repo}")
    monkeypatch.setattr(managed_server, "download_file", lambda repo, file: f"/cache/{file}")

    whole = _download(hub)
    other = _download(hub, "--file", "model.safetensors")

    assert whole.exit_code == 0 and "Downloaded to /cache/org/repo" in whole.output and "To use it:" in whole.output
    assert other.exit_code == 0 and "Downloaded to /cache/model.safetensors" in other.output
    assert _names(hub) == ["Swift-27B-Q8"]


def test_an_interrupted_download_records_nothing_and_exits_2_with_the_error_line(hub, monkeypatch):
    def interrupted(*_a, **_k):
        raise OSError("connection reset")

    monkeypatch.setattr(managed_server, "download_file", interrupted)

    result = _download(hub, "--file", "m.gguf")

    assert result.exit_code == 2 and "download failed: connection reset" in result.output
    assert _names(hub) == ["Swift-27B-Q8"]
