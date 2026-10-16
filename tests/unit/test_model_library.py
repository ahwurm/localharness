"""The model library's pure core: the scan over roots, the library file, and the moved record builders."""
from __future__ import annotations

import json
import os
import stat
from datetime import date
from pathlib import Path

import pytest

from localharness.agent import gguf_header
from localharness.agent.gguf_header import FileKind, rank_models
from localharness.provider import model_library
from localharness.provider.model_library import Entry, Library, library_path, scan
from tests.unit.gguf_support import write_gguf
from tests.unit.procfs_support import fake_proc  # noqa: F401  (fixture)

TODAY = date(2026, 10, 16)


def model(path: Path, *, mtp: int | None = 1, embedding: int = 4096, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    kv = {"general.architecture": "llama", "general.type": "model", "llama.embedding_length": embedding,
          "llama.context_length": 32768, "llama.block_count": 65, **extra}
    if mtp is not None:
        kv["llama.nextn_predict_layers"] = mtp
    return write_gguf(path, kv)


def run(root: Path, lib: Path, *, recorded=(), scope="new", first_run=False, today=TODAY,
        rank=lambda ms: list(ms)):
    return scan([root], set(recorded), lib, scope=scope, rank=rank, first_run=first_run, today=today)


def real(path: Path) -> str:
    return os.path.realpath(path)


@pytest.fixture
def reads(monkeypatch):
    seen: list[str] = []
    original = gguf_header.read_header

    def counting(path, want):
        seen.append(os.fspath(path))
        return original(path, want)

    monkeypatch.setattr(gguf_header, "read_header", counting)
    return seen


@pytest.fixture
def no_reads(monkeypatch):
    def refuse(path, want):
        raise AssertionError(f"read {path}")

    monkeypatch.setattr(gguf_header, "read_header", refuse)


@pytest.fixture
def roots(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    return root, library_path(tmp_path / "cfg")


def test_a_new_model_is_typed_once_shown_saved_under_the_lock_and_remembered(roots, reads):
    root, lib = roots
    a = model(root / "A.gguf")

    first = run(root, lib, first_run=True)

    [shown] = first.new
    assert shown.model.mtp == 1 and shown.first_seen == "2026-10-16" and shown.declined is False
    assert shown.size == a.stat().st_size and shown.real == real(a)
    assert first.library.entries[real(a)] == Entry("2026-10-16", a.stat().st_size, FileKind.MODEL, False)
    assert first.changed is True and reads == [str(a)]

    assert first.library.save(lib) is True
    assert json.loads(lib.read_text())["version"] == 1
    assert stat.S_IMODE(lib.stat().st_mode) == 0o600
    assert stat.S_IMODE((lib.parent / "overrides.yaml.lock").stat().st_mode) == 0o600


def test_the_next_scan_reads_no_header_and_finds_nothing_new(roots, reads, monkeypatch):
    root, lib = roots
    a = model(root / "A.gguf")
    assert run(root, lib, first_run=True).library.save(lib)
    reads.clear()
    monkeypatch.setattr(gguf_header, "read_header", lambda *a, **k: pytest.fail("a header was read"))

    again = run(root, lib)

    assert again.new == () and [s.real for s in again.seen] == [real(a)]
    assert again.changed is False and again.baselined is False


def test_a_peer_record_builder_names_a_model_once_and_skips_listening_ports(fake_proc, tmp_path):  # noqa: F811
    found = gguf_header.discover([model(tmp_path / "Peer-7B.gguf")]).models
    fake_proc.listen(9002, inode=5, uid=os.getuid())

    [peer] = model_library.peers_for(found, start=9001, taken_names={"Peer-7B"}, taken_ports={9001})

    assert (peer.name, peer.base_url) == ("Peer-7B-2", "http://127.0.0.1:9003/v1")
    assert peer.lifecycle.model == str(tmp_path / "Peer-7B.gguf") and peer.lifecycle.port == 9003


def test_free_ports_skip_the_ports_a_record_uses_and_the_ones_listening(fake_proc):  # noqa: F811
    fake_proc.listen(9002, inode=5, uid=os.getuid())

    assert model_library.free_ports(9001, 3, taken={9001, 9004}) == [9003, 9005, 9006]
    assert model_library.free_ports(9001, 0, taken=set()) == []
    assert model_library.free_ports(65534, 5, taken=set()) == [65534, 65535]


def test_nothing_under_the_roots_is_nothing_found_and_rank_is_never_asked(tmp_path):
    def rank(models):
        raise AssertionError("ranked")

    found = scan([tmp_path / "missing", tmp_path], set(), library_path(tmp_path), scope="all", rank=rank)

    assert (found.new, found.seen, found.declined) == ((), (), ())
