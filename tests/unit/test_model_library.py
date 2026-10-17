"""The model library's pure core: the scan over roots, the library file, and the moved record builders."""
from __future__ import annotations

import json
import os
import stat
from datetime import date
from pathlib import Path

import pytest

from localharness.provider import gguf_header
from localharness.provider.gguf_header import FileKind, rank_models
from localharness.provider import model_library
from localharness.provider.model_library import Entry, Library, library_path, scan
from tests.unit.gguf_support import Raw, write_gguf
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
        rank=lambda ms: list(ms), checkpoints=()):
    return scan([root], set(recorded), lib, scope=scope, rank=rank, first_run=first_run, today=today,
                checkpoints=checkpoints)


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


# ---- first sighting, never mtime ---------------------------------------------------------------------

def test_a_file_copied_in_with_an_old_mtime_is_new_and_a_known_one_with_a_future_mtime_is_not(roots, tmp_path, reads):
    root, lib = roots
    known = model(root / "Known.gguf")
    assert run(root, lib, first_run=True).library.save(lib)
    source = model(tmp_path / "elsewhere" / "Copied.gguf", embedding=2048)
    os.utime(source, (1_000_000_000, 1_000_000_000))
    shutil_copy2(source, root / "Copied.gguf")
    future = known.stat().st_mtime + 365 * 86400
    os.utime(known, (future, future))
    reads.clear()

    found = run(root, lib)

    assert [s.path.name for s in found.new] == ["Copied.gguf"] and reads == [str(root / "Copied.gguf")]
    assert [s.path.name for s in found.seen] == ["Known.gguf"]


def shutil_copy2(src: Path, dst: Path) -> None:
    import shutil

    shutil.copy2(src, dst)


def saved(root: Path, lib: Path, *, today=TODAY, **kw):
    found = run(root, lib, today=today, **kw)
    assert found.library.save(lib)
    return found


def test_a_resized_known_model_is_retyped_once_and_keeps_its_first_seen(roots, reads):
    root, lib = roots
    a = model(root / "A.gguf")
    saved(root, lib, today=date(2026, 1, 1), first_run=True)
    model(a, embedding=2048, extra_pad="x" * 40)
    reads.clear()

    found = run(root, lib)

    assert found.new == () and reads == [str(a)]
    assert found.library.entries[real(a)].first_seen == "2026-01-01" and found.changed is True


def test_one_byte_of_growth_retypes_a_file(roots, reads):
    root, lib = roots
    a = model(root / "A.gguf")
    saved(root, lib, first_run=True)
    reads.clear()
    with open(a, "ab") as f:
        f.write(b"\0")

    found = run(root, lib)

    assert reads == [str(a)] and found.library.entries[real(a)].size == a.stat().st_size


def test_a_known_other_file_that_grows_into_a_model_is_new_with_its_first_seen_reset(roots):
    root, lib = roots
    path = write_gguf(root / "Late.gguf", {"general.architecture": "llama", "general.type": "adapter"})
    saved(root, lib, today=date(2026, 1, 1), first_run=True)
    assert Library.load(lib).entries[real(path)].verdict is FileKind.OTHER
    model(path)

    found = run(root, lib)

    [shown] = found.new
    assert shown.first_seen == "2026-10-16" and shown.model.name == "Late"


def test_an_unreadable_file_is_retried_every_scan_and_new_once_it_reads_as_a_model(roots, reads):
    root, lib = roots
    path = root / "Partial.gguf"
    path.write_bytes(b"GGUF-not-yet")
    saved(root, lib, first_run=True)
    assert Library.load(lib).entries[real(path)].verdict is FileKind.UNREADABLE
    reads.clear()

    assert run(root, lib).new == () and reads == [str(path)]

    model(path)
    [shown] = run(root, lib).new
    assert shown.path == path


def test_known_other_projector_and_shard_files_are_never_read(roots, no_reads):
    root, lib = roots
    for name, kv in {
        "lora.gguf": {"general.architecture": "llama", "general.type": "adapter"},
        "mmproj-A.gguf": {"general.architecture": "clip", "general.type": "mmproj",
                          "clip.vision.projection_dim": 4096},
        "B-00002-of-00002.gguf": {"general.architecture": "llama", "split.no": Raw(2, b"\x01\x00"),
                                  "split.count": Raw(2, b"\x02\x00")},
    }.items():
        write_gguf(root / name, kv)
    entries = {}
    for name, kind in {"lora.gguf": "other", "mmproj-A.gguf": "projector", "B-00002-of-00002.gguf": "shard"}.items():
        entries[real(root / name)] = {"first_seen": "2026-01-01", "size": (root / name).stat().st_size,
                                      "verdict": kind, "declined": False}
    lib.parent.mkdir()
    lib.write_text(json.dumps({"version": 1, "entries": entries}))

    found = run(root, lib)

    assert found.new == () and found.changed is False


def test_a_known_projector_pairs_with_a_model_that_arrives_in_its_directory(roots, reads):
    root, lib = roots
    proj = write_gguf(root / "mmproj-Vis.gguf", {"general.architecture": "clip", "general.type": "mmproj",
                                                   "clip.vision.projection_dim": 4096})
    saved(root, lib, first_run=True)
    vis = model(root / "Vis-7B.gguf")
    reads.clear()

    [shown] = run(root, lib).new

    assert shown.model.mmproj == proj and sorted(reads) == sorted([str(proj), str(vis)])


# ---- the baseline ------------------------------------------------------------------------------------

def test_a_missing_library_on_a_later_run_baselines_and_asks_nothing(roots):
    root, lib = roots
    a = model(root / "A.gguf")

    found = run(root, lib)

    assert found.new == () and found.baselined is True and found.changed is True
    assert [s.real for s in found.seen] == [real(a)]
    assert found.library.entries[real(a)].verdict is FileKind.MODEL


def test_a_missing_library_on_a_first_run_has_no_baseline(roots):
    root, lib = roots
    model(root / "A.gguf")

    found = run(root, lib, first_run=True)

    assert found.baselined is False and len(found.new) == 1


def corrupt_library(kind: str, lib: Path) -> None:
    lib.parent.mkdir(parents=True, exist_ok=True)
    if kind == "fifo":
        os.mkfifo(lib)
    elif kind == "big":
        lib.write_bytes(b" " * (2 << 20))
    elif kind == "list":
        lib.write_text("[]")
    elif kind == "version":
        lib.write_text('{"version": 2, "entries": {}}')
    elif kind == "bool-version":
        lib.write_text('{"version": true, "entries": {}}')
    elif kind == "many":
        lib.write_text(json.dumps({"version": 1, "entries": {f"/x/{i}.gguf": good_entry() for i in range(257)}}))
    elif kind == "deep":
        lib.write_text("[" * 100_000)
    elif kind == "empty":
        lib.write_text("")
    else:
        lib.write_text(kind)


def good_entry(**kw) -> dict:
    return {"first_seen": "2026-01-01", "size": 10, "verdict": "model", "declined": False, **kw}


@pytest.mark.parametrize("kind", ["fifo", "big", "list", "version", "bool-version", "many", "deep", "empty",
                                  "{not json", '{"version": 1}', '{"version": 1, "entries": []}'])
def test_a_corrupt_library_baselines_like_a_missing_one(roots, kind):
    root, lib = roots
    a = model(root / "A.gguf")
    corrupt_library(kind, lib)

    loaded = Library.load(lib)
    found = run(root, lib)

    assert loaded.corrupt is True and loaded.entries == {}
    assert found.new == () and found.baselined is True and found.changed is True
    assert found.library.entries[real(a)].verdict is FileKind.MODEL


def test_a_library_that_cannot_be_read_is_not_rebaselined_and_is_never_overwritten(roots, monkeypatch):
    root, lib = roots
    a = model(root / "A.gguf")
    saved(root, lib, first_run=True)
    run(root, lib).library.marked(declined=[real(a)]).save(lib)
    before = lib.read_bytes()

    def refuse(path):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(model_library, "open_regular", refuse)
    loaded = Library.load(lib)
    found = run(root, lib)

    assert loaded.unreadable is True and loaded.corrupt is False and loaded.entries == {}
    assert found.new == found.seen == found.declined == () and found.baselined is False and found.changed is False
    assert found.library.marked(declined=[real(a)]).save(lib) is False
    assert lib.read_bytes() == before


def test_a_merged_save_keeps_a_decline_and_an_entry_another_process_wrote_in_between(roots):
    root, lib = roots
    a, b = model(root / "A.gguf"), model(root / "B.gguf")
    saved(root, lib, first_run=True)
    snapshot = run(root, lib)
    other = Library.load(lib)
    c = model(root / "C.gguf")
    other_entries = {**other.entries, real(a): Entry("2026-10-16", other.entries[real(a)].size, FileKind.MODEL, True),
                     real(c): Entry("2026-10-17", c.stat().st_size, FileKind.MODEL)}
    assert Library(other_entries).save(lib)

    assert snapshot.library.marked(declined=[real(b)], dropped=[]).save(lib, merge=True)

    result = Library.load(lib).entries
    assert result[real(a)].declined is True and result[real(b)].declined is True and real(c) in result


def test_a_merged_save_removes_what_the_snapshot_dropped_or_pruned_and_refuses_an_unreadable_file(roots, monkeypatch):
    root, lib = roots
    a, b = model(root / "A.gguf"), model(root / "B.gguf")
    saved(root, lib, first_run=True)
    pruned = Library.load(lib).unrecorded({real(a)})

    assert pruned.marked(dropped=[real(b)]).save(lib, merge=True)
    assert Library.load(lib).entries == {}

    saved(root, lib, first_run=True)
    monkeypatch.setattr(model_library, "open_regular", lambda path: (_ for _ in ()).throw(PermissionError(13, "x")))
    assert Library.load(lib).unreadable is True
    assert Library({}).save(lib, merge=True) is False


def test_a_removed_entry_makes_a_known_file_new_again(roots):
    root, lib = roots
    a = model(root / "A.gguf")
    b = model(root / "B.gguf", embedding=2048)
    saved(root, lib, first_run=True)
    data = json.loads(lib.read_text())
    del data["entries"][real(a)]
    lib.write_text(json.dumps(data))

    assert [s.path.name for s in run(root, lib).new] == ["A.gguf"]


# ---- the library file --------------------------------------------------------------------------------

def test_the_library_holds_exactly_the_file_cap_and_exactly_the_entry_cap(tmp_path):
    lib = tmp_path / "lib.json"
    body = json.dumps({"version": 1, "entries": {f"/x/{i}.gguf": good_entry() for i in range(256)}})
    lib.write_text(body)
    assert len(Library.load(lib).entries) == 256

    lib.write_text(body + " " * ((1 << 20) - len(body)))
    assert lib.stat().st_size == model_library.LIBRARY_CAP and len(Library.load(lib).entries) == 256

    lib.write_text(body + " " * ((1 << 20) - len(body) + 1))
    assert Library.load(lib).corrupt is True


@pytest.mark.parametrize("bad", [
    {"first_seen": "2026-13-45"}, {"first_seen": "20261016"}, {"first_seen": "2026-1-1"}, {"first_seen": 20261016},
    {"first_seen": "2026-01-01T00:00:00"}, {"size": -1}, {"size": 1.5}, {"size": True}, {"size": "10"},
    {"verdict": "weights"}, {"verdict": 3}, {"verdict": ["model"]}, {"declined": "yes"}, {"declined": 1},
    {"repo": "org/a/b"}, {"repo": "../x"}, {"repo": "a" * 200}, {"repo": 3}, {"repo": ""},
])
def test_an_invalid_entry_is_dropped_alone(tmp_path, bad):
    lib = tmp_path / "lib.json"
    lib.write_text(json.dumps({"version": 1, "entries": {"/x/good.gguf": good_entry(), "/x/bad.gguf": good_entry(**bad)}}))

    loaded = Library.load(lib)

    assert list(loaded.entries) == ["/x/good.gguf"] and loaded.corrupt is False


def test_a_relative_key_and_a_non_object_entry_are_dropped_alone(tmp_path):
    lib = tmp_path / "lib.json"
    lib.write_text(json.dumps({"version": 1, "entries": {
        "/x/good.gguf": good_entry(), "rel/bad.gguf": good_entry(), "/x/list.gguf": [1]}}))

    assert list(Library.load(lib).entries) == ["/x/good.gguf"]


def test_save_writes_sorted_private_json_and_leaves_out_files_that_are_gone(tmp_path):
    here = tmp_path / "here.gguf"
    here.write_bytes(b"x")
    lib = tmp_path / "cfg" / "model-library.json"
    entries = {str(here): Entry("2026-01-01", 1, FileKind.MODEL), str(tmp_path / "gone.gguf"): Entry("2026-01-01", 1, FileKind.OTHER)}

    assert Library(entries).save(lib) is True

    assert list(json.loads(lib.read_text())["entries"]) == [str(here)]
    assert stat.S_IMODE(lib.stat().st_mode) == 0o600


def test_save_writes_nothing_when_the_overlay_lock_is_busy(tmp_path, monkeypatch):
    import fcntl

    from localharness.config import overlay

    monkeypatch.setattr(overlay, "OVERLAY_LOCK_BUDGET_SECONDS", 0.05)
    fd = os.open(tmp_path / "overrides.yaml.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert Library({}).save(tmp_path / "model-library.json") is False
    finally:
        os.close(fd)
    assert not (tmp_path / "model-library.json").exists()


def test_save_returns_false_when_the_write_itself_fails(tmp_path, monkeypatch):
    def fail(path, text):
        raise OSError(28, "no space")

    monkeypatch.setattr(model_library, "atomic_write_text", fail)

    assert Library({}).save(tmp_path / "model-library.json") is False


def test_a_saved_library_loads_back_equal(roots):
    root, lib = roots
    model(root / "A.gguf")
    found = saved(root, lib, first_run=True)

    assert Library.load(lib).entries == found.library.entries


# ---- recorded files, declines, what changed ----------------------------------------------------------

def test_a_recorded_file_is_left_out_of_the_found_library_and_new_again_once_unrecorded(roots):
    root, lib = roots
    a = model(root / "A.gguf")
    b = model(root / "B.gguf", embedding=2048)
    first = saved(root, lib, first_run=True)
    assert first.library.marked(declined=[real(b)]).save(lib)

    found = run(root, lib, recorded={real(a)})

    assert real(a) not in found.library.entries and found.library.entries[real(b)].declined is True
    assert found.changed is True and found.new == ()
    assert [s.path.name for s in found.declined] == ["B.gguf"]
    assert found.library.save(lib)

    assert [s.path.name for s in run(root, lib).new] == ["A.gguf"]


def test_unrecorded_leaves_out_the_named_realpaths_and_keeps_the_rest(roots):
    root, lib = roots
    a = model(root / "A.gguf")
    b = model(root / "B.gguf", embedding=2048)
    library = saved(root, lib, first_run=True).library

    kept = library.unrecorded({real(b)})

    assert set(kept.entries) == {real(a)} and set(library.entries) == {real(a), real(b)}


def test_marked_declines_and_drops_by_realpath(roots):
    root, lib = roots
    a = model(root / "A.gguf")
    b = model(root / "B.gguf", embedding=2048)
    library = saved(root, lib, first_run=True).library

    marked = library.marked(declined=[real(a)], dropped=[real(b)])

    assert marked.entries[real(a)].declined is True and real(b) not in marked.entries
    assert library.entries[real(a)].declined is False


def test_a_dropped_entry_is_a_removed_peer_and_new_again(roots):
    root, lib = roots
    a = model(root / "A.gguf")
    library = saved(root, lib, first_run=True).library
    assert library.marked(dropped=[real(a)]).save(lib)

    assert [s.path.name for s in run(root, lib).new] == ["A.gguf"]


def test_changed_follows_what_the_saved_file_would_differ_by(roots):
    root, lib = roots
    a = model(root / "A.gguf")

    assert run(root, lib).changed is True  # a missing file is a change: saving makes the next scan no baseline
    saved(root, lib)
    assert run(root, lib).changed is False

    write_gguf(root / "lora.gguf", {"general.architecture": "llama", "general.type": "adapter"})
    added = run(root, lib)
    assert added.new == () and added.changed is True and real(root / "lora.gguf") in added.library.entries
    assert added.library.save(lib)
    assert run(root, lib).changed is False

    with open(a, "ab") as f:
        f.write(b"\0")
    assert run(root, lib).changed is True


def test_changed_is_true_for_a_missing_library_over_an_empty_root(roots):
    root, lib = roots
    assert run(root, lib).changed is True


def test_scope_all_types_known_files_fresh_and_scope_new_reads_nothing_for_them(roots, reads):
    root, lib = roots
    a = model(root / "A.gguf")
    b = model(root / "B.gguf", embedding=2048)
    first = saved(root, lib, first_run=True)
    assert first.library.marked(declined=[real(b)]).save(lib)
    reads.clear()

    narrow = run(root, lib, scope="new")
    assert reads == [] and [s.model for s in narrow.seen + narrow.declined] == [None, None]

    wide = run(root, lib, scope="all")
    assert sorted(reads) == sorted([str(a), str(b)])
    assert [s.model.name for s in wide.seen] == ["A"] and [s.model.name for s in wide.declined] == ["B"]
    assert wide.changed is False


# ---- the walk's edges --------------------------------------------------------------------------------

def test_two_paths_to_one_realpath_are_one_entry_the_first_in_walk_order(roots):
    root, lib = roots
    a = model(root / "A.gguf")
    (root / "B-link.gguf").symlink_to(a)

    found = run(root, lib, first_run=True)

    assert [s.path for s in found.new] == [a] and list(found.library.entries) == [real(a)]


def test_new_comes_out_in_the_order_rank_returns(roots):
    root, lib = roots
    for name in "ABC":
        model(root / f"{name}.gguf", embedding=1000 + ord(name))

    found = run(root, lib, first_run=True, rank=lambda ms: sorted(ms, key=lambda m: m.name, reverse=True))

    assert [s.path.name for s in found.new] == ["C.gguf", "B.gguf", "A.gguf"]


def test_equal_rank_files_come_out_in_path_order_on_every_run(roots):
    root, lib = roots
    for name in ("m3", "m1", "m2"):
        os.utime(model(root / f"{name}.gguf", embedding=len(name) * 100 + ord(name[1])), (5_000, 5_000))

    def rank(ms):
        return rank_models(ms, unit_model=None, ledger_paths=(), fits={})

    orders = {tuple(s.path.name for s in run(root, lib, first_run=True, rank=rank).new) for _ in range(3)}

    assert orders == {("m1.gguf", "m2.gguf", "m3.gguf")}


def test_first_seen_defaults_to_the_utc_date_the_scan_ran(roots):
    root, lib = roots
    model(root / "A.gguf")

    [shown] = scan([root], set(), lib, scope="new", rank=lambda ms: list(ms), first_run=True).new

    assert date.fromisoformat(shown.first_seen) and len(shown.first_seen) == 10


def test_a_peer_record_carries_only_what_launches_it(tmp_path):
    proj = write_gguf(tmp_path / "mmproj-Peer.gguf", {"general.architecture": "clip", "general.type": "mmproj",
                                                        "clip.vision.projection_dim": 4096})
    found = gguf_header.discover([model(tmp_path / "Peer.gguf"), proj]).models

    [peer] = model_library.peers_for(found, start=9100, taken_names=(), taken_ports=())

    life = peer.lifecycle
    assert (life.runtime, life.mmproj, life.served_name, life.port, life.gpu) == (
        "llamacpp", str(proj), "Peer", 9100, True)
    assert not (life.binary or life.extra_args or life.env or life.config_ini or life.api_key_file)
    assert peer.provider_type == "llamacpp" and peer.gpu is True


# ---- one owner of the file, no cli/ import -----------------------------------------------------------

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"


def sources():
    for path in sorted(SRC.rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), path.read_text(encoding="utf-8")


def writes_the_library(source: str) -> bool:
    import ast

    def names(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                yield n.id
            elif isinstance(n, ast.Attribute):
                yield n.attr

    return any(isinstance(n, ast.Call) and "atomic_write_text" in set(names(n.func))
               and any({"LIBRARY_FILE", "library_path"} & set(names(arg)) for arg in [*n.args, *(k.value for k in n.keywords)])
               for n in ast.walk(ast.parse(source)))


def checkpoint(tmp_path: Path, repo_id: str, size: int = 10) -> model_library.Checkpoint:
    snap = tmp_path / "snaps" / repo_id.replace("/", "--")
    snap.mkdir(parents=True, exist_ok=True)
    return model_library.Checkpoint(repo_id, snap, real(snap), size)


def test_checkpoints_are_first_seen_after_the_ggufs_in_repo_id_order_keyed_by_the_snapshot(roots, tmp_path):
    root, lib = roots
    model(root / "A.gguf")
    b, a = checkpoint(tmp_path, "org/B"), checkpoint(tmp_path, "org/A")

    found = saved(root, lib, first_run=True, checkpoints=[b, a])

    assert [s.checkpoint.repo_id if s.checkpoint else s.path.name for s in found.new] == ["A.gguf", "org/A", "org/B"]
    assert found.new[1].model is None and found.new[1].real == a.real and found.new[1].size == 10
    assert found.library.entries[a.real] == Entry("2026-10-16", 10, FileKind.MODEL, False, "org/A")
    assert json.loads(lib.read_text(encoding="utf-8"))["entries"][a.real]["repo"] == "org/A"
    assert Library.load(lib).entries[b.real].repo == "org/B"


def test_a_checkpoint_a_record_names_by_repo_id_or_snapshot_is_never_listed(roots, tmp_path):
    root, lib = roots
    a, b = checkpoint(tmp_path, "org/A"), checkpoint(tmp_path, "org/B")

    assert [s.checkpoint.repo_id for s in run(root, lib, first_run=True, recorded={"org/A"}, checkpoints=[a, b]).new] \
        == ["org/B"]
    assert [s.checkpoint.repo_id for s in run(root, lib, first_run=True, recorded={b.real}, checkpoints=[a, b]).new] \
        == ["org/A"]


def test_unrecorded_leaves_out_an_entry_whose_repo_a_record_names(roots, tmp_path):
    root, lib = roots
    a, b = checkpoint(tmp_path, "org/A"), checkpoint(tmp_path, "org/B")
    library = saved(root, lib, first_run=True, checkpoints=[a, b]).library

    assert set(library.unrecorded({"org/A"}).entries) == {b.real}
    assert set(library.unrecorded({b.real}).entries) == {a.real}


def test_a_seen_checkpoint_is_not_new_again_and_a_baselined_one_is_never_new(roots, tmp_path):
    root, lib = roots
    a = checkpoint(tmp_path, "org/A")

    assert run(root, lib, checkpoints=[a]).new == ()
    saved(root, lib, first_run=True, checkpoints=[a])

    assert run(root, lib, checkpoints=[a]).new == ()
    assert [s.checkpoint.repo_id for s in run(root, lib, scope="all", checkpoints=[a]).seen] == ["org/A"]


def test_only_the_library_module_and_the_gate_name_the_library_file():
    named = {rel for rel, text in sources() if "model-library.json" in text}
    assert named == {"provider/model_library.py", "agent/gate_types.py"}


def test_the_library_module_imports_nothing_from_cli():
    import ast

    tree = ast.parse((SRC / "provider" / "model_library.py").read_text(encoding="utf-8"))
    modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in modules if m and m.startswith("localharness.cli")]


def test_no_other_module_writes_the_library_file():
    assert [rel for rel, text in sources() if rel != "provider/model_library.py" and writes_the_library(text)] == []


@pytest.mark.parametrize("source", [
    "atomic_write_text(LIBRARY_FILE, t)",
    "private_files.atomic_write_text(model_library.library_path(d), t)",
    "atomic_write_text(path=library_path(d), text=t)",
])
def test_the_writer_pin_sees_a_second_writer(source):
    assert writes_the_library(source) is True
    assert writes_the_library("atomic_write_text(other_path, t)") is False
