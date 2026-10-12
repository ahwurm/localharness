"""config/project_records.py — the per-root instructions store.

Properties held down here: the record lives in the GLOBAL config dir at 0600, keyed by the
resolved root; a damaged store reads as "no record", never an exception; every access runs under
a sidecar lock that is a different inode from the data file; and the module never opens an
instructions file for writing.
"""
# ruff: noqa: F811  (instructions_home is imported as a fixture and requested by name)
from __future__ import annotations

import ast
import hashlib
import logging
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from localharness.config import project_records as pr
from localharness.config.project_records import (
    InstructionsRecord,
    StoreUnavailable,
    read_instructions,
    write_instructions,
)
from tests.unit.instructions_support import (
    instructions_home,  # noqa: F401
    load_store,
    make_root,
    store_file,
)
from tests.unit.test_import_direction import classify


def _record(root: Path, **kw) -> InstructionsRecord:
    real = (Path(root) / "CLAUDE.md").resolve()
    base: dict = {"path": str(real), "name": "CLAUDE.md", "digest": "ab" * 32, "size": 12, "state": "yes"}
    base.update(kw)
    return InstructionsRecord(**base)


# --------------------------------------------------------------------------- shape


def test_the_module_is_core_and_names_its_files():
    assert classify("config/project_records.py") == "core"
    assert pr.PROJECT_RECORDS_FILE == "project_records.yaml"
    assert pr.PROJECT_RECORDS_LOCK_FILE == "project_records.yaml.lock"


def test_paths_resolve_at_call_time(instructions_home, tmp_path, monkeypatch):
    assert pr.store_path() == instructions_home / "project_records.yaml"
    assert pr.lock_path() == instructions_home / "project_records.yaml.lock"
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("LOCALHARNESS_HOME", str(elsewhere))
    assert pr.store_path() == elsewhere / "project_records.yaml"
    assert pr.store_path(tmp_path / "explicit") == tmp_path / "explicit" / "project_records.yaml"


def test_a_record_round_trips_at_0600_keyed_by_the_resolved_root(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "hello"})
    record = _record(root)
    write_instructions(root, record)

    assert read_instructions(root) == record
    assert stat.S_IMODE(store_file().stat().st_mode) == 0o600
    data = load_store()
    assert data["version"] == 1
    assert list(data["roots"]) == [str(root.resolve())]
    assert data["roots"][str(root.resolve())]["instructions"] == {
        "path": record.path, "name": "CLAUDE.md", "digest": "ab" * 32, "size": 12, "state": "yes",
    }


def test_optional_fields_round_trip(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "hello"})
    shadow = pr.ShadowRecord(name="CLAUDE.md", path=str(root / "CLAUDE.md"), digest="cd" * 32)
    record = _record(
        root, harness_written="ef" * 32, pending=(("s1", "2026-10-11T10:00:00+00:00"),
                                                  ("s2", "2026-10-11T10:01:00+00:00")),
        shadow=shadow,
    )
    write_instructions(root, record)
    assert read_instructions(root) == record
    raw = load_store()["roots"][str(root.resolve())]["instructions"]
    assert raw["pending"] == {"s1": "2026-10-11T10:00:00+00:00", "s2": "2026-10-11T10:01:00+00:00"}
    assert raw["shadow"] == {"name": "CLAUDE.md", "path": str(root / "CLAUDE.md"), "digest": "cd" * 32}


def test_a_symlinked_checkout_and_its_real_path_share_one_entry(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    link = tmp_path / "link"
    link.symlink_to(root)
    write_instructions(link, _record(root))
    assert read_instructions(root) == _record(root)
    assert list(load_store()["roots"]) == [str(root.resolve())]


def test_a_repointed_root_symlink_is_a_new_root(instructions_home, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    link = tmp_path / "L"
    link.symlink_to(a)
    write_instructions(link, _record(a, name="A.md"))
    link.unlink()
    link.symlink_to(b)
    assert read_instructions(link) is None
    write_instructions(link, _record(b, name="B.md"))
    link.unlink()
    link.symlink_to(a)
    assert read_instructions(link).name == "A.md"
    link.unlink()
    link.symlink_to(b)
    assert read_instructions(link).name == "B.md"


@pytest.mark.parametrize("content", [
    "[1, 2]", "just text", "key: [unclosed", "", "roots: 5", "roots: {x: 5}", "- a\n- b\n",
])
def test_a_damaged_store_reads_as_none_and_never_raises(instructions_home, tmp_path, content):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    store_file().write_text(content, encoding="utf-8")
    assert read_instructions(root) is None


def test_a_binary_store_reads_as_none(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    store_file().write_bytes(b"\xff\xfe\x00bad")
    assert read_instructions(root) is None


@pytest.mark.parametrize("patch", [
    {"state": "maybe"}, {"size": "12"}, {"size": True}, {"digest": 5}, {"path": None},
])
def test_a_wrong_typed_record_reads_as_none(instructions_home, tmp_path, patch):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    write_instructions(root, _record(root))
    data = load_store()
    key = str(root.resolve())
    data["roots"][key]["instructions"].update(patch)
    store_file().write_text(yaml.safe_dump(data), encoding="utf-8")
    assert read_instructions(root) is None


def test_a_missing_required_field_reads_as_none(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    write_instructions(root, _record(root))
    data = load_store()
    del data["roots"][str(root.resolve())]["instructions"]["name"]
    store_file().write_text(yaml.safe_dump(data), encoding="utf-8")
    assert read_instructions(root) is None


def test_wrong_typed_optional_fields_are_dropped(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    write_instructions(root, _record(root))
    data = load_store()
    raw = data["roots"][str(root.resolve())]["instructions"]
    raw.update({"harness_written": 5, "pending": ["x"], "shadow": {"name": "n"}})
    store_file().write_text(yaml.safe_dump(data), encoding="utf-8")
    assert read_instructions(root) == _record(root)


def test_a_wrong_typed_root_entry_is_replaced_on_write(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    store_file().write_text(yaml.safe_dump({"version": 1, "roots": {str(root.resolve()): 7}}),
                            encoding="utf-8")
    assert read_instructions(root) is None
    write_instructions(root, _record(root))
    assert read_instructions(root) == _record(root)


def test_another_roots_entry_survives_a_write(instructions_home, tmp_path):
    one = make_root(tmp_path / "one", {"CLAUDE.md": "1"})
    two = make_root(tmp_path / "two", {"CLAUDE.md": "2"})
    write_instructions(one, _record(one))
    write_instructions(two, _record(two, state="off"))
    assert read_instructions(one).state == "yes"
    assert read_instructions(two).state == "off"


# --------------------------------------------------------------------------- lock + inode


def test_the_sidecar_lock_exists_beside_the_store_at_0600(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    write_instructions(root, _record(root))
    lock = instructions_home / "project_records.yaml.lock"
    assert lock.exists()
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600


def test_a_write_replaces_the_data_inode_but_not_the_locks(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    write_instructions(root, _record(root))
    data_before = os.stat(store_file()).st_ino
    lock_before = os.stat(pr.lock_path()).st_ino
    write_instructions(root, _record(root, size=99))
    assert os.stat(store_file()).st_ino != data_before
    assert os.stat(pr.lock_path()).st_ino == lock_before


def test_a_read_leaves_the_store_untouched(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    assert read_instructions(root) is None
    assert not store_file().exists()
    write_instructions(root, _record(root))
    before = os.stat(store_file())
    assert read_instructions(root) is not None
    after = os.stat(store_file())
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_a_symlinked_sidecar_is_refused_and_its_target_left_alone(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    target = tmp_path / "victim.txt"
    target.write_bytes(b"precious")
    pr.lock_path().symlink_to(target)
    with pytest.raises(StoreUnavailable):
        write_instructions(root, _record(root))
    assert target.read_bytes() == b"precious"
    assert not store_file().exists()


def test_store_unavailable_is_an_oserror():
    assert issubclass(StoreUnavailable, OSError)


# --------------------------------------------------------------------------- candidates

_NO_FIFO = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")


def test_the_candidates_are_tried_in_order(instructions_home, tmp_path):
    root = make_root(tmp_path, {
        "CLAUDE.md": "a", "AGENTS.md": "b", ".localharness/INSTRUCTIONS.md": "c",
    })
    assert pr.first_candidate(root).name == "CLAUDE.md"
    (root / "CLAUDE.md").unlink()
    assert pr.first_candidate(root).name == "AGENTS.md"
    (root / "AGENTS.md").unlink()
    found = pr.first_candidate(root)
    assert found.name == ".localharness/INSTRUCTIONS.md"
    assert found.path == root / ".localharness" / "INSTRUCTIONS.md"
    assert found.real == (root / ".localharness" / "INSTRUCTIONS.md").resolve()


def test_instruction_candidates_are_the_three_lexical_paths(tmp_path):
    assert pr.instruction_candidates(tmp_path) == (
        tmp_path / "CLAUDE.md", tmp_path / "AGENTS.md", tmp_path / ".localharness" / "INSTRUCTIONS.md",
    )


def test_no_candidate_is_none(instructions_home, tmp_path):
    assert pr.first_candidate(make_root(tmp_path, {"README.md": "x"})) is None


def test_a_link_out_of_the_root_is_skipped_silently(instructions_home, tmp_path, capsys, caplog):
    outside = tmp_path / "outside.md"
    outside.write_text("secret")
    root = make_root(tmp_path, {"AGENTS.md": "regular"})
    (root / "CLAUDE.md").symlink_to(outside)
    with caplog.at_level(logging.WARNING):
        found = pr.first_candidate(root)
    assert found.name == "AGENTS.md"
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert caplog.records == []


def test_a_link_inside_the_root_keeps_its_real_path(instructions_home, tmp_path):
    root = make_root(tmp_path, {"docs/agents.md": "x"})
    (root / "CLAUDE.md").symlink_to(root / "docs" / "agents.md")
    found = pr.first_candidate(root)
    assert found.name == "CLAUDE.md"
    assert found.path == root / "CLAUDE.md"
    assert found.real == (root / "docs" / "agents.md").resolve()


@_NO_FIFO
def test_a_fifo_is_not_a_candidate_and_guarded_read_does_not_block(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": "regular"})
    os.mkfifo(root / "CLAUDE.md")
    started = time.monotonic()
    assert pr.first_candidate(root).name == "AGENTS.md"
    read = pr.guarded_read(root / "CLAUDE.md", root, 1000)
    assert read.failure == "not a regular file" and read.data is None
    assert time.monotonic() - started < 1


@pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="needs /dev/zero")
def test_a_link_to_a_device_is_not_a_candidate_and_never_read(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    (root / "CLAUDE.md").symlink_to("/dev/zero")
    started = time.monotonic()
    assert pr.first_candidate(root) is None
    read = pr.guarded_read(root / "CLAUDE.md", root, 1000)
    assert read.failure == "not a regular file"
    assert time.monotonic() - started < 1


def test_a_directory_named_like_a_candidate_is_skipped(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": "x"})
    (root / "CLAUDE.md").mkdir()
    assert pr.first_candidate(root).name == "AGENTS.md"


def test_a_dangling_link_is_skipped(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": "x"})
    (root / "CLAUDE.md").symlink_to(root / "missing.md")
    assert pr.first_candidate(root).name == "AGENTS.md"


def test_claude_folder_contents_are_never_candidates(instructions_home, tmp_path):
    root = make_root(tmp_path, {".claude/commands/x.md": "x", ".claude/agents/y.md": "y"})
    assert pr.first_candidate(root) is None


# --------------------------------------------------------------------------- guarded read


def test_a_file_of_exactly_cap_bytes_is_returned_with_its_digest(tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": b"x" * 50})
    real = (root / "CLAUDE.md").resolve()
    read = pr.guarded_read(real, root, 50)
    assert read.failure is None and read.data == b"x" * 50 and read.size == 50
    assert read.digest == hashlib.sha256(b"x" * 50).hexdigest()


def test_one_byte_over_the_cap_is_too_large(tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": b"x" * 51})
    read = pr.guarded_read((root / "CLAUDE.md").resolve(), root, 50)
    assert read.failure == "too_large" and read.size == 51 and read.data is None


def test_a_missing_file_is_unreadable(tmp_path):
    root = make_root(tmp_path, {})
    read = pr.guarded_read((root / "CLAUDE.md").resolve(), root, 50)
    assert read.failure == "unreadable"


def test_the_digest_is_over_the_raw_bytes(tmp_path):
    raw = b"a\r\n\x1b[2Jb"
    root = make_root(tmp_path, {"CLAUDE.md": raw})
    read = pr.guarded_read((root / "CLAUDE.md").resolve(), root, 100)
    assert read.data == raw and read.digest == hashlib.sha256(raw).hexdigest()


def test_a_swapped_parent_directory_is_not_the_recorded_file(tmp_path):
    root = make_root(tmp_path, {"docs/CLAUDE.md": "inside"})
    other = tmp_path / "other"
    other.mkdir()
    (other / "CLAUDE.md").write_text("elsewhere")
    recorded = (root / "docs" / "CLAUDE.md").resolve()
    assert pr.guarded_read(recorded, root, 100).failure is None
    (root / "docs" / "CLAUDE.md").unlink()
    (root / "docs").rmdir()
    (root / "docs").symlink_to(other)           # outside the root
    assert pr.guarded_read(recorded, root, 100).failure == "not a regular file"
    (root / "docs").unlink()
    (root / "real").mkdir()
    (root / "real" / "CLAUDE.md").write_text("again")
    (root / "docs").symlink_to(root / "real")   # inside the root, still a swap
    assert pr.guarded_read(recorded, root, 100).failure == "not a regular file"


def test_a_path_outside_the_root_is_refused(tmp_path):
    root = make_root(tmp_path, {})
    outside = tmp_path / "out.md"
    outside.write_text("x")
    assert pr.guarded_read(outside.resolve(), root, 100).failure == "not a regular file"


def test_a_last_component_link_is_refused(tmp_path):
    root = make_root(tmp_path, {"real.md": "x"})
    (root / "CLAUDE.md").symlink_to(root / "real.md")
    assert pr.guarded_read(root / "CLAUDE.md", root, 100).failure == "not a regular file"


# --------------------------------------------------------------------------- accept


def test_accept_records_yes_with_the_digest(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "hello"})
    candidate = pr.first_candidate(root)
    record = pr.accept_instructions(root, candidate)
    assert record == InstructionsRecord(
        path=str(candidate.real), name="CLAUDE.md",
        digest=hashlib.sha256(b"hello").hexdigest(), size=5, state="yes",
    )
    assert read_instructions(root) == record


def test_accept_over_the_ceiling_writes_nothing(instructions_home, tmp_path, monkeypatch):
    root = make_root(tmp_path, {"CLAUDE.md": b"123456789"})
    monkeypatch.setattr(pr, "ACCEPT_MAX_BYTES", 8)
    assert pr.accept_instructions(root, pr.first_candidate(root)) is None
    assert not store_file().exists()


def test_accept_on_a_guard_failure_writes_nothing(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    candidate = pr.first_candidate(root)
    (root / "CLAUDE.md").unlink()
    assert pr.accept_instructions(root, candidate) is None
    assert load_store() == {}


def test_accept_binds_a_hard_link_by_its_digest(instructions_home, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("shared")
    root = make_root(tmp_path, {})
    os.link(outside, root / "CLAUDE.md")
    record = pr.accept_instructions(root, pr.first_candidate(root))
    assert record is not None and record.digest == hashlib.sha256(b"shared").hexdigest()


def test_accept_keeps_marks_and_clears_the_refusal_fields(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "new text"})
    marks = (("s1", "2026-10-11T10:00:00+00:00"),)
    shadow = pr.ShadowRecord("AGENTS.md", str(root / "AGENTS.md"), "cd" * 32)
    write_instructions(root, _record(root, harness_written="ef" * 32, pending=marks, shadow=shadow))
    record = pr.accept_instructions(root, pr.first_candidate(root))
    assert record.pending == marks
    assert record.harness_written is None and record.shadow is None


def test_accept_keeps_a_mark_that_has_no_record_yet(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    store_file().write_text(yaml.safe_dump({"version": 1, "roots": {
        str(root.resolve()): {"instructions": {"pending": {"s9": "2026-10-11T10:00:00+00:00"}}},
    }}), encoding="utf-8")
    assert read_instructions(root) is None
    record = pr.accept_instructions(root, pr.first_candidate(root))
    assert record.pending == (("s9", "2026-10-11T10:00:00+00:00"),)


# --------------------------------------------------------------------------- display


@pytest.mark.parametrize("size,text", [
    (0, "0.0 KB"), (410, "0.4 KB"), (5222, "5.1 KB"), (10240, "10 KB"),
    (65536, "64 KB"), (215040, "210 KB"),
])
def test_format_kb(size, text):
    assert pr.format_kb(size) == text


def test_display_name_strips_escapes():
    assert pr.display_name("CLAU\x1b[31mDE.md") == "CLAUDE.md"


def test_the_token_constant_equals_the_contexts():
    from localharness.agent.context import APPROX_CHARS_PER_TOKEN

    assert pr._APPROX_CHARS_PER_TOKEN == APPROX_CHARS_PER_TOKEN


def test_no_public_name_carries_the_hash_algorithm():
    assert [n for n in dir(pr) if "sha256" in n.lower()] == []
    assert pr._digest(b"x") == hashlib.sha256(b"x").hexdigest()


def test_the_accept_ceiling_is_two_mebibytes():
    assert pr.ACCEPT_MAX_BYTES == 2 * 1024 * 1024


# --------------------------------------------------------------------------- never writes


def _module_calls() -> list[ast.Call]:
    tree = ast.parse(Path(pr.__file__).read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _dotted(node: ast.AST) -> str:
    return ast.unparse(node)


def _flags_text(func: ast.FunctionDef, call: ast.Call) -> str:
    """The text of an ``os.open`` flags argument, following a plain local name to its assignment."""
    arg = call.args[1] if len(call.args) > 1 else None
    if isinstance(arg, ast.Name):
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == arg.id for t in node.targets
            ):
                return _dotted(node.value)
    return _dotted(arg) if arg is not None else ""


def test_the_module_opens_a_candidate_only_read_only():
    tree = ast.parse(Path(pr.__file__).read_text(encoding="utf-8"))
    seen = 0
    for func in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for call in (n for n in ast.walk(func) if isinstance(n, ast.Call)):
            if _dotted(call.func) != "os.open":
                continue
            seen += 1
            first, flags = _dotted(call.args[0]), _flags_text(func, call)
            assert "O_RDONLY" in flags or "lock" in first, ast.unparse(call)
            if "lock" not in first:
                assert not any(w in flags for w in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND"))
    assert seen >= 3  # the sidecar, the store read, the guarded read


def test_the_only_writer_is_atomic_write_overlay_on_the_store():
    writers = [c for c in _module_calls() if _dotted(c.func) == "atomic_write_overlay"]
    assert writers
    for call in writers:
        assert isinstance(call.args[0], ast.Call) and _dotted(call.args[0].func) == "store_path"


def test_the_module_calls_no_other_file_writer():
    banned = {"write_text", "write_bytes", "read_text"}
    for call in _module_calls():
        if isinstance(call.func, ast.Attribute):
            assert call.func.attr not in banned, ast.unparse(call)
        if isinstance(call.func, ast.Name):
            assert call.func.id != "open", ast.unparse(call)


# --------------------------------------------------------------------------- notices


def _notice(kind="narrowing", subject="a.b", file="/x/config.yaml", value="10000", text="dropped"):
    return pr.Notice(kind=kind, subject=subject, file=file, value=value, text=text)


def test_unseen_on_an_empty_store_returns_all_and_writes_nothing(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    n1, n2 = _notice(subject="one"), _notice(subject="two")
    assert pr.unseen(root, [n1, n2]) == [n1, n2]
    assert not store_file().exists()


def test_replaced_notices_are_seen(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    n1, n2 = _notice(subject="one"), _notice(subject="two")
    pr.replace_notices(root, [n1, n2])
    assert pr.unseen(root, [n1, n2]) == []
    assert pr.read_notices(root) == frozenset({n1, n2})


def test_a_changed_value_is_unseen_again(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    pr.replace_notices(root, [_notice(value="10000")])
    changed = _notice(value="20000")
    assert pr.unseen(root, [changed]) == [changed]


def test_a_notice_that_disappears_and_returns_prints_again(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    n1, n2 = _notice(subject="one"), _notice(subject="two")
    pr.replace_notices(root, [n1])
    pr.replace_notices(root, [n1, n2])
    pr.replace_notices(root, [n1])
    assert pr.unseen(root, [n1, n2]) == [n2]


def test_unseen_collapses_a_repeat_and_keeps_the_first_position(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    n1, n2 = _notice(subject="one"), _notice(subject="two")
    assert pr.unseen(root, [n2, n1, n2, n1]) == [n2, n1]


def test_replacing_with_nothing_clears_the_notices(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    pr.replace_notices(root, [_notice()])
    pr.replace_notices(root, [])
    assert pr.read_notices(root) == frozenset()
    assert load_store()["roots"] == {}


def test_replace_notices_keeps_the_instructions_and_write_keeps_the_notices(
    instructions_home, tmp_path
):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    record, n1 = _record(root), _notice()
    write_instructions(root, record)
    pr.replace_notices(root, [n1])
    assert read_instructions(root) == record
    write_instructions(root, _record(root, size=99))
    assert pr.read_notices(root) == frozenset({n1})
    pr.replace_notices(root, [])
    assert read_instructions(root) == _record(root, size=99)


def test_malformed_notice_entries_are_skipped(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    good = _notice(subject="good")
    store_file().write_text(yaml.safe_dump({"version": 1, "roots": {str(root.resolve()): {
        "notices": [
            {"kind": "narrowing", "subject": "x", "file": "f", "value": "v"},      # missing text
            {"kind": "narrowing", "subject": "x", "file": "f", "value": 5, "text": "t"},
            "not a mapping",
            {"kind": good.kind, "subject": good.subject, "file": good.file,
             "value": good.value, "text": good.text},
        ],
    }}}), encoding="utf-8")
    assert pr.read_notices(root) == frozenset({good})


def test_a_non_list_notices_field_reads_as_empty(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    store_file().write_text(yaml.safe_dump({"version": 1, "roots": {
        str(root.resolve()): {"notices": "oops"}}}), encoding="utf-8")
    assert pr.read_notices(root) == frozenset()


def test_stored_notices_are_sorted_so_the_file_is_stable(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    pr.replace_notices(root, [_notice(subject="b"), _notice(subject="a")])
    first = store_file().read_bytes()
    pr.replace_notices(root, [_notice(subject="a"), _notice(subject="b")])
    assert store_file().read_bytes() == first


def test_notice_value_is_repr_capped_at_200():
    assert pr.notice_value(10000) == "10000"
    assert pr.notice_value("abc") == "'abc'"
    capped = pr.notice_value("x" * 500)
    assert len(capped) == 200 and capped.endswith("…")
    assert pr.notice_value("y" * 198) == repr("y" * 198)  # exactly 200 chars: untouched


# --------------------------------------------------------------------------- lock across processes

_HOLD = (
    "import fcntl, os, sys, time\n"
    "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
    "fcntl.flock(fd, fcntl.LOCK_EX)\n"
    "print('locked', flush=True)\n"
    "time.sleep(float(sys.argv[2]))\n"
)


def _hold_the_lock(seconds: float) -> subprocess.Popen:
    pr.lock_path().parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(pr.lock_path()), str(seconds)],
        stdout=subprocess.PIPE, text=True,
    )
    assert child.stdout.readline().strip() == "locked"
    return child


def test_a_write_waits_for_another_process_holding_the_lock(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    child = _hold_the_lock(0.6)
    try:
        started = time.monotonic()
        pr.replace_notices(root, [_notice()])
        waited = time.monotonic() - started
    finally:
        child.wait()
    assert waited >= 0.5
    assert pr.read_notices(root) == frozenset({_notice()})


def test_a_lock_held_past_the_budget_fails_closed(instructions_home, tmp_path, monkeypatch):
    root = make_root(tmp_path, {})
    pr.replace_notices(root, [_notice(subject="before")])
    before = store_file().read_bytes()
    monkeypatch.setattr(pr, "LOCK_BUDGET_SECONDS", 0.2)
    child = _hold_the_lock(2.0)
    try:
        started = time.monotonic()
        with pytest.raises(StoreUnavailable):
            pr.replace_notices(root, [_notice(subject="after")])
        assert time.monotonic() - started < 1
    finally:
        child.kill()
        child.wait()
    assert store_file().read_bytes() == before


def test_a_read_also_fails_closed_rather_than_reading_unlocked(instructions_home, tmp_path, monkeypatch):
    root = make_root(tmp_path, {})
    monkeypatch.setattr(pr, "LOCK_BUDGET_SECONDS", 0.2)
    child = _hold_the_lock(2.0)
    try:
        with pytest.raises(StoreUnavailable):
            read_instructions(root)
        with pytest.raises(StoreUnavailable):
            pr.unseen(root, [_notice()])
    finally:
        child.kill()
        child.wait()


def test_a_lock_left_by_a_dead_process_is_harmless(instructions_home, tmp_path):
    root = make_root(tmp_path, {})
    child = _hold_the_lock(30)
    child.kill()
    child.wait()
    pr.replace_notices(root, [_notice()])
    assert pr.read_notices(root) == frozenset({_notice()})


# --------------------------------------------------------------------------- WR-03 / WR-51: an unreadable store


def _two_roots(tmp_path: Path) -> tuple[Path, Path]:
    a = tmp_path / "a"
    b = tmp_path / "b"
    for r in (a, b):
        r.mkdir()
        (r / "CLAUDE.md").write_text("x", encoding="utf-8")
        write_instructions(r, _record(r, state="off" if r is b else "yes"))
    return a, b


def test_an_unreadable_store_is_unavailable_and_the_next_write_does_not_erase_it(
    instructions_home, tmp_path,
):
    a, b = _two_roots(tmp_path)
    before = store_file().read_bytes()
    store_file().chmod(0)
    try:
        if os.access(store_file(), os.R_OK):  # running as root: the mode does not bite
            pytest.skip("permission bits are not enforced for this user")
        with pytest.raises(StoreUnavailable):
            read_instructions(a)
        with pytest.raises(StoreUnavailable):
            pr.replace_notices(a, [pr.Notice("k", "s", "f", "v", "t")])
    finally:
        store_file().chmod(0o600)
    assert store_file().read_bytes() == before
    assert read_instructions(b).state == "off"


def test_a_symlinked_store_is_unavailable_and_left_alone(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    target = tmp_path / "elsewhere.yaml"
    target.write_text("roots: {}\n", encoding="utf-8")
    store_file().symlink_to(target)
    with pytest.raises(StoreUnavailable):
        read_instructions(root)
    with pytest.raises(StoreUnavailable):
        write_instructions(root, _record(root))
    assert store_file().is_symlink()
    assert target.read_text(encoding="utf-8") == "roots: {}\n"


def test_a_directory_in_the_stores_place_is_unavailable(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    store_file().mkdir()
    with pytest.raises(StoreUnavailable):
        read_instructions(root)


def test_an_absent_or_empty_store_is_no_records_not_unavailable(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    assert read_instructions(root) is None
    store_file().write_text("", encoding="utf-8")
    assert read_instructions(root) is None
    assert not store_file().with_name(store_file().name + ".corrupt").exists()


def test_a_damaged_store_is_kept_beside_it_before_a_write_replaces_it(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "x"})
    store_file().write_text("key: [unclosed", encoding="utf-8")
    assert read_instructions(root) is None  # a read leaves it alone
    copy = store_file().with_name(store_file().name + ".corrupt")
    assert not copy.exists()
    write_instructions(root, _record(root))
    assert copy.read_text(encoding="utf-8") == "key: [unclosed"
    assert stat.S_IMODE(copy.stat().st_mode) == 0o600
    assert read_instructions(root) is not None
