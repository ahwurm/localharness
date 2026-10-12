"""config/project_records.py — the per-root instructions store.

Properties held down here: the record lives in the GLOBAL config dir at 0600, keyed by the
resolved root; a damaged store reads as "no record", never an exception; every access runs under
a sidecar lock that is a different inode from the data file; and the module never opens an
instructions file for writing.
"""
# ruff: noqa: F811  (instructions_home is imported as a fixture and requested by name)
from __future__ import annotations

import os
import stat
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
