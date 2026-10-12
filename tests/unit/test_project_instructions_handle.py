"""The project-instructions handle: the D-28 state machine, marks, delivery, set_state, slash texts.

Every store path is the hermetic ``LOCALHARNESS_HOME`` of the autouse fixture; nothing here
touches the real ``~/.localharness``.
"""
# ruff: noqa: F811  (instructions_home is imported as a fixture and requested by name)
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from localharness.config import project_records as pr
from localharness.config.project_records import (
    ProjectInstructions,
    open_project_instructions,
)
from tests.unit.instructions_support import (
    instructions_home,  # noqa: F401
    load_store,
    make_root,
    store_file,
)

WINDOW = 65536
TEXT = "a" * 5221 + "\n"  # 5,222 bytes: 5.1 KB


def _open(root: Path, **kw) -> ProjectInstructions:
    args = {
        "sitting_id": "s1", "max_ratio": 0.25, "basis": "layer", "no_input": False,
        "window_tokens": WINDOW,
    }
    args.update(kw)
    handle = open_project_instructions(root, **args)
    assert handle is not None
    return handle


def _digest(data: str | bytes) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def _record_of(root: Path) -> dict:
    return load_store()["roots"][str(root.resolve())]["instructions"]


# --------------------------------------------------------------------------- tracer


def test_a_fresh_roots_file_is_used_with_one_start_line_and_nothing_is_written(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    assert h.start_line == "Using CLAUDE.md (5.1 KB); /instructions off"
    assert h.summary_entity == "CLAUDE.md"
    assert h.state.state == "none"
    assert not store_file().exists()
    text, notice = h.section(WINDOW)
    assert text == TEXT
    assert notice is not None
    assert notice.cause == "start"
    assert notice.name == "CLAUDE.md"
    assert notice.text == h.start_line
    assert h.section(WINDOW) == (TEXT, None)
    assert not store_file().exists()


def test_the_receipt_writes_the_record_once_and_the_next_start_is_silent(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    h.section(WINDOW)
    h.on_delivered()
    real = (root / "CLAUDE.md").resolve()
    record = _record_of(root)
    assert record == {
        "path": str(real), "name": "CLAUDE.md", "digest": _digest(TEXT), "size": 5222,
        "state": "yes",
    }
    written = store_file().read_bytes()
    h.on_delivered()
    assert store_file().read_bytes() == written
    nxt = _open(root, sitting_id="s2")
    assert nxt.start_line is None
    assert nxt.summary_entity == "CLAUDE.md"
    assert nxt.section(WINDOW) == (TEXT, None)
    assert store_file().read_bytes() == written


def test_a_root_with_no_candidate_has_no_handle(instructions_home, tmp_path):
    root = make_root(tmp_path, {"README.md": "x"})
    assert open_project_instructions(
        root, sitting_id="s1", max_ratio=0.25, basis="layer", no_input=False,
        window_tokens=WINDOW,
    ) is None


def test_the_constructor_does_no_filesystem_call(instructions_home, tmp_path, monkeypatch):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})

    def boom(*a, **k):
        raise AssertionError("filesystem call in the constructor")

    for name in ("open", "stat", "lstat"):
        monkeypatch.setattr(os, name, boom)
    h = ProjectInstructions(
        root, sitting_id="s", max_ratio=0.25, basis="layer", no_input=False,
    )
    monkeypatch.undo()
    assert h.start_line is None
    assert h.summary_entity is None


def test_the_candidates_are_the_three_paths_in_order(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    assert h.candidates == (
        root / "CLAUDE.md", root / "AGENTS.md", root / ".localharness/INSTRUCTIONS.md"
    )


def test_a_receipt_before_the_first_section_means_no_start_line_in_it(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    h.on_delivered()
    assert h.section(WINDOW) == (TEXT, None)


def test_a_matching_record_written_elsewhere_drops_the_owed_line(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    other = _open(root, sitting_id="s2")
    other.on_delivered()
    assert h.section(WINDOW) == (TEXT, None)
    assert h.start_line is None


def test_an_awaiting_trust_root_without_a_record_is_unused_and_silent(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, basis="awaiting_trust")
    assert h.start_line is None
    assert h.summary_entity is None
    assert h.section(WINDOW) == (None, None)
    pr.accept_instructions(root, pr.first_candidate(root))
    assert h.section(WINDOW) == (TEXT, None)


def test_the_handle_state_names_the_file_before_any_read(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": TEXT})
    h = _open(root, basis="awaiting_trust")
    assert h.state.name == "AGENTS.md"
    assert h.state.path == str((root / "AGENTS.md").resolve())
