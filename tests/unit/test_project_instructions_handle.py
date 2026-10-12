"""The project-instructions handle: the D-28 state machine, marks, delivery, set_state, slash texts.

Every store path is the hermetic ``LOCALHARNESS_HOME`` of the autouse fixture; nothing here
touches the real ``~/.localharness``.
"""
# ruff: noqa: F811  (instructions_home is imported as a fixture and requested by name)
from __future__ import annotations

import fcntl
import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from localharness.config import project_records as pr
from localharness.config.project_records import (
    ProjectInstructions,
    open_project_instructions,
    run_instructions_command,
)
from tests.unit.instructions_support import (
    dead_pid,
    instructions_home,  # noqa: F401
    live_child,
    load_store,
    make_root,
    register_session,
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


# --------------------------------------------------------------------------- helpers for change sensing


def _recorded(root: Path, **kw) -> ProjectInstructions:
    """A handle on a root whose current first candidate was accepted (a yes record)."""
    cand = pr.first_candidate(root)
    assert cand is not None
    assert pr.accept_instructions(root, cand) is not None
    return _open(root, **kw)


def _rewrite(root: Path, text: str, name: str = "CLAUDE.md") -> None:
    (root / name).write_text(text, encoding="utf-8")


def _put_mark(root: Path, sid: str, when: str = "2026-10-12T00:00:00+00:00") -> None:
    with pr._store_transaction() as data:
        data["roots"][str(root.resolve())]["instructions"].setdefault("pending", {})[sid] = when


def _stored_marks(root: Path) -> dict:
    return _record_of(root).get("pending", {})


@contextmanager
def _lock_held():
    pr.lock_path().parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(pr.lock_path(), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        os.close(fd)


CHANGED = "something else\n"
HW = "CLAUDE.md changed by the model; not used — /instructions on"
ADOPTED = "CLAUDE.md changed, using it"


# --------------------------------------------------------------------------- marks


def test_a_mark_is_written_before_it_returns_and_a_model_write_is_dropped(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    assert set(_stored_marks(root)) == {"s1"}
    assert _stored_marks(root)["s1"].endswith("+00:00")
    _rewrite(root, CHANGED)
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.cause == "harness_written"
    assert notice.text == HW
    record = _record_of(root)
    assert record["harness_written"] == _digest(CHANGED)
    assert record["digest"] == _digest(TEXT)
    assert "pending" not in record
    assert h.section(WINDOW) == (None, None)
    assert h.state.state == "harness_written"


def test_a_mark_the_store_could_not_take_still_drops_a_model_write(
    instructions_home, tmp_path, monkeypatch,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    monkeypatch.setattr(pr, "LOCK_BUDGET_SECONDS", 0.1)
    with _lock_held():
        h.mark_pending()  # the store is unavailable: returns without raising
    assert "pending" not in _record_of(root)
    _rewrite(root, CHANGED)
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.cause == "harness_written"
    assert _record_of(root)["harness_written"] == _digest(CHANGED)


def test_a_mark_with_no_record_is_a_mark_only_mapping(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, basis="awaiting_trust")
    h.mark_pending()
    assert set(_record_of(root)) == {"pending"}
    assert pr.read_instructions(root) is None


def test_a_change_no_call_named_is_adopted_with_a_line(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    _rewrite(root, CHANGED)
    text, notice = h.section(WINDOW)
    assert text == CHANGED
    assert notice is not None and notice.cause == "adopted" and notice.text == ADOPTED
    record = _record_of(root)
    assert record["digest"] == _digest(CHANGED)
    assert record["size"] == len(CHANGED)
    assert h.section(WINDOW) == (CHANGED, None)


def test_a_mark_with_an_unchanged_file_is_cleared(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    assert h.section(WINDOW) == (TEXT, None)
    assert "pending" not in _record_of(root)
    _rewrite(root, CHANGED)  # the mark is gone: this is now the person's change
    assert h.section(WINDOW)[1].cause == "adopted"


def test_another_live_sessions_mark_is_never_cleared_by_this_handle(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    with live_child() as pid:
        register_session("s2", pid=pid)
        _put_mark(root, "s2")
        assert h.section(WINDOW) == (TEXT, None)
        assert set(_stored_marks(root)) == {"s2"}
        _rewrite(root, CHANGED)
        text, notice = h.section(WINDOW)
        assert text is None and notice.cause == "harness_written"
        assert set(_stored_marks(root)) == {"s2"}
        assert _record_of(root)["harness_written"] == _digest(CHANGED)


def test_a_dead_sessions_mark_is_cleared_while_the_digest_holds(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    register_session("s3", pid=dead_pid())
    _put_mark(root, "s3")
    assert h.section(WINDOW) == (TEXT, None)
    assert "pending" not in _record_of(root)


def test_a_dead_sessions_mark_is_still_a_mark_when_the_digest_differs(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    _put_mark(root, "s3")  # never registered: not live
    _rewrite(root, CHANGED)
    text, notice = h.section(WINDOW)
    assert text is None and notice.cause == "harness_written"
    assert "pending" not in _record_of(root)


# --------------------------------------------------------------------------- --no-input


def test_no_input_uses_a_changed_file_under_a_record_and_writes_nothing(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    _recorded(root)
    h = _open(root, no_input=True, sitting_id="n1")
    before = store_file().read_bytes()
    assert h.section(WINDOW) == (TEXT, None)
    _rewrite(root, CHANGED)
    assert h.section(WINDOW) == (CHANGED, None)
    assert store_file().read_bytes() == before


def test_no_input_skips_a_file_with_a_live_foreign_mark_and_ignores_a_dead_one(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    _recorded(root)
    h = _open(root, no_input=True, sitting_id="n1")
    _rewrite(root, CHANGED)
    _put_mark(root, "gone")
    before = store_file().read_bytes()
    assert h.section(WINDOW) == (CHANGED, None)  # a dead mark is absent
    with live_child() as pid:
        register_session("s2", pid=pid)
        _put_mark(root, "s2")
        before = store_file().read_bytes()
        assert h.section(WINDOW) == (None, None)
        assert store_file().read_bytes() == before


def test_no_input_without_a_record_uses_nothing_and_never_marks(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, no_input=True)
    assert h.start_line is None
    assert h.section(WINDOW) == (None, None)
    h.mark_pending()
    h.on_delivered()
    assert not store_file().exists()


def test_no_input_own_mark_keeps_the_model_write_out_without_writing(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    _recorded(root)
    h = _open(root, no_input=True, sitting_id="n1")
    before = store_file().read_bytes()
    h.mark_pending()
    _rewrite(root, CHANGED)
    assert h.section(WINDOW) == (None, None)
    assert store_file().read_bytes() == before


# --------------------------------------------------------------------------- shadow


def test_a_file_above_the_recorded_one_under_a_mark_is_a_shadow(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    _rewrite(root, CHANGED, "CLAUDE.md")
    text, notice = h.section(WINDOW)
    assert text == TEXT
    assert notice is not None and notice.cause == "shadowed"
    assert notice.shadowed == "CLAUDE.md" and notice.name == "AGENTS.md"
    assert notice.text == "CLAUDE.md (model edit) not used; using AGENTS.md — /instructions on to switch"
    record = _record_of(root)
    assert record["shadow"]["name"] == "CLAUDE.md"
    assert record["shadow"]["digest"] == _digest(CHANGED)
    assert set(record["pending"]) == {"s1"}  # the unchanged AGENTS.md does not clear the mark
    assert h.section(WINDOW) == (TEXT, None)


def test_a_file_above_the_recorded_one_with_no_mark_is_adopted(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": TEXT})
    h = _recorded(root)
    _rewrite(root, CHANGED, "CLAUDE.md")
    text, notice = h.section(WINDOW)
    assert text == CHANGED
    assert notice is not None and notice.cause == "adopted" and notice.text == ADOPTED
    record = _record_of(root)
    assert record["name"] == "CLAUDE.md"
    assert record["path"] == str((root / "CLAUDE.md").resolve())
    assert record["digest"] == _digest(CHANGED)


# --------------------------------------------------------------------------- refusals


def test_a_fifo_swapped_in_is_refused_once_without_blocking(instructions_home, tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no mkfifo")
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    (root / "CLAUDE.md").unlink()
    os.mkfifo(root / "CLAUDE.md")
    started = time.monotonic()
    text, notice = h.section(WINDOW)
    assert time.monotonic() - started < 1
    assert text is None
    assert notice is not None and notice.cause == "refused"
    assert notice.text == "CLAUDE.md not used this turn (not a regular file)"
    assert h.section(WINDOW) == (None, None)


def test_a_link_to_a_device_swapped_in_is_refused(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    (root / "CLAUDE.md").unlink()
    (root / "CLAUDE.md").symlink_to("/dev/zero")
    started = time.monotonic()
    text, notice = h.section(WINDOW)
    assert time.monotonic() - started < 1
    assert text is None
    assert notice.text == "CLAUDE.md not used this turn (not a regular file)"


def test_a_deleted_file_is_refused_as_unreadable(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    (root / "CLAUDE.md").unlink()
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice.text == "CLAUDE.md not used this turn (unreadable)"
    assert h.section(WINDOW) == (None, None)


def test_a_file_grown_past_the_cap_is_refused_once_naming_the_limit(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    (root / "CLAUDE.md").write_bytes(b"x" * 215040)
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice.cause == "too_large"
    assert notice.text == "CLAUDE.md (210 KB) over the 64 KB limit (instructions.max_ratio); not used"
    assert h.section(WINDOW) == (None, None)
    assert h.state.state == "too_large"


def test_an_oversize_the_start_already_reported_is_silent(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    (root / "CLAUDE.md").write_bytes(b"x" * 215040)
    real = str((root / "CLAUDE.md").resolve())
    pr.replace_notices(root, [pr.Notice("instructions_too_large", real, real, "215040", "t")])
    assert h.section(WINDOW) == (None, None)


def test_an_oversize_that_is_returned_is_owed_to_the_store_until_delivered(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    (root / "CLAUDE.md").write_bytes(b"x" * 215040)
    h.section(WINDOW)
    h.on_delivered()
    real = str((root / "CLAUDE.md").resolve())
    stored = pr.read_notices(root)
    assert [(n.kind, n.subject, n.file, n.value) for n in stored] == [
        ("instructions_too_large", real, real, "215040")
    ]
    assert _record_of(root)["digest"] == _digest(TEXT)  # the yes record is untouched


def test_an_unavailable_store_is_one_refusal_and_no_text(instructions_home, tmp_path, monkeypatch):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    monkeypatch.setattr(pr, "LOCK_BUDGET_SECONDS", 0.1)
    with _lock_held():
        text, notice = h.section(WINDOW)
        assert text is None
        assert notice is not None and notice.cause == "refused"
        assert notice.text == "CLAUDE.md not used this turn (unreadable store)"
        assert h.section(WINDOW) == (None, None)
    assert h.section(WINDOW) == (TEXT, None)


# --------------------------------------------------------------------------- cancel


def test_a_cancelled_transition_is_written_and_its_line_held_for_the_next_call(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    _rewrite(root, CHANGED)
    assert h.section(WINDOW, cancelled=True) == (None, None)
    assert _record_of(root)["harness_written"] == _digest(CHANGED)
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.cause == "harness_written"
    assert h.section(WINDOW) == (None, None)


def test_a_call_with_its_own_line_drops_the_held_one(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    _rewrite(root, CHANGED)
    h.section(WINDOW, cancelled=True)
    (root / "CLAUDE.md").unlink()
    _, notice = h.section(WINDOW)
    assert notice.cause == "refused"
    assert h.section(WINDOW) == (None, None)


# --------------------------------------------------------------------------- a sitting with no record


def test_a_model_write_before_the_line_is_delivered_is_never_recorded_as_the_persons(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    text, notice = h.section(WINDOW)
    assert text == TEXT and notice.cause == "start"
    h.mark_pending()
    _rewrite(root, CHANGED)
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.cause == "harness_written"
    assert h.section(WINDOW) == (None, None)
    h.on_delivered()
    record = _record_of(root)
    assert record["digest"] == _digest(TEXT)
    assert record["harness_written"] == _digest(CHANGED)
    assert "pending" not in record


def test_a_hand_edit_in_a_sitting_with_no_record_is_used_with_no_second_line(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    h.section(WINDOW)
    _rewrite(root, CHANGED)
    assert h.section(WINDOW) == (CHANGED, None)
    assert not store_file().exists()


# --------------------------------------------------------------------------- links


def test_an_in_root_link_is_used_at_every_call_by_its_real_path(instructions_home, tmp_path):
    root = make_root(tmp_path, {"docs/agents.md": TEXT})
    (root / "CLAUDE.md").symlink_to("docs/agents.md")
    h = _recorded(root)
    assert _record_of(root)["path"] == str((root / "docs/agents.md").resolve())
    assert h.section(WINDOW) == (TEXT, None)
    assert h.section(WINDOW) == (TEXT, None)
    h.mark_pending()
    assert h.section(WINDOW) == (TEXT, None)
    assert "pending" not in _record_of(root)  # the first present candidate resolves to the record


def test_a_hard_link_to_an_outside_file_is_used_and_bound_by_its_digest(instructions_home, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text(TEXT, encoding="utf-8")
    root = make_root(tmp_path, {})
    os.link(outside, root / "CLAUDE.md")
    h = _recorded(root)
    assert h.section(WINDOW) == (TEXT, None)
    outside.write_text(CHANGED, encoding="utf-8")  # same inode: the digest moves
    text, notice = h.section(WINDOW)
    assert text == CHANGED and notice.cause == "adopted"


def test_an_off_record_is_never_used_or_re_adopted(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    record = pr.read_instructions(root)
    pr.write_instructions(root, pr.InstructionsRecord(**{**record.__dict__, "state": "off"}))
    assert h.section(WINDOW) == (None, None)
    _rewrite(root, CHANGED)
    assert h.section(WINDOW) == (None, None)
    assert h.state.state == "off"


# --------------------------------------------------------------------------- the start pre-check


def test_a_change_between_sessions_is_adopted_through_the_one_start_line(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    _recorded(root)
    _rewrite(root, CHANGED)
    before = store_file().read_bytes()
    h = _open(root, sitting_id="s2")
    assert h.start_line == "Using CLAUDE.md (0.0 KB); /instructions off"
    assert h.summary_entity == "CLAUDE.md"
    assert store_file().read_bytes() == before
    text, notice = h.section(WINDOW)
    assert text == CHANGED
    assert notice is not None and notice.cause == "start"  # never an adopted line as well
    assert h.section(WINDOW) == (CHANGED, None)
    h.on_delivered()
    assert _record_of(root)["digest"] == _digest(CHANGED)
    assert h.start_line is None


def test_a_change_under_a_dead_sessions_mark_is_the_models_and_is_held_for_the_first_assembly(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    _recorded(root)
    _put_mark(root, "gone")
    _rewrite(root, CHANGED)
    h = _open(root, sitting_id="s2")
    assert h.start_line is None
    assert h.summary_entity == "CLAUDE.md (model edit; /instructions on)"
    record = _record_of(root)
    assert record["harness_written"] == _digest(CHANGED)
    assert "pending" not in record
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.cause == "harness_written" and notice.text == HW
    assert h.section(WINDOW) == (None, None)


def test_a_dead_mark_on_an_unchanged_file_is_cleared_at_start(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    _recorded(root)
    _put_mark(root, "gone")
    h = _open(root, sitting_id="s2")
    assert "pending" not in _record_of(root)
    assert h.summary_entity == "CLAUDE.md"
    assert h.start_line is None


def test_an_oversize_file_at_start_sets_the_notice_and_the_entity_and_no_line(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": b"x" * 215040})
    h = _open(root)
    real = str((root / "CLAUDE.md").resolve())
    assert h.too_large_notice == pr.Notice(
        "instructions_too_large", real, real, "215040",
        "CLAUDE.md (210 KB) over the 64 KB limit (instructions.max_ratio); not used",
    )
    assert h.summary_entity == "CLAUDE.md (too large)"
    assert h.start_line is None
    assert h.state.state == "too_large"


def test_an_off_record_shows_off_and_stays_off_through_a_hand_edit(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    assert run_instructions_command(h, "off").startswith("instructions off")
    again = _open(root, sitting_id="s2")
    assert again.summary_entity == "CLAUDE.md (off)"
    assert again.start_line is None
    assert again.section(WINDOW) == (None, None)
    _rewrite(root, CHANGED)
    assert again.section(WINDOW) == (None, None)


def test_a_file_the_account_does_not_own_is_used_only_under_a_record(instructions_home, tmp_path, monkeypatch):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(root / "CLAUDE.md").st_uid + 1)
    h = _open(root)
    assert h.start_line is None
    assert h.summary_entity == "CLAUDE.md (not yours; /instructions on)"  # WR-53
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.cause == "refused" and notice.reason == "not owned by you"
    assert notice.text == "CLAUDE.md not used (not owned by you); /instructions on"
    assert h.section(WINDOW) == (None, None)  # once
    pr.accept_instructions(root, pr.first_candidate(root))  # an answered prompt covers any owner
    assert h.section(WINDOW) == (TEXT, None)


def test_on_brings_in_a_file_the_account_does_not_own(instructions_home, tmp_path, monkeypatch):
    """A trusted root asks no question, so `/instructions on` is the way in (WR-53)."""
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(root / "CLAUDE.md").st_uid + 1)
    h = _open(root, basis="trusted")
    assert h.section(WINDOW)[0] is None
    assert run_instructions_command(h, "on") == "using CLAUDE.md (5.1 KB)"
    assert h.section(WINDOW) == (TEXT, None)
    assert h.summary_entity == "CLAUDE.md (not yours; /instructions on)"  # the start's own, unchanged
    assert _record_of(root)["state"] == "yes"


def test_a_shadow_found_at_start_under_a_mark_is_recorded_and_held(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": TEXT})
    _recorded(root)
    _put_mark(root, "gone")
    _rewrite(root, CHANGED, "CLAUDE.md")
    h = _open(root, sitting_id="s2")
    assert h.summary_entity == "AGENTS.md"
    assert _record_of(root)["shadow"]["name"] == "CLAUDE.md"
    text, notice = h.section(WINDOW)
    assert text == TEXT
    assert notice is not None and notice.cause == "shadowed"
    assert h.section(WINDOW) == (TEXT, None)


def test_a_file_above_the_recorded_one_at_start_with_no_mark_takes_the_start_line(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": TEXT})
    _recorded(root)
    _rewrite(root, CHANGED, "CLAUDE.md")
    h = _open(root, sitting_id="s2")
    assert h.start_line == "Using CLAUDE.md (0.0 KB); /instructions off"
    text, notice = h.section(WINDOW)
    assert text == CHANGED and notice.cause == "start"
    h.on_delivered()
    assert _record_of(root)["name"] == "CLAUDE.md"


@pytest.mark.parametrize("case", ["unchanged", "changed", "dead_mark", "changed_dead_mark", "oversize"])
def test_the_no_input_pre_check_never_writes_and_never_owes_a_line(instructions_home, tmp_path, case):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    _recorded(root)
    if case in ("changed", "changed_dead_mark"):
        _rewrite(root, CHANGED)
    if case in ("dead_mark", "changed_dead_mark"):
        _put_mark(root, "gone")
    if case == "oversize":
        (root / "CLAUDE.md").write_bytes(b"x" * 215040)
    before = store_file().read_bytes()
    h = _open(root, no_input=True, sitting_id="n1")
    h.section(WINDOW)
    h.on_delivered()
    assert h.start_line is None
    assert store_file().read_bytes() == before


# --------------------------------------------------------------------------- state changes and /instructions


def test_off_then_on_round_trip_and_their_texts(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    assert run_instructions_command(h, "off") == (
        "instructions off for this workspace; /instructions on to use CLAUDE.md"
    )
    assert _record_of(root)["state"] == "off"
    assert _record_of(root)["digest"] == _digest(TEXT)
    assert h.section(WINDOW) == (None, None)
    assert run_instructions_command(h, "ON") == "using CLAUDE.md (5.1 KB)"
    assert _record_of(root)["state"] == "yes"
    assert h.section(WINDOW) == (TEXT, None)


def test_on_brings_a_dropped_model_edit_back_with_the_current_digest(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    _rewrite(root, CHANGED)
    assert h.section(WINDOW)[1].cause == "harness_written"
    assert run_instructions_command(h, "on") == "using CLAUDE.md (0.0 KB)"
    record = _record_of(root)
    assert record["digest"] == _digest(CHANGED)
    assert "harness_written" not in record and "shadow" not in record
    assert h.section(WINDOW) == (CHANGED, None)


def test_on_in_a_shadowed_root_records_the_first_resolving_candidate(instructions_home, tmp_path):
    root = make_root(tmp_path, {"AGENTS.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    _rewrite(root, CHANGED, "CLAUDE.md")
    assert h.section(WINDOW)[1].cause == "shadowed"
    assert run_instructions_command(h, "on") == "using CLAUDE.md (0.0 KB)"
    record = _record_of(root)
    assert record["name"] == "CLAUDE.md"
    assert "shadow" not in record
    assert "pending" not in record  # the person's answer settles this session's own mark
    assert h.section(WINDOW) == (CHANGED, None)


def test_on_keeps_another_live_sessions_mark(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    with live_child() as pid:
        register_session("s2", pid=pid)
        _put_mark(root, "s2")
        run_instructions_command(h, "on")
        assert set(_stored_marks(root)) == {"s2"}


def test_off_with_no_record_creates_one_from_the_file(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    run_instructions_command(h, "off")
    record = _record_of(root)
    assert record["state"] == "off"
    assert record["digest"] == _digest(TEXT)
    h.on_delivered()  # the owed line was settled by the answer
    assert _record_of(root)["state"] == "off"
    assert h.start_line is None


def test_on_over_the_ceiling_names_why_and_writes_nothing(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": b"x" * (pr.ACCEPT_MAX_BYTES + 1)})
    h = _open(root, basis="awaiting_trust")
    assert run_instructions_command(h, "on") == "CLAUDE.md not used this turn (too large)"
    assert not store_file().exists()


def test_the_show_text_carries_the_path_size_full_digest_and_state(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    real = (root / "CLAUDE.md").resolve()
    assert run_instructions_command(h, "show") == (
        f"CLAUDE.md: {str(real)!r} · 5.1 KB · sha256 {_digest(TEXT)} · yes"
    )
    assert len(_digest(TEXT)) == 64


def test_the_other_slash_texts(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    assert run_instructions_command(None, "show") == "no instructions file in this workspace"
    assert run_instructions_command(None, "off") == "no instructions file in this workspace"
    assert run_instructions_command(h, "preview") == "usage: /instructions off | on | show"
    assert run_instructions_command(h, "") == "usage: /instructions off | on | show"
    assert run_instructions_command(h, "off now") == "usage: /instructions off | on | show"


def test_a_declined_handle_is_inert_and_writes_nothing(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    h.decline()
    assert h.section(WINDOW) == (None, None)
    h.mark_pending()
    h.on_delivered()
    h.set_state("yes")
    assert run_instructions_command(h, "on") == (
        "instructions are not used in this session (the workspace was declined)"
    )
    assert run_instructions_command(h, "off") == (
        "instructions are not used in this session (the workspace was declined)"
    )
    assert run_instructions_command(h, "show").startswith("CLAUDE.md: ")
    assert not store_file().exists()


def test_set_state_under_no_input_changes_this_sitting_only(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    _recorded(root)
    h = _open(root, no_input=True, sitting_id="n1")
    before = store_file().read_bytes()
    assert run_instructions_command(h, "off").startswith("instructions off")
    assert store_file().read_bytes() == before
    assert h.section(WINDOW) == (None, None)
    assert run_instructions_command(h, "on") == "using CLAUDE.md (5.1 KB)"
    assert h.section(WINDOW) == (TEXT, None)
    assert store_file().read_bytes() == before


def test_one_shared_handle_dedupes_a_refusal_across_parent_and_child(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    (root / "CLAUDE.md").unlink()
    parent = h.section(WINDOW)
    child = h.section(WINDOW)
    assert [n for _, n in (parent, child) if n is not None] == [parent[1]]


def test_a_failed_set_state_never_raises(instructions_home, tmp_path, monkeypatch):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    monkeypatch.setattr(pr, "LOCK_BUDGET_SECONDS", 0.1)
    with _lock_held():
        assert run_instructions_command(h, "off") == "instructions not changed (unreadable store)"
        assert run_instructions_command(h, "on") == "CLAUDE.md not used this turn (unreadable store)"
    assert _record_of(root)["state"] == "yes"  # the store was unavailable: nothing changed


# --------------------------------------------------------------------------- CR-51: a vanished record


def test_a_record_that_vanishes_under_a_mark_never_adopts_the_models_rewrite(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    assert h.section(WINDOW) == (TEXT, None)
    h.mark_pending()
    _rewrite(root, CHANGED)
    store_file().unlink()
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.cause == "refused" and notice.reason == "record lost"
    assert notice.text == "CLAUDE.md not used this turn (record lost); /instructions on"
    assert h.section(WINDOW) == (None, None)
    assert not store_file().exists()  # nothing was adopted or written
    assert run_instructions_command(h, "on") == f"using CLAUDE.md ({pr.format_kb(len(CHANGED))})"
    assert h.section(WINDOW) == (CHANGED, None)


def test_a_record_that_vanishes_does_not_undo_off(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.set_state("off")
    store_file().unlink()
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.reason == "record lost"
    assert not store_file().exists()


def test_a_record_written_by_the_receipt_that_vanishes_is_not_a_fresh_root(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    assert h.section(WINDOW)[0] == TEXT
    h.on_delivered()
    store_file().unlink()
    _rewrite(root, CHANGED)
    text, notice = h.section(WINDOW)
    assert text is None and notice.reason == "record lost"
    assert not store_file().exists()


def test_a_second_record_loss_after_on_is_said_again(instructions_home, tmp_path):
    """IN-06: the once-per-handle line is re-armed by `/instructions on`."""
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    assert h.section(WINDOW) == (TEXT, None)
    store_file().unlink()
    first = h.section(WINDOW)[1]
    assert first is not None and first.reason == "record lost"
    assert h.section(WINDOW) == (None, None)  # once per loss
    assert run_instructions_command(h, "on").startswith("using CLAUDE.md")
    assert h.section(WINDOW) == (TEXT, None)
    store_file().unlink()
    second = h.section(WINDOW)[1]
    assert second is not None and second.reason == "record lost"


def test_a_corrupt_store_under_a_running_sitting_is_not_a_fresh_root(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    assert h.section(WINDOW) == (TEXT, None)
    store_file().write_text("[: not yaml", encoding="utf-8")
    _rewrite(root, CHANGED)
    text, notice = h.section(WINDOW)
    assert text is None and notice.reason == "record lost"


# --------------------------------------------------------------------------- WR-07/WR-55: a mark is not a record seen


def test_a_mark_in_a_sitting_that_never_had_a_record_is_not_a_record_lost_awaiting_trust(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, basis="awaiting_trust")
    assert h.section(WINDOW) == (None, None)
    h.mark_pending()  # a call that names the file
    assert h.section(WINDOW) == (None, None)  # no "record lost": no record ever existed


def test_a_mark_under_no_input_without_a_record_is_not_a_record_lost(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, no_input=True)
    assert h.section(WINDOW) == (None, None)
    h.mark_pending()
    assert h.section(WINDOW) == (None, None)
    assert not store_file().exists()


def test_no_input_on_then_a_mark_keeps_the_file_in_use_every_turn(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, no_input=True)
    assert run_instructions_command(h, "on").startswith("using CLAUDE.md")
    assert h.section(WINDOW) == (TEXT, None)
    h.mark_pending()  # a harness call that names the file; the person's yes stands
    assert h.section(WINDOW) == (TEXT, None)
    assert h.section(WINDOW) == (TEXT, None)  # not dropped from the first turn end on
    assert run_instructions_command(h, "on").startswith("using CLAUDE.md")
    assert h.section(WINDOW) == (TEXT, None)
    assert not h._marked


def test_no_input_on_then_a_marked_rewrite_is_dropped_with_the_notice_and_nothing_is_written(
    instructions_home, tmp_path,
):
    """WR-57: the person's in-memory yes accepted the bytes they saw; a rewrite a call named is the
    model's, so it is dropped (D-28 step 7) and offered back, never adopted every turn."""
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, no_input=True)
    assert run_instructions_command(h, "on").startswith("using CLAUDE.md")
    assert h.section(WINDOW) == (TEXT, None)
    h.mark_pending()
    _rewrite(root, CHANGED)
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.cause == "harness_written"
    assert notice.text == "CLAUDE.md changed by the model; not used — /instructions on"
    assert h.section(WINDOW) == (None, None)  # the dropped bytes stay out, with no second notice
    assert h.section(WINDOW) == (None, None)
    assert not store_file().exists()
    assert run_instructions_command(h, "on").startswith("using CLAUDE.md")  # offered back
    assert h.section(WINDOW) == (CHANGED, None)
    assert not store_file().exists()


def test_no_input_on_then_a_second_marked_rewrite_is_dropped_again(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, no_input=True)
    run_instructions_command(h, "on")
    h.mark_pending()
    _rewrite(root, CHANGED)
    assert h.section(WINDOW)[0] is None
    h.mark_pending()
    _rewrite(root, CHANGED + "and more\n")
    text, notice = h.section(WINDOW)
    assert text is None and notice is not None and notice.cause == "harness_written"


def test_no_input_on_then_an_unmarked_hand_edit_is_adopted(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, no_input=True)
    run_instructions_command(h, "on")
    assert h.section(WINDOW) == (TEXT, None)
    _rewrite(root, CHANGED)  # no call named the file: the person's change
    assert h.section(WINDOW) == (CHANGED, None)
    assert not store_file().exists()


def test_no_input_on_then_a_dropped_rewrite_and_a_later_hand_edit_is_the_persons(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root, no_input=True)
    run_instructions_command(h, "on")
    h.mark_pending()
    _rewrite(root, CHANGED)
    assert h.section(WINDOW)[0] is None
    _rewrite(root, "the person's own words\n")
    assert h.section(WINDOW) == ("the person's own words\n", None)


def test_a_mark_does_not_pre_empt_the_not_owned_line(instructions_home, tmp_path, monkeypatch):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(root / "CLAUDE.md").st_uid + 1)
    h = _open(root)
    h.mark_pending()
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.reason == "not owned by you"
    assert notice.text == "CLAUDE.md not used (not owned by you); /instructions on"


# --------------------------------------------------------------------------- WR-01: a recorded file that is gone


def test_a_recorded_file_that_is_gone_is_replaced_by_the_next_candidate(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT, "AGENTS.md": CHANGED})
    h = _recorded(root)
    (root / "CLAUDE.md").unlink()
    text, notice = h.section(WINDOW)
    assert text == CHANGED
    assert notice is not None and notice.cause == "adopted"
    assert notice.text == "AGENTS.md changed, using it"
    record = _record_of(root)
    assert record["name"] == "AGENTS.md" and record["digest"] == _digest(CHANGED)
    assert h.section(WINDOW) == (CHANGED, None)


def test_a_recorded_file_gone_before_start_takes_the_start_line_for_the_next_one(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT, "AGENTS.md": CHANGED})
    _recorded(root)
    (root / "CLAUDE.md").unlink()
    nxt = _open(root, sitting_id="s2")
    assert nxt.start_line == "Using AGENTS.md (0.0 KB); /instructions off"
    assert nxt.section(WINDOW)[0] == CHANGED


def test_a_recorded_file_that_is_gone_under_a_mark_is_still_refused(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT, "AGENTS.md": CHANGED})
    h = _recorded(root)
    h.mark_pending()
    (root / "CLAUDE.md").unlink()
    text, notice = h.section(WINDOW)
    assert text is None
    assert notice is not None and notice.reason == "unreadable"


# --------------------------------------------------------------------------- WR-05: a mark outlives its turn


def test_a_mark_on_an_undelivered_start_line_is_cleared_when_the_file_is_unchanged(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    assert h.section(WINDOW)[0] == TEXT  # the start line is owed and never delivered
    h.mark_pending()  # a read that merely names the file
    assert h.section(WINDOW) == (TEXT, None)
    assert not store_file().exists() or "pending" not in load_store()["roots"].get(
        str(root.resolve()), {}
    ).get("instructions", {})
    _rewrite(root, CHANGED)  # the person's own edit afterwards
    assert h.section(WINDOW) == (CHANGED, None)  # used, not dropped as the model's


def test_a_mark_left_after_a_drop_is_cleared_while_the_dropped_bytes_stay_in_the_file(
    instructions_home, tmp_path,
):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    _rewrite(root, CHANGED)
    assert h.section(WINDOW)[1].cause == "harness_written"
    h.mark_pending()  # one later named read
    assert set(_stored_marks(root)) == {"s1"}
    assert h.section(WINDOW) == (None, None)
    assert "pending" not in _record_of(root)
    _rewrite(root, "the person's fix-up\n")  # no mark stands: this is the person's change
    text, notice = h.section(WINDOW)
    assert text == "the person's fix-up\n"
    assert notice is not None and notice.cause == "adopted"


def test_a_dropped_start_line_file_then_a_hand_edit_is_the_persons_change(
    instructions_home, tmp_path,
):
    """WR-05 it2: the awaiting-step drop kept the mark, so the person's fix-up of the dropped
    bytes was called "changed by the model" again."""
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    assert h.section(WINDOW)[0] == TEXT  # the start line is owed and never delivered
    h.mark_pending()
    _rewrite(root, CHANGED)  # a harness call changed the bytes the line described
    text, notice = h.section(WINDOW)
    assert text is None and notice is not None and notice.cause == "harness_written"
    assert h.state.state == "harness_written"
    assert "pending" not in load_store().get("roots", {}).get(str(root.resolve()), {}).get(
        "instructions", {}
    )
    assert h.section(WINDOW) == (None, None)  # the dropped bytes stay out, silently
    _rewrite(root, "the person's fix-up\n")  # no mark stands: this is the person's change
    assert h.section(WINDOW) == ("the person's fix-up\n", None)
    assert h.state.state != "harness_written"


def test_a_second_model_write_after_the_drop_is_dropped_again(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _open(root)
    assert h.section(WINDOW)[0] == TEXT
    h.mark_pending()
    _rewrite(root, CHANGED)
    assert h.section(WINDOW)[1].cause == "harness_written"
    h.mark_pending()  # the model names the file again
    _rewrite(root, "yet another\n")
    text, notice = h.section(WINDOW)
    assert text is None and notice is not None and notice.cause == "harness_written"


def test_another_live_sessions_mark_survives_the_clearing(instructions_home, tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": TEXT})
    h = _recorded(root)
    h.mark_pending()
    _rewrite(root, CHANGED)
    h.section(WINDOW)
    with live_child() as pid:
        register_session("s2", pid=pid)
        _put_mark(root, "s2")
        h.mark_pending()
        assert h.section(WINDOW) == (None, None)
        assert set(_stored_marks(root)) == {"s2"}
