"""The filer's walk, task folders, reads and writes: no link followed, no folder adopted but its maker's."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from localharness.orchestrator.task_flow.filer import Docs, Refused, digest
from localharness.orchestrator.task_flow.record import file_doc, open_task


def state(tmp_path: Path) -> Path:
    root = tmp_path / "proj" / ".localharness"
    root.mkdir(parents=True)
    return root


def test_tracer_a_doc_filed_by_descriptor_is_verified_and_a_hand_edit_is_caught(tmp_path):
    docs = Docs(state(tmp_path), True, "a")
    assert docs.new_task() == 1
    folder = docs.state_dir / "agents" / "a" / "tasks" / "t1"
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    text = "objective · turn 1"
    got, truncated = docs.write(1, "decisions.md", text)
    assert (got, truncated) == (digest(text), False)
    assert stat.S_IMODE((folder / "decisions.md").stat().st_mode) == 0o600
    record = open_task(1, "objective", 1)
    file_doc(record, "new", ("decisions.md", None), got)
    assert docs.read(1, "decisions.md") == text
    assert docs.verified(record, ("decisions.md", None))
    (folder / "decisions.md").write_text("edited by hand")
    assert not docs.verified(record, ("decisions.md", None))


def test_tracer_a_linked_tasks_folder_is_refused_and_its_target_left_empty(tmp_path):
    root, elsewhere = state(tmp_path), tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "agents" / "a").mkdir(parents=True)
    os.symlink(elsewhere, root / "agents" / "a" / "tasks")
    with pytest.raises(Refused):
        Docs(root, True, "a").new_task()
    assert list(elsewhere.iterdir()) == []


def test_tracer_an_unfiled_key_and_an_unadopted_task_are_not_trusted(tmp_path):
    docs = Docs(state(tmp_path), True, "a")
    docs.new_task()
    record = open_task(1, "objective", 1)
    docs.write(1, "plan.md", "p")
    assert not docs.verified(record, ("plan.md", None))
    with pytest.raises(Refused):
        docs.write(2, "plan.md", "p")


def test_tracer_a_second_task_is_one_past_the_highest(tmp_path):
    docs = Docs(state(tmp_path), True, "a")
    assert [docs.new_task(), docs.new_task()] == [1, 2]
    assert docs.adopted == {1, 2}
