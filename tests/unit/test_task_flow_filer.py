"""The filer's walk, task folders, reads and writes: no link followed, no folder adopted but its maker's."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from localharness.core.caps import Caps
from localharness.orchestrator.task_flow import filer
from localharness.orchestrator.task_flow.filer import SESSION_DOCS, TRUNCATION_MARKER, Docs, Refused, digest
from localharness.orchestrator.task_flow.record import file_doc, open_task

BOUND = Caps.DOC + len(TRUNCATION_MARKER)


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


def tree(tmp_path: Path) -> tuple[Path, Path]:
    root = state(tmp_path)
    (root / "agents" / "a" / "tasks" / "t1").mkdir(parents=True)
    (root / "agents" / "a" / "sessions").mkdir()
    return root, root / "agents" / "a"


def snap(top: Path) -> dict[str, bytes | None]:
    return {str(p.relative_to(top)): p.read_bytes() if p.is_file() else None for p in sorted(top.rglob("*"))}


def link(tmp_path: Path, component: str) -> tuple[Path, Path]:
    """The state tree with `component` replaced by a symlink to where it really lives."""
    root, agent = tree(tmp_path)
    at = {"root": root, "agents": root / "agents", "a": agent, "tasks": agent / "tasks",
          "t1": agent / "tasks" / "t1", "sessions": agent / "sessions"}[component]
    outside = tmp_path / "outside"
    at.rename(outside)
    os.symlink(outside, at)
    return root, outside


def act(docs: Docs, op: str) -> object:
    docs.adopted.add(1)
    return {"new_task": docs.new_task, "write": lambda: docs.write(1, "plan.md", "x"),
            "read": lambda: docs.read(1, "plan.md"), "session": lambda: docs.write_session("jtbd.md", "x")}[op]()


WORKSPACE = [("new_task", c) for c in ("root", "agents", "a", "tasks")] \
    + [(op, c) for op in ("write", "read") for c in ("root", "agents", "a", "tasks", "t1")] \
    + [("session", c) for c in ("root", "agents", "a", "sessions")]


@pytest.mark.parametrize(("op", "component"), WORKSPACE)
def test_workspace_layer_refuses_a_symlink_at_every_component(tmp_path, op, component):
    root, outside = link(tmp_path, component)
    before = snap(outside)
    with pytest.raises(Refused):
        act(Docs(root, True, "a"), op)
    assert snap(outside) == before


@pytest.mark.parametrize(("op", "component"), [w for w in WORKSPACE if w[1] not in ("root", "agents")])
def test_machine_layer_refuses_a_symlink_below_agents(tmp_path, op, component):
    root, outside = link(tmp_path, component)
    before = snap(outside)
    with pytest.raises(Refused):
        act(Docs(root, False, "a"), op)
    assert snap(outside) == before


@pytest.mark.parametrize("component", ["root", "agents"])
def test_machine_layer_follows_a_linked_root_and_a_linked_agents_folder(tmp_path, component):
    root, outside = link(tmp_path, component)
    docs = Docs(root, False, "a")
    assert docs.new_task() == 2
    docs.write(2, "plan.md", "p")
    assert (next(outside.rglob("t2")) / "plan.md").read_text() == "p"
    assert docs.read(2, "plan.md") == "p"


@pytest.mark.parametrize("agent", ["a/b", "../a", "", ".", "..", "a\x00"])
def test_an_agent_name_that_is_not_one_component_is_refused(tmp_path, agent):
    with pytest.raises(Refused):
        Docs(state(tmp_path), True, agent)


def test_eexist_retries_past_the_new_highest(tmp_path, monkeypatch):
    root, agent = tree(tmp_path)
    (agent / "tasks" / "t2").mkdir()
    real, seen = filer._listing, []

    def once(fd):
        seen.append(fd)
        return ["t1"] if len(seen) == 1 else real(fd)

    monkeypatch.setattr(filer, "_listing", once)
    docs = Docs(root, True, "a")
    assert docs.new_task() == 3
    assert (agent / "tasks" / "t3").is_dir()


def test_eexist_gives_up_when_every_folder_is_taken(tmp_path, monkeypatch):
    root, agent = tree(tmp_path)
    for n in range(2, 10):
        (agent / "tasks" / f"t{n}").mkdir()
    monkeypatch.setattr(filer, "_highest", lambda fd: 0)
    with pytest.raises(Refused):
        Docs(root, True, "a").new_task()


def test_adopt_only_the_folders_this_session_made(tmp_path):
    root, agent = tree(tmp_path)
    (agent / "tasks" / "t5").mkdir()
    (agent / "tasks" / "t5" / "plan.md").write_text("theirs")
    docs = Docs(root, True, "a")
    assert docs.new_task() == 6
    assert docs.adopted == {6}
    with pytest.raises(Refused):
        docs.write(5, "plan.md", "mine")
    assert docs.read(5, "plan.md") == "theirs"
    assert (agent / "tasks" / "t5" / "plan.md").read_text() == "theirs"


def test_a_fifo_doc_is_refused_without_blocking(tmp_path):
    root, agent = tree(tmp_path)
    os.mkfifo(agent / "tasks" / "t1" / "step-1.md")
    with pytest.raises(Refused):
        Docs(root, True, "a").read(1, "step-1.md")


def test_a_symlinked_doc_is_refused_and_a_symlinked_session_doc_too(tmp_path):
    root, agent = tree(tmp_path)
    victim = tmp_path / "victim"
    victim.write_text("secret")
    os.symlink(victim, agent / "tasks" / "t1" / "step-1.md")
    os.symlink(victim, agent / "sessions" / "jtbd.md")
    docs = Docs(root, True, "a")
    with pytest.raises(Refused):
        docs.read(1, "step-1.md")
    with pytest.raises(Refused):
        docs.read_session("jtbd.md")


def test_a_doc_one_byte_past_the_bound_is_stale_and_one_at_it_reads(tmp_path):
    root, agent = tree(tmp_path)
    docs = Docs(root, True, "a")
    (agent / "tasks" / "t1" / "plan.md").write_text("x" * (BOUND + 1))
    assert docs.read(1, "plan.md") is None
    (agent / "tasks" / "t1" / "plan.md").write_text("x" * BOUND)
    assert docs.read(1, "plan.md") == "x" * BOUND
    assert docs.read(1, "research.md") is None


def test_a_session_doc_past_the_bound_or_missing_reads_as_empty(tmp_path):
    root, agent = tree(tmp_path)
    docs = Docs(root, True, "a")
    assert docs.read_session("jtbd.md") == ""
    (agent / "sessions" / "jtbd.md").write_text("x" * (BOUND + 1))
    assert docs.read_session("jtbd.md") == ""
    (agent / "sessions" / "jtbd.md").write_bytes(b"\xff\xfe")
    with pytest.raises(Refused):
        docs.read_session("jtbd.md")
    assert Docs(root, True, "b").read_session("jtbd.md") == ""


def test_a_doc_over_the_cap_is_truncated_with_the_marker_and_still_reads(tmp_path):
    root, agent = tree(tmp_path)
    docs = Docs(root, True, "a")
    docs.adopted.add(1)
    got, truncated = docs.write(1, "plan.md", "x" * (Caps.DOC + 100))
    text = docs.read(1, "plan.md")
    assert truncated and text.endswith(TRUNCATION_MARKER) and digest(text) == got
    assert len(text) == Caps.DOC + len(TRUNCATION_MARKER)
    got, truncated = docs.write(1, "plan.md", "é" * Caps.DOC)
    text = docs.read(1, "plan.md")
    assert truncated and text.endswith(TRUNCATION_MARKER) and len(text.encode()) <= BOUND
    assert docs.write(1, "plan.md", "x" * Caps.DOC)[1] is False


def test_a_planted_link_at_a_doc_is_replaced_never_followed(tmp_path):
    root, agent = tree(tmp_path)
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    os.symlink(victim, agent / "tasks" / "t1" / "plan.md")
    os.symlink(victim, agent / "sessions" / "reply-review.md")
    docs = Docs(root, True, "a")
    docs.adopted.add(1)
    docs.write(1, "plan.md", "mine")
    docs.write_session("reply-review.md", "section")
    assert not (agent / "tasks" / "t1" / "plan.md").is_symlink()
    assert (agent / "tasks" / "t1" / "plan.md").read_text() == "mine"
    assert (agent / "sessions" / "reply-review.md").read_text() == "section"
    assert victim.read_text() == "untouched"


def test_session_docs_are_a_fixed_pair_and_reset_to_empty(tmp_path):
    root, agent = tree(tmp_path)
    docs = Docs(root, True, "a")
    assert SESSION_DOCS == ("jtbd.md", "reply-review.md")
    with pytest.raises(Refused):
        docs.write_session("other.md", "x")
    with pytest.raises(Refused):
        docs.read_session("other.md")
    docs.write_session("jtbd.md", "the job")
    assert docs.read_session("jtbd.md") == "the job"
    docs.reset_sessions()
    assert [(agent / "sessions" / n).read_text() for n in SESSION_DOCS] == ["", ""]
    assert sorted(p.name for p in (agent / "sessions").iterdir()) == sorted(SESSION_DOCS)


def test_a_sessions_folder_is_made_when_missing(tmp_path):
    root = state(tmp_path)
    docs = Docs(root, True, "a")
    docs.write_session("jtbd.md", "x")
    assert stat.S_IMODE((root / "agents" / "a" / "sessions").stat().st_mode) == 0o700


@pytest.mark.parametrize("name", ["../plan.md", "notes.md", "plan", "plan.md/x", "review-review.md", "step-0.md", ""])
def test_a_name_that_is_not_a_task_doc_is_refused_on_both_sides(tmp_path, name):
    root, agent = tree(tmp_path)
    docs = Docs(root, True, "a")
    docs.adopted.add(1)
    with pytest.raises(Refused):
        docs.write(1, name, "x")
    with pytest.raises(Refused):
        docs.read(1, name)


@pytest.mark.parametrize("name", ["decisions.md", "research-review.md", "plan-review.md", "step-1.md", "step-12-review.md", "review.md"])
def test_every_task_doc_name_is_accepted(tmp_path, name):
    root, agent = tree(tmp_path)
    docs = Docs(root, True, "a")
    docs.adopted.add(1)
    docs.write(1, name, "x")
    assert docs.read(1, name) == "x"


@pytest.mark.parametrize("task", [-1, 0.5, True])
def test_a_task_that_is_not_a_count_is_refused(tmp_path, task):
    root, agent = tree(tmp_path)
    with pytest.raises(Refused):
        Docs(root, True, "a").read(task, "plan.md")


def test_verified_is_false_for_no_row_a_refused_read_and_a_stale_doc(tmp_path):
    root, agent = tree(tmp_path)
    docs = Docs(root, True, "a")
    record = open_task(1, "o", 1)
    key = ("plan.md", None)
    assert not docs.verified(record, key)
    file_doc(record, "plan", key, "0" * 64)
    assert not docs.verified(record, key)
    os.mkfifo(agent / "tasks" / "t1" / "plan.md")
    assert not docs.verified(record, key)
    os.unlink(agent / "tasks" / "t1" / "plan.md")
    big = "x" * (BOUND + 1)
    (agent / "tasks" / "t1" / "plan.md").write_text(big)
    record.filed[key].digest = digest(big)
    assert not docs.verified(record, key)


def test_path_names_the_doc_without_opening_it(tmp_path):
    root, agent = tree(tmp_path)
    assert Docs(root, True, "a").path(1, "plan.md") == agent / "tasks" / "t1" / "plan.md"
