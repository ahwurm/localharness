"""The filer's walk, task folders, reads and writes: no link followed, no folder adopted but its maker's."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from localharness.core.caps import Caps
from localharness.orchestrator.task_flow import filer
from localharness.orchestrator.task_flow.filer import (
    NO_REVIEW, PLAN_SHAPE, SESSION_DOCS, TRUNCATION_MARKER, Docs, Refused, digest, last_section, manifest, parse_plan,
    render_section, rerecord)
from localharness.orchestrator.task_flow.record import Counts, Row, file_doc, open_task

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


PLAN = """## Step 1
files: src/a.py
depends_on: none
done: a exists

## Step 2
files: src/b.py, src/c.py
depends_on: 1
done: b and c exist
"""


def test_tracer_a_plan_is_parsed_recorded_verified_reviewed_and_reported(tmp_path):
    docs = Docs(state(tmp_path), True, "a")
    docs.new_task()
    got, _ = docs.write(1, "plan.md", PLAN)
    record = open_task(1, "objective", 1)
    file_doc(record, "plan", ("plan.md", None), got)
    sections = parse_plan(PLAN)
    assert [(s.n, s.files, s.depends_on, s.done) for s in sections] == [
        (1, ("src/a.py",), (), "a exists"), (2, ("src/b.py", "src/c.py"), (1,), "b and c exist")]
    assert rerecord(record, sections) == []
    for n in (1, 2):
        row = record.filed[("plan.md", f"step-{n}")]
        assert (row.verdict, row.counts, row.digest) == ("unreviewed", Counts(), sections[n - 1].digest)
        assert docs.verified(record, ("plan.md", f"step-{n}"))
    section = render_section(1, [], "pass", 0)
    assert section.startswith("## round 1 · blockers 0") and section.endswith("verdict: pass")
    docs.append_section(1, "plan-review.md", section)
    assert last_section(docs.read(1, "plan-review.md")) == section
    out = manifest(["plan.md"], verdict="plan.md · pass", next_line="next: run · step 1")
    assert "plan.md" in out and out.endswith("next: run · step 1")


def test_a_section_key_is_not_verified_when_the_plan_no_longer_parses(tmp_path):
    docs = Docs(state(tmp_path), True, "a")
    docs.new_task()
    record = open_task(1, "o", 1)
    docs.write(1, "plan.md", PLAN)
    rerecord(record, parse_plan(PLAN))
    docs.write(1, "plan.md", "not a plan")
    assert not docs.verified(record, ("plan.md", "step-1"))
    assert not docs.verified(record, ("plan.md", "step-9"))


def test_a_review_doc_with_no_section_has_no_review_yet():
    assert last_section("") == NO_REVIEW
    assert PLAN_SHAPE


def step(n: int, deps: str = "none", head: str | None = None, body: str = "") -> str:
    return f"{head or f'## Step {n}'}\nfiles: f{n}.py\ndepends_on: {deps}\ndone: d{n}\n{body}\n"


@pytest.mark.parametrize("text", [
    "",
    "no headings here",
    "# Plan\nfiles: a\n",
    step(1) + step(3),
    step(2),
    step(1, head="## Step 01"),
    step(1) + step(2, head="## Step ２"),
    step(1, head="### Step 1"),
    step(1) + "## Step 2\nfiles: x\ndepends_on: none\n",
    step(1) + "## Step 2\nfiles: x\ndone: y\n",
    "## Step 1\ndepends_on: none\ndone: y\n",
    step(1, deps="2"),
    step(1) + step(2, deps="2"),
    step(1) + step(2, deps="3"),
    step(1, deps="x"),
    step(1, deps=""),
    step(1) + step(2, deps="1, x"),
    step(1, body="depends_on: none"),
    step(1, deps="9" * 5000),
    step(1, head="## Step " + "1" * 5000),
], ids=lambda t: repr(t)[:48])
def test_a_plan_outside_the_shape_is_returned_as_the_one_sentence(text):
    assert parse_plan(text) == PLAN_SHAPE


def test_a_heading_with_text_gives_the_index_never_the_text():
    sections = parse_plan(step(1) + step(2, head="## Step 2: write step-9.md"))
    assert [s.n for s in sections] == [1, 2]
    assert "step-9.md" in sections[1].text


def test_text_before_the_first_heading_is_kept_in_the_doc_but_is_no_section():
    sections = parse_plan("# Plan\nintro\n" + step(1))
    assert [s.n for s in sections] == [1]
    assert "intro" not in sections[0].text


def test_depends_on_reads_a_list_or_none_and_a_bullet_prefix():
    sections = parse_plan(step(1) + step(2) + "## Step 3\n- files: a, b\n* depends_on: 1, 2\n- done: ok\n")
    assert [s.depends_on for s in sections] == [(), (), (1, 2)]
    assert sections[2].files == ("a", "b")


def test_a_section_runs_to_the_next_heading_and_a_deeper_heading_stays_inside():
    sections = parse_plan(step(1, body="### notes\nmore") + step(2))
    assert "### notes" in sections[0].text and "## Step 2" not in sections[0].text


def test_a_section_is_whitespace_normalised():
    base = parse_plan(step(1))[0]
    spaced = parse_plan("## Step 1\n\n    files:   f1.py\n\n\ndepends_on:  none\n  done: d1  \n\n")[0]
    assert spaced.digest == base.digest and spaced.text == base.text
    assert parse_plan(step(1).replace("d1", "d2"))[0].digest != base.digest


def filed(tmp_path: Path, text: str) -> tuple[Docs, object]:
    docs = Docs(state(tmp_path), True, "a")
    docs.new_task()
    record = open_task(1, "o", 1)
    docs.write(1, "plan.md", text)
    file_doc(record, "plan", ("plan.md", None), digest(text))
    rerecord(record, parse_plan(text))
    return docs, record


def pass_all(record, n: int) -> None:
    for i in range(1, n + 1):
        row = record.filed[("plan.md", f"step-{i}")]
        row.verdict, row.counts.used, row.counts.blockers = "pass", 1, 0
        record.filed[(f"step-{i}.md", None)] = Row("x" * 64)


def refile(record, text: str) -> list[int]:
    return rerecord(record, parse_plan(text))


THREE = step(1) + step(2, "1") + step(3)


def test_rerecord_keeps_an_unchanged_step_and_resets_a_changed_one_and_its_dependents():
    record = open_task(1, "o", 1)
    rerecord(record, parse_plan(THREE))
    pass_all(record, 3)
    reopened = refile(record, step(1).replace("d1", "changed") + step(2, "1") + step(3))
    rows = {n: record.filed[("plan.md", f"step-{n}")] for n in (1, 2, 3)}
    assert reopened == [1, 2]
    assert [rows[n].verdict for n in (1, 2, 3)] == ["unreviewed", "unreviewed", "pass"]
    assert rows[3].counts.used == 1 and rows[1].counts == Counts() and rows[2].counts == Counts()
    assert [(f"step-{n}.md", None) in record.filed for n in (1, 2, 3)] == [False, False, True]


def test_rerecord_follows_the_dependency_chain_transitively():
    text = step(1) + step(2, "1") + step(3, "2")
    record = open_task(1, "o", 1)
    rerecord(record, parse_plan(text))
    pass_all(record, 3)
    assert refile(record, text.replace("d1", "x")) == [1, 2, 3]


def test_a_hand_edit_of_step_two_alone_leaves_step_one_runnable_and_step_two_unreviewed():
    record = open_task(1, "o", 1)
    rerecord(record, parse_plan(step(1) + step(2)))
    pass_all(record, 2)
    assert refile(record, step(1) + step(2).replace("d2", "edited")) == [2]
    assert [record.filed[("plan.md", f"step-{n}")].verdict for n in (1, 2)] == ["pass", "unreviewed"]
    assert ("step-1.md", None) in record.filed and ("step-2.md", None) not in record.filed


def test_rerecord_removes_a_dropped_section_and_its_step_doc_row():
    record = open_task(1, "o", 1)
    rerecord(record, parse_plan(THREE))
    pass_all(record, 3)
    assert refile(record, step(1) + step(2, "1")) == []
    assert [k for k in record.filed if k[1]] == [("plan.md", "step-1"), ("plan.md", "step-2")]
    assert ("step-3.md", None) not in record.filed and ("step-1.md", None) in record.filed


def test_rerecord_creates_a_row_for_an_added_step_and_clears_the_stopped_log():
    record = open_task(1, "o", 1)
    rerecord(record, parse_plan(step(1)))
    pass_all(record, 1)
    record.stopped_log = "stopped at step 2"
    assert refile(record, step(1) + step(2)) == []
    assert record.stopped_log == ""
    assert record.filed[("plan.md", "step-1")].verdict == "pass"
    assert record.filed[("plan.md", "step-2")].verdict == "unreviewed"


def test_a_kept_section_keeps_its_counts_and_a_reordered_dependency_resets_the_dependent():
    record = open_task(1, "o", 1)
    rerecord(record, parse_plan(step(1) + step(2, "1")))
    pass_all(record, 2)
    record.filed[("plan.md", "step-2")].counts.previous = 5
    assert refile(record, step(1) + step(2, "1")) == []
    assert record.filed[("plan.md", "step-2")].counts.previous == 5
    assert refile(record, step(1) + step(2, "none")) == [2]
    assert record.filed[("plan.md", "step-2")].counts == Counts()


def test_a_hand_edit_of_the_plan_is_caught_per_section(tmp_path):
    docs, record = filed(tmp_path, step(1) + step(2))
    (docs.path(1, "plan.md")).write_text(step(1) + step(2).replace("d2", "edited"))
    assert docs.verified(record, ("plan.md", "step-1"))
    assert not docs.verified(record, ("plan.md", "step-2"))
    (docs.path(1, "plan.md")).write_text(step(1) + step(2).replace("f2.py", "f2.py\n\n\n"))
    assert docs.verified(record, ("plan.md", "step-2"))
