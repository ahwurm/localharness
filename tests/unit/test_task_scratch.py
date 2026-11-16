"""A task's scratch: its folder, its removal, and every line that says so with its count."""
import os
import time

import pytest

from localharness.core.run_folder import SCRATCH_SENTENCE, begin, end
from localharness.orchestrator.task_flow import runner
from localharness.orchestrator.task_flow.filer import Docs
from localharness.orchestrator.task_flow.record import scratch_clause
from localharness.tools.builtin.task_tool import TaskTool
from tests.unit.test_repl_task import command, staged
from tests.unit.test_task_flow_record import TYPED, make_flow

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-relative scratch")


def put(flow, task, n):
    folder = flow.docs.scratch(task)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"f{i}.txt").write_text("x")


def gone(path):
    for _ in range(200):
        if not path.exists():
            return True
        time.sleep(0.01)
    return False


def test_scratch_is_a_folder_under_the_root_made_private_and_no_scratchpad_is_written(tmp_path):
    docs = Docs(tmp_path / "state", True, "a", tmp_path / "scratch")
    assert docs.scratch(3) == tmp_path / "scratch" / "t3"
    flow, *_ = make_flow(tmp_path)
    flow.open("x", 1)
    assert flow.docs.scratch(1).stat().st_mode & 0o777 == 0o700
    assert not (flow.docs.state_dir / "scratchpad").exists()


def test_a_linked_scratch_root_is_refused_and_nothing_is_made_behind_it(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "scratch").symlink_to(tmp_path / "real")
    flow, *_ = make_flow(tmp_path)
    flow.open("x", 1)
    assert list((tmp_path / "real").iterdir()) == []


def test_no_scratch_root_makes_no_folder_and_the_notes_say_the_project(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.docs.scratch_root = None
    flow.open("x", 1)
    assert flow.docs.scratch(1) is None and not (tmp_path / "scratch").exists()
    assert runner.scratch_line(flow) == runner.NO_SCRATCH
    assert runner.NO_SCRATCH in TaskTool(flow).info().description and SCRATCH_SENTENCE not in TaskTool(flow).info().description


def test_the_notes_the_briefs_and_the_tool_carry_the_path_and_the_one_sentence(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.open("x", 1)
    where = runner.scratch_line(flow)
    assert str(tmp_path / "scratch" / "t1") in where and SCRATCH_SENTENCE in where and where in flow.note()
    assert SCRATCH_SENTENCE in TaskTool(flow).info().description


def test_the_count_is_taken_once_a_turn_and_caps_at_a_hundred_and_more(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.open("x", 1)
    put(flow, 1, 3)
    assert flow.scratch_count(1) == 3
    put(flow, 1, 5)
    assert flow.scratch_count(1) == 3
    flow.end_turn()
    assert flow.scratch_count(1) == 5
    put(flow, 1, 150)
    flow.end_turn()
    assert flow.scratch_count(1) == 101 and scratch_clause(101) == "100+ scratch files"


def test_the_paused_row_and_the_next_task_gloss_carry_the_count_and_a_zero_leaves_the_clause_out(tmp_path):
    flow = staged(tmp_path, decisions="pass", plan="pass")
    assert command(flow, "new").lines == ("◆ t1 open and not blocked · /task plan (revises) · new <objective> (closes t1)",)
    flow.end_turn()
    put(flow, 1, 12)
    assert command(flow, "new").lines == (
        "◆ t1 open and not blocked · /task plan (revises) · new <objective> (closes t1; its 12 scratch files go)",)
    flow.record.paused = True
    assert "clear (deletes t1 and its 12 scratch files)" in command(flow, "").lines[1]
    flow.record.paused = True
    flow.end_turn()
    put(flow, 1, 0)
    for f in flow.docs.scratch(1).iterdir():
        f.unlink()
    assert "clear (deletes t1)" in command(flow, "").lines[1]


def test_new_over_an_open_task_closes_it_and_removes_its_scratch_with_the_count(tmp_path):
    flow = staged(tmp_path, decisions="pass")
    put(flow, 1, 12)
    old = flow.docs.scratch(1)
    lines = command(flow, "new next one").lines
    assert lines[0].endswith("not resumable, docs kept; 12 scratch files removed")
    assert gone(old) and flow.docs.scratch(2).is_dir()


def test_new_over_an_open_task_with_no_doc_and_no_scratch_keeps_the_plain_closed_line(tmp_path):
    flow = staged(tmp_path)
    assert command(flow, "new next").lines[0] == "◆ t1 closed · not resumable, docs kept"


def test_new_over_a_done_task_says_so_with_the_count(tmp_path):
    flow = staged(tmp_path, decisions="pass")
    flow.record.stage = "done"
    put(flow, 1, 12)
    lines = command(flow, "new next").lines
    assert lines[0] == "◆ t1 done · 12 scratch files removed" and gone(flow.docs.scratch(1) or tmp_path)


def test_a_done_task_row_says_how_long_the_scratch_stays_and_a_zero_says_nothing(tmp_path):
    flow = staged(tmp_path, decisions="pass")
    flow.record.stage = "done"
    assert command(flow, "").lines[1] == "◆ t1 done · /task show review"
    flow.end_turn()
    put(flow, 1, 12)
    assert command(flow, "").lines[1].endswith(
        "12 scratch files stay until the next task, /task clear or the session ends; ask for a copy to keep them")


def test_clear_removes_the_docs_and_the_scratch_and_names_both_counts(tmp_path):
    flow = staged(tmp_path, decisions="pass")
    put(flow, 1, 12)
    (flow.docs.state_dir / "agents" / "a" / "tasks" / "t1" / "decisions.md").write_text("d")
    scratch = flow.docs.scratch(1)
    line = command(flow, "clear").lines[0]
    assert line.endswith("t1 (1 doc, 12 scratch files)") and gone(scratch)
    flow = staged(tmp_path / "b")
    assert command(flow, "clear").lines[0].endswith("t1 (0 docs)")


def test_the_model_opening_over_a_task_removes_its_scratch_and_says_so(tmp_path):
    flow, _, rows, _ = make_flow(tmp_path, messages=TYPED)
    flow.turn = lambda: 1
    flow.open("old", 0)
    put(flow, 1, 12)
    old = flow.docs.scratch(1)
    flow.open_for_model()
    assert rows[-1][1] == "opened by the model" and rows[-1][3].startswith("12 scratch files removed") and gone(old)


def test_the_session_end_removes_the_scratch_a_task_made(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "config.yaml").write_text("{}")
    run = begin(cfg)
    flow, *_ = make_flow(tmp_path)
    flow.docs.scratch_root = run.scratch
    flow.open("x", 1)
    put(flow, 1, 3)
    end(run)
    assert not run.path.exists()


def test_a_scratch_that_the_remover_cannot_open_is_no_error(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.docs.scratch_root = tmp_path / "missing"
    assert flow.drop_scratch(1) == 0
