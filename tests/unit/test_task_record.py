"""The persisted task record: round-trip, substantiation, atomic private save, load, reconcile."""
import json
import logging
import os
import stat

import pytest

from localharness.agent.task_context import Receipt, Requirement, TaskContext
from localharness.agent.task_record import (
    CORRUPT_NOTICE, Decision, TaskRecord, TaskRecordTooLarge, TaskState, normalize_quote,
)


def make_record(tmp_path, **kw):
    ctx = TaskContext("Write a short report on tide pools", stop_boundary="outline only",
                      requested_status="checkpoint")
    return TaskRecord(context=ctx, workspace=str(tmp_path), assignment="Outline in outline.md", **kw)


def started(tmp_path, *turns):
    state = TaskState(tmp_path / "agents" / "a" / "task.json", workspace=str(tmp_path))
    for turn in turns:
        state.observe_human(turn)
    state.begin(make_record(tmp_path))
    state.save()
    return state


def test_round_trip_preserves_every_field(tmp_path):
    outline = tmp_path / "outline.md"
    rec = make_record(tmp_path, instructions_path=str(tmp_path / "SKILL.md"),
                      decisions=[Decision("Five sections", "human", 2), Decision("Plain tone", "model", 2)],
                      human_turns=["first", "second"], revision_budget=2, revisions_used=1,
                      delegation_budget=3, questions=["Which audience?"], next_action="draft")
    req = Requirement("lint", "Lint passes", "bash_exec", {"command": "lint"}, "exit_code", 0,
                      ("outline",), "2", "model")
    rec.context.revise(req)
    rec.context.artifacts["outline"] = outline
    rec.context.receipts["lint"] = Receipt("bash_exec", "c1", req.fingerprint(), {"outline": None}, "passed")
    rec.context.waivers["lint"] = (req.fingerprint(), "skip lint for now")
    rec.context.status = "checkpoint"
    back = TaskRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert back == rec
    assert back.context.requirements["lint"].dependencies == ("outline",)
    assert back.context.requirements["lint"].origin == "model"
    assert back.context.artifacts["outline"] == outline
    assert back.context.requirements["lint"].fingerprint() == req.fingerprint()
    data = rec.to_dict()
    data["format"] = 2
    with pytest.raises(ValueError):
        TaskRecord.from_dict(data)


def test_quotes_are_substantiated_only_by_runtime_human_turns(tmp_path):
    assert normalize_quote(' “Stop  after\nthe OUTLINE” ') == "stop after the outline"
    state = started(tmp_path, "Please stop after the outline for my review.")
    assert state.substantiated("stop  AFTER the outline")
    assert not state.substantiated("outline")  # under 8 normalized chars
    assert not state.substantiated(None)
    assert not state.substantiated("write the whole report now")
    # Model output never enters the turn list: only observe_human appends.
    state.record_result("write", {}, "c1", success=True, metadata={}, before={})
    assert state.current.human_turns == ["Please stop after the outline for my review."]


def test_observe_without_record_is_memory_only(tmp_path):
    state = TaskState(tmp_path / "task.json", workspace=str(tmp_path))
    for i in range(30):
        state.observe_human(f"turn {i} " + "x" * 3000)
    assert len(state.recent_turns) == 24
    assert all(len(t) <= 2000 for t in state.recent_turns)
    assert state.recent_turns[0].startswith("turn 6 ")
    assert not (tmp_path / "task.json").exists()
    assert state.packet() == ""
    assert state.finalize("Answer") == "Answer"
    assert state.revisions() == {}


def test_observe_with_record_forwards_mirrors_and_saves(tmp_path):
    state = started(tmp_path, "Start the outline please.")
    state.observe_human("Correction: five sections, not three.")
    assert state.current.context.latest_human == "Correction: five sections, not three."
    assert state.current.human_turns[-1] == "Correction: five sections, not three."
    on_disk = json.loads(state.path.read_text())
    assert on_disk["human_turns"][-1] == "Correction: five sections, not three."


def test_save_is_private_atomic_and_bounded(tmp_path):
    state = started(tmp_path, "Begin.")
    assert stat.S_IMODE(os.stat(state.path).st_mode) == 0o600
    assert not state.path.with_name("task.json.tmp").exists()
    state.current.human_turns = ["y" * 2000 for _ in range(40)]
    state.save()
    saved = json.loads(state.path.read_text())
    assert len(state.path.read_bytes()) <= 64 * 1024
    assert len(saved["human_turns"]) < 40
    state.current.next_action = "z" * (70 * 1024)
    with pytest.raises(TaskRecordTooLarge, match="Task record exceeds 64 KiB; narrow the record"):
        state.save()


def test_load_missing_corrupt_and_foreign(tmp_path, caplog):
    path = tmp_path / "task.json"
    state, notice = TaskState.load(path, workspace=str(tmp_path))
    assert state.current is None and notice is None and state.path == path
    for bad in (b"{not json", json.dumps({"format": 9}).encode(), json.dumps({"format": 1}).encode()):
        path.write_bytes(bad)
        with caplog.at_level(logging.WARNING):
            state, notice = TaskState.load(path, workspace=str(tmp_path))
        assert state.current is None and notice == CORRUPT_NOTICE
        assert not path.exists() and path.with_name("task.json.corrupt").exists()
    good = started(tmp_path)
    good.path.replace(path)
    state, notice = TaskState.load(path, workspace=str(tmp_path / "elsewhere"))
    assert state.current is None and notice is None and path.exists()
    state, notice = TaskState.load(path, workspace=str(tmp_path))
    assert state.current.id == good.current.id and notice is None


def test_reconcile_notes_changed_and_missing_artifacts(tmp_path):
    outline, notes = tmp_path / "outline.md", tmp_path / "notes.md"
    outline.write_text("# One")
    notes.write_text("n")
    state = started(tmp_path, "Begin the outline.")
    ctx = state.current.context
    ctx.artifacts.update(outline=outline, notes=notes)
    ctx.revise(Requirement("lint", "Lint", "check", {}, None, None, ("outline",)))
    state.record_result("check", {}, "c1", success=True, metadata={}, before=state.revisions())
    assert ctx.outcomes()["lint"] == "passed"
    state.save()
    loaded, _ = TaskState.load(state.path, workspace=str(tmp_path))
    assert loaded.notes == []
    outline.write_text("# Two")
    notes.unlink()
    loaded, _ = TaskState.load(state.path, workspace=str(tmp_path))
    assert "outline changed since last session" in loaded.notes
    assert "notes missing" in loaded.notes
    assert loaded.current.context.outcomes()["lint"] == "stale"
    assert "outline changed since last session" in loaded.packet()


def test_packet_contents_and_completion(tmp_path):
    state = started(tmp_path, "Stop after the outline for review.", "Correction: five sections.")
    state.current.decisions += [Decision("Five sections", "human", 2), Decision("Plain tone", "model", 2)]
    packet = state.packet()
    for part in (f"Task {state.current.id}", "tide pools", "Outline in outline.md",
                 "Accepted decisions: [human] Five sections; [assumption] Plain tone",
                 "Correction: five sections.", "Requested stopping status: checkpoint",
                 "Declared evidence"):
        assert part in packet
    state.current.context.status = "complete"
    assert state.packet() == ""


def test_finalize_closed_without_checks_retires_record(tmp_path):
    state = started(tmp_path, "Begin.")
    ctx = state.current.context
    ctx.requested_status = "complete"
    state.current.closed = True
    assert state.finalize("All done.") == "All done."
    assert ctx.status == "unknown"  # nothing machine-checkable was declared
    assert state.packet() == ""
    assert "closed (no machine checks declared)" in state.show()
    assert json.loads(state.path.read_text())["closed"] is True


def test_finalize_checkpoint_keeps_packet(tmp_path):
    state = started(tmp_path, "Begin.")
    state.current.closed = False
    assert state.finalize("Outline ready.") == "Outline ready."
    assert state.status == "checkpoint"
    assert state.packet() != ""
    state.status = "blocked"
    assert state.current.context.status == "blocked"
    empty = TaskState()
    empty.status = "blocked"
    assert empty.status == "none"


def test_show_and_clear(tmp_path):
    state = TaskState(tmp_path / "task.json", workspace=str(tmp_path))
    assert state.show() == "No task record."
    state = started(tmp_path, "Begin.")
    state.current.decisions.append(Decision("Five sections", "human", 1))
    text = state.show()
    assert state.current.id in text and "human: Five sections" in text and "revision 0/1" in text
    state.clear()
    assert state.current is None and not state.path.exists()
