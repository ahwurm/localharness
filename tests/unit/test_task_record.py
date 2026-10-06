"""The persisted task record: round-trip, numbered human turns, atomic private save, load, reconcile."""
import json
import logging
import os
import stat

import pytest

from localharness.agent.task_context import Receipt, Requirement, TaskContext
from localharness.agent.task_record import (
    CORRUPT_NOTICE, Decision, TaskRecord, TaskRecordTooLarge, TaskState,
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
                      decisions=[Decision("Five sections", "human", 2, "second"), Decision("Plain tone", "model")],
                      human_turns=[{"n": 1, "text": "first"}, {"n": 2, "text": "second"}],
                      artifact_history={"outline": ["a" * 64, "b" * 64]},
                      artifact_history_dropped={"outline": 3}, checkpoint_turn=1,
                      checkpoint_text="first", retired=True,
                      questions=["Which audience?"], next_action="draft")
    req = Requirement("lint", "Lint passes", "bash_exec", {"command": "lint"}, "exit_code", 0,
                      ("outline",), "2", "human", 2, "second")
    rec.context.revise(req)
    rec.context.artifacts["outline"] = outline
    rec.context.receipts["lint"] = Receipt("bash_exec", "c1", req.fingerprint(), {"outline": None}, "passed")
    rec.context.waivers["lint"] = (req.fingerprint(), "skip lint for now")
    rec.context.status = "checkpoint"
    back = TaskRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert back == rec
    assert back.context.requirements["lint"].dependencies == ("outline",)
    assert back.context.requirements["lint"].origin == "human"
    assert back.context.requirements["lint"].human_turn == 2
    assert back.context.artifacts["outline"] == outline
    assert back.context.requirements["lint"].fingerprint() == req.fingerprint()
    data = rec.to_dict()
    data["format"] = 2
    with pytest.raises(ValueError):
        TaskRecord.from_dict(data)


def test_human_turns_are_numbered_and_cited(tmp_path):
    state = started(tmp_path, "Please stop after the outline.", "Five sections.", "Plain tone.")
    assert [t["n"] for t in state.current.human_turns] == [1, 2, 3]
    assert state.cite(2) == {"n": 2, "text": "Five sections."}
    assert state.cite(9) is None and state.cite("2") is None
    # Model output never enters the turn list: only observe_human appends.
    state.record_result("write", {}, "c1", success=True, metadata={}, before={})
    assert len(state.current.human_turns) == 3
    for i in range(27):
        state.observe_human(f"more {i}")
    assert [t["n"] for t in state.current.human_turns] == list(range(7, 31))  # absolute, bounded 24
    assert state.cite(6) is None and state.cite(30)["text"] == "more 26"
    loaded, _ = TaskState.load(state.path, workspace=str(tmp_path))
    loaded.observe_human("after restart")
    assert loaded.current.human_turns[-1] == {"n": 31, "text": "after restart"}


def test_observe_without_record_is_memory_only(tmp_path):
    state = TaskState(tmp_path / "task.json", workspace=str(tmp_path))
    for i in range(30):
        state.observe_human(f"turn {i} " + "x" * 3000)
    assert len(state.recent_turns) == 24
    assert all(len(t["text"]) <= 2000 for t in state.recent_turns)
    assert state.recent_turns[0]["text"].startswith("turn 6 ") and state.recent_turns[0]["n"] == 7
    assert not (tmp_path / "task.json").exists()
    assert state.packet() == ""
    assert state.finalize("Answer") == "Answer"
    assert state.revisions() == {}


def test_observe_with_record_forwards_mirrors_and_saves(tmp_path):
    state = started(tmp_path, "Start the outline please.")
    state.observe_human("Correction: five sections, not three.")
    assert state.current.context.latest_human == "Correction: five sections, not three."
    assert state.current.human_turns[-1] == {"n": 2, "text": "Correction: five sections, not three."}
    on_disk = json.loads(state.path.read_text())
    assert on_disk["human_turns"][-1] == {"n": 2, "text": "Correction: five sections, not three."}


def test_save_is_private_atomic_and_bounded(tmp_path):
    state = started(tmp_path, "Begin.")
    assert stat.S_IMODE(os.stat(state.path).st_mode) == 0o600
    assert not state.path.with_name("task.json.tmp").exists()
    state.current.human_turns = [{"n": i, "text": "y" * 2000} for i in range(40)]
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
    assert "notes removed since last session" in loaded.notes
    assert loaded.current.context.outcomes()["lint"] == "stale"
    assert "outline changed since last session" in loaded.packet()
    notes.write_text("back")  # a saved None (unreadable at save) notes nothing, whatever it is now
    data = json.loads(state.path.read_text())
    data["artifact_revisions"] = {"notes": None}
    state.path.write_text(json.dumps(data))
    again, _ = TaskState.load(state.path, workspace=str(tmp_path))
    assert again.notes == []


def test_packet_contents_and_completion(tmp_path):
    state = started(tmp_path, "Stop after the outline for review.", "Correction: five sections.")
    state.current.decisions += [Decision("Five sections", "human", 2, "Correction: five sections."),
                                Decision("Plain tone", "model")]
    packet = state.packet()
    for part in (f"Task {state.current.id}", "tide pools", "Outline in outline.md",
                 'Accepted decisions: [human turn 2: "Correction: five sections."] Five sections; '
                 "[assumption] Plain tone", "Human turns on record: 1-2 (cite one with human_turn)",
                 "Correction: five sections.", "Requested stopping status: checkpoint",
                 "Declared evidence"):
        assert part in packet
    assert "Revision budget" not in packet
    state.current.decisions[0] = Decision("Long", "human", 1, "w" * 300)
    assert '"' + "w" * 200 + '…"' in state.packet()  # display clip; the stored text stays whole
    assert "w" * 300 in state.show()
    state.current.context.status = "complete"
    assert state.packet() != ""  # passing alone never retires; only a settled close does
    state.current.closed = True
    assert state.packet() == ""


def test_finalize_closed_without_checks_retires_record(tmp_path):
    state = started(tmp_path, "Begin.")
    ctx = state.current.context
    ctx.requested_status = "complete"
    state.current.closed = True
    assert state.finalize("All done.") == "All done."
    assert ctx.status == "unknown"  # nothing machine-checkable was declared
    assert state.packet() == "" and state.current.retired
    assert "closed (no machine checks declared)" in state.show()
    saved = json.loads(state.path.read_text())
    assert saved["closed"] is True and saved["retired"] is True


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
    state.current.decisions += [Decision("Five sections", "human", 1, "Begin."), Decision("Plain", "model")]
    text = state.show()
    assert state.current.id in text and 'human turn 1: "Begin." -> Five sections' in text
    assert "assumption: Plain" in text and "budget" not in text
    assert "human turns:\n  1: Begin." in text
    state.clear()
    assert state.current is None and not state.path.exists()


# --- 0.16.5 slice 2: references, judgments, artifact revisions ---

from localharness.agent.context import ContentStore  # noqa: E402
from localharness.agent.task_record import Judgment, Reference  # noqa: E402


def test_round_trip_with_references_and_judgments(tmp_path):
    rec = make_record(tmp_path, references=[Reference("voice", "voice.md", "abc", "f" * 64, "current"),
                                            Reference("pasted", handle="def")],
                      judgments=[Judgment("tone", "Plain tone", "ok", "para 2", "cut adverbs", "waived",
                                          'turn 4: "skip it"')])
    assert TaskRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec
    old = make_record(tmp_path).to_dict()
    for k in ("references", "judgments", "artifact_history", "retired"):
        old.pop(k)
    back = TaskRecord.from_dict(old)
    assert (back.references, back.judgments, back.artifact_history, back.retired) == ([], [], {}, False)


def refs_state(tmp_path, *refs):
    state = started(tmp_path, "Begin the draft.")
    state.current.references = list(refs)
    return state


def test_refresh_snapshots_redeclares_and_resnapshots_changes(tmp_path):
    voice = tmp_path / "voice.md"
    voice.write_text("Short sentences.")
    state, store = refs_state(tmp_path, Reference("voice", "voice.md")), ContentStore()
    state.refresh_references(store)
    ref = state.current.references[0]
    assert ref.status == "current" and store.get(ref.handle) == "Short sentences."
    assert store.active_step == f"task {state.current.id}"
    assert [r.source for r in store.active_references] == ["voice"]
    assert json.loads(state.path.read_text())["references"][0]["handle"] == ref.handle
    store.clear_active_references()  # the per-human-turn reset
    state.refresh_references(store)
    assert [r.handle for r in store.active_references] == [ref.handle] and ref.status == "current"
    first = ref.handle
    voice.write_text("Longer, winding sentences.")
    state.refresh_references(store)
    assert ref.status == "refreshed (changed)" and ref.handle != first
    assert store.get(ref.handle) == "Longer, winding sentences."
    assert [r.handle for r in store.active_references] == [ref.handle]
    state.refresh_references(store)
    assert ref.status == "current"


def test_refresh_problem_statuses_never_raise(tmp_path, monkeypatch):
    import localharness.agent.task_record as tr
    (tmp_path / "big.md").write_text("x" * 64)
    (tmp_path / "ok.md").write_text("fine")
    monkeypatch.setattr(tr, "MAX_REFERENCE_BYTES", 32)
    store = ContentStore()
    for i in range(4):
        store.declare_active_reference("model step", store.put(f"other {i}"), f"other{i}")
    state = refs_state(tmp_path, Reference("big", "big.md"), Reference("gone", handle="0123456789ab"),
                       Reference("lost", "lost.md"), Reference("ok", "ok.md"))
    state.refresh_references(store)
    big, gone, lost, ok = state.current.references
    assert big.status == "too large; narrow the reference" and big.handle is None
    assert gone.status == "unavailable; read it again"
    assert lost.status == "missing; the file could not be read"
    assert ok.status.startswith("unprotected: ") and "four references" in ok.status
    assert store.active_step == "model step"  # merged into the model-declared set, not replaced
    packet = state.packet()
    assert "References: big: too large; narrow the reference; gone: unavailable; read it again" in packet


def test_refresh_is_inert_without_a_live_record(tmp_path):
    (tmp_path / "v.md").write_text("v")
    empty = TaskState(tmp_path / "none.json", workspace=str(tmp_path))
    store = ContentStore()
    empty.refresh_references(store)
    assert not store.active_references and not (tmp_path / "none.json").exists()
    state = refs_state(tmp_path, Reference("v", "v.md"))
    state.refresh_references(None)
    assert state.current.references[0].handle is None
    state.current.closed = True
    state.refresh_references(store)
    assert not store.active_references and state.current.references[0].handle is None


def test_reconcile_nulls_handles_and_notes_reference_changes(tmp_path):
    (tmp_path / "voice.md").write_text("one")
    (tmp_path / "src.md").write_text("src")
    state = refs_state(tmp_path, Reference("voice", "voice.md"), Reference("src", "src.md"),
                       Reference("pasted", handle="abc"))
    store = ContentStore()
    store.put("pasted body")
    state.refresh_references(store)
    state.save()
    (tmp_path / "voice.md").write_text("two")
    (tmp_path / "src.md").unlink()
    loaded, _ = TaskState.load(state.path, workspace=str(tmp_path))
    assert [r.handle for r in loaded.current.references] == [None, None, None]
    assert loaded.current.references[2].status == "unavailable; read it again"
    assert "reference voice changed since last session" in loaded.notes
    assert "reference src removed since last session" in loaded.notes


def lint_state(tmp_path):
    draft = tmp_path / "draft.md"
    draft.write_text("v1")
    state = started(tmp_path, "Begin.")
    state.current.context.artifacts["draft"] = draft
    state.current.context.revise(
        Requirement("lint", "Lint", "bash_exec", {"command": "lint"}, "exit_code", 0, ("draft",)))
    state.save()

    def lint(code, call):
        state.record_result("bash_exec", {"command": "lint"}, call, success=True,
                            metadata={"exit_code": code}, before=state.revisions())

    def edit(text, call):
        before = state.revisions()
        draft.write_text(text)
        state.record_result("write", {"path": "draft.md"}, call, success=True, metadata={}, before=before)

    return state, lint, edit


def test_artifact_revisions_and_run_facts_in_packet(tmp_path):
    state, lint, edit = lint_state(tmp_path)
    assert "Artifacts: draft: rev 1" in state.packet()
    edit("v2", "c1")
    lint(1, "c2")
    packet = state.packet()
    assert "Artifacts: draft: rev 2" in packet and "lint: failed — Lint" in packet
    edit("v3", "c3")
    packet = state.packet()
    assert "Artifacts: draft: rev 3" in packet
    assert "lint: failed (ran at draft rev 2, draft now rev 3) — Lint" in packet
    assert "Revision budget" not in packet
    edit("v2", "c4")  # a revert is the same content: no new revision
    assert "Artifacts: draft: rev 2" in state.packet() and "lint: failed — Lint" in state.packet()
    for i in range(20):
        edit(f"w{i}", f"e{i}")
    rec = state.current
    assert len(rec.artifact_history["draft"]) == 16 and rec.artifact_history_dropped["draft"] == 7
    assert "Artifacts: draft: rev 23" in state.packet()  # numbering survives the cap
    loaded, _ = TaskState.load(state.path, workspace=str(tmp_path))
    assert "Artifacts: draft: rev 23" in loaded.packet()


def test_finalize_appends_one_evidence_line(tmp_path):
    state, lint, edit = lint_state(tmp_path)
    ctx = state.current.context
    ctx.requested_status = "complete"
    lint(1, "c1")
    assert state.finalize("4") == "4\n\nTask evidence: lint: failed."
    state.current.judgments = [Judgment("tone", "Plain tone")]  # opinion: never part of the line
    ctx.revise(Requirement("review", "Review", "bash_exec", {"command": "review"}))
    state.observe_human("Skip the review gate for this draft.")
    ctx.waive("review", human_decision='turn 2: "Skip the review gate for this draft."')
    assert state.finalize("Draft done.") == (
        'Draft done.\n\nTask evidence: lint: failed; review: waived (turn 2: '
        '"Skip the review gate for this draft.").')
    lint(0, "c2")
    del ctx.requirements["review"]
    assert state.finalize("Draft done.") == "Draft done."  # all passed: nothing appended
    bare = started(tmp_path, "Begin.")
    assert bare.finalize("Answer") == "Answer"  # nothing declared: nothing appended


def test_retire_rule(tmp_path):
    state, lint, edit = lint_state(tmp_path)
    rec = state.current
    rec.context.requested_status = "complete"
    lint(1, "c1")
    rec.closed = True
    assert state._live() is rec and state.finalize("Done.") == "Done.\n\nTask evidence: lint: failed."
    assert not rec.retired and state.packet() != ""  # closed with failed evidence stays live
    rec.context.waive("lint", human_decision='turn 1: "Begin."')
    assert state._live() is None
    assert state.finalize("Done.") == "Done." and rec.retired and state.status == "complete"
    edit("v9", "c2")
    assert state._live() is None and state.finalize("Later.") == "Later."  # latched
    other = started(tmp_path, "Begin.")
    other.begin_delegation("c1", "reviewer", "review")
    other.record_delegation("c1", ToolResult(output="ok", metadata={}))
    other.current.closed = True
    assert other._live() is other.current
    assert other.finalize("Done.") == "Done.\n\nTask evidence: reviewer (d1): completed, not integrated."
    assert not other.current.retired


def test_completed_record_does_not_rewrite_later_replies(tmp_path):
    state = started(tmp_path, "Skip the lint gate for this draft.")
    ctx = state.current.context
    ctx.requested_status = "complete"
    ctx.revise(Requirement("lint", "Lint", "bash_exec", {"command": "lint"}))
    ctx.waive("lint", human_decision='turn 1: "Skip the lint gate for this draft."')
    state.current.closed = True
    assert state.finalize("Done.") == "Done."  # closed and settled: retires, words unchanged
    assert state.status == "complete" and state.current.retired
    assert state.finalize("4") == "4"  # a later aside is not touched


def test_show_lists_references_and_judgments(tmp_path):
    state = refs_state(tmp_path, Reference("voice", "voice.md"), Reference("pasted", handle="abc"))
    state.current.judgments = [Judgment("tone", "Plain tone", assessment="mostly plain", fix="cut adverbs")]
    text = state.show()
    assert "  voice: voice.md (pending)" in text and "  pasted: handle abc (pending)" in text
    assert "  tone: open — Plain tone | assessment: mostly plain | fix: cut adverbs" in text


# --- 0.16.5 slice 3: the record owns delegations ------------------------------------------------

from localharness.agent.task_record import Delegation  # noqa: E402
from localharness.tools.base import ToolResult  # noqa: E402

def ok_result(output="SUBAGENT RUN COMPLETE ... tail words", **meta):
    return ToolResult(output=output, metadata={"delegated_to": "reviewer", **meta})


def test_delegations_round_trip_and_legacy_records_load_empty(tmp_path):
    rec = make_record(tmp_path, delegations=[Delegation("d1", "reviewer", "review", "completed",
                                                        ["/w/r.md"], "f", "u", "r", "used", "c1")])
    assert TaskRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec
    legacy = rec.to_dict()
    del legacy["delegations"]
    assert legacy["format"] == 1 and TaskRecord.from_dict(legacy).delegations == []


def test_begin_delegation_inert_without_record_and_never_refuses(tmp_path, caplog):
    bare = TaskState(tmp_path / "task.json", workspace=str(tmp_path))
    assert bare.begin_delegation("c1", "reviewer", "review") is None
    assert not (tmp_path / "task.json").exists()
    state = started(tmp_path)
    for i in range(1, 9):
        assert state.begin_delegation(f"c{i}", "writer", "draft") is None
    saved = json.loads(state.path.read_text())["delegations"]
    assert [(d["id"], d["status"], d["call_id"]) for d in saved[:2]] == [
        ("d1", "running", "c1"), ("d2", "running", "c2")]
    with caplog.at_level(logging.WARNING):
        assert state.begin_delegation("c9", "writer", "again") is None  # capacity: not recorded
    assert len(state.current.delegations) == 8 and "not recorded" in caplog.text
    state.current.delegations[1].integrated = "used"
    state.current.delegations[3].integrated = "used"
    state.begin_delegation("c10", "writer", "again")  # oldest integrated (d2) makes room
    assert [d.id for d in state.current.delegations] == ["d1", "d3", "d4", "d5", "d6", "d7", "d8", "d9"]


@pytest.mark.parametrize("result,status", [
    (ok_result(status="completed"), "completed"),
    (ok_result(), "completed"),
    (ok_result(status="budget_exhausted"), "budget_exhausted"),
    (ToolResult(output="", success=False, error="Agent 'reviewer' failed: boom",
                error_type="execution_error"), "failed: execution_error"),
    (ToolResult(output="", success=False, error="timed out", error_type="timeout_error"), "timeout"),
    (None, "failed: execution_error"),
])
def test_record_delegation_status_table(tmp_path, result, status):
    state = started(tmp_path)
    state.begin_delegation("c1", "reviewer", "review")
    state.record_delegation("c1", result)
    state.record_delegation("unknown", result)  # no entry for that call: no-op
    (d,) = state.current.delegations
    assert d.status == status
    if result is None or not result.success:
        assert d.findings == ((result.error or "")[:400] if result else "")


def test_record_delegation_uses_runtime_artifacts_and_handoff_text(tmp_path):
    state = started(tmp_path)
    state.begin_delegation("c1", "reviewer", "review")
    handoff = {"status": "completed", "artifacts": "claimed.md", "findings": "two claims",
               "uncertainties": "tone", "remaining": "none"}
    state.record_delegation("c1", ok_result(status="completed", artifacts=["/w/review.md"],
                                            handoff=handoff))
    (d,) = state.current.delegations
    assert (d.artifacts, d.findings, d.uncertainties, d.remaining) == (
        ["/w/review.md"], "two claims", "tone", "none")
    state.begin_delegation("c2", "writer", "draft")
    state.record_delegation("c2", ok_result(output="x" * 900 + "THE END", status="completed"))
    assert state.current.delegations[1].findings == ("x" * 900 + "THE END")[-400:]


def test_running_delegation_becomes_interrupted_on_human_turn_and_on_load(tmp_path):
    state = started(tmp_path)
    state.begin_delegation("c1", "reviewer", "review")
    loaded, _ = TaskState.load(state.path, str(tmp_path))
    assert loaded.current.delegations[0].status == "interrupted"
    assert "delegation d1 reviewer interrupted" in loaded.notes
    assert json.loads(state.path.read_text())["delegations"][0]["status"] == "interrupted"
    state.observe_human("carry on")
    assert state.current.delegations[0].status == "interrupted"
    assert "delegation d1 reviewer interrupted" in state.notes


def test_packet_lists_delegations_and_finalize_names_unresolved(tmp_path):
    state = started(tmp_path)
    state.current.context.requested_status = "complete"
    state.begin_delegation("c1", "reviewer", "review")
    state.record_delegation("c1", ok_result(status="completed"))
    state.current.delegations[0].integrated = "fixed both claims"
    state.begin_delegation("c2", "writer", "draft")
    state.record_delegation("c2", None)
    assert ("Delegations (2): d1 reviewer: completed, integrated; d2 writer: failed: execution_error"
            in state.packet())
    assert state.unresolved_delegations() == [state.current.delegations[1]]
    assert state.finalize("Done.") == (
        "Done.\n\nTask evidence: writer (d2): failed: execution_error, not integrated.")


def test_show_lists_delegations(tmp_path):
    state = started(tmp_path)
    state.begin_delegation("c1", "reviewer", "independent review")
    state.record_delegation("c1", ok_result(status="completed", artifacts=["/w/review.md"]))
    text = state.show()
    assert "delegations:" in text and "d1 reviewer: completed — independent review" in text
    assert "/w/review.md" in text


# --- 0.16.5 slice 4: unintegrated delegations keep a passed record live ------------------------

def test_finalize_with_unintegrated_delegation_keeps_record_live(tmp_path):
    state = started(tmp_path, "Write it and run the lint.")
    ctx = state.current.context
    ctx.requested_status = "complete"
    ctx.revise(Requirement("lint", "Lint passes", "bash_exec", {"command": "lint"}, "exit_code", 0))
    state.record_result("bash_exec", {"command": "lint"}, "c1", success=True,
                        metadata={"exit_code": 0}, before={})
    state.begin_delegation("c2", "reviewer", "review")
    state.record_delegation("c2", ok_result(status="completed"))
    assert state.finalize("Done.") == (
        "Done.\n\nTask evidence: lint: passed; reviewer (d1): completed, not integrated.")
    assert state.status == "unknown"
    assert state.packet() != ""
    assert json.loads(state.path.read_text())["context"]["status"] == "unknown"
    state.current.delegations[0].integrated = "used it"
    assert state.finalize("Done.") == "Done."
    assert state.status == "complete"
