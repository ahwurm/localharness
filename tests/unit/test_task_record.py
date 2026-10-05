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


def worked(state):
    """One tool dispatch this turn: finalize checks only turns with tool activity (R8)."""
    state.record_result("write", {}, "c0", success=True, metadata={}, before={})
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
    state = worked(started(tmp_path, "Begin."))
    ctx = state.current.context
    ctx.requested_status = "complete"
    state.current.closed = True
    assert state.finalize("All done.") == "All done."
    assert ctx.status == "unknown"  # nothing machine-checkable was declared
    assert state.packet() == ""
    assert "closed (no machine checks declared)" in state.show()
    assert json.loads(state.path.read_text())["closed"] is True


def test_finalize_checkpoint_keeps_packet(tmp_path):
    state = worked(started(tmp_path, "Begin."))
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


# --- 0.16.5 slice 2: references, judgments, revision budget ---

from localharness.agent.context import ContentStore  # noqa: E402
from localharness.agent.task_record import BUDGET_EXHAUSTED, Judgment, Reference  # noqa: E402


def test_round_trip_with_references_judgments_and_verified_revision(tmp_path):
    rec = make_record(tmp_path, references=[Reference("voice", "voice.md", "abc", "f" * 64, "current"),
                                            Reference("pasted", handle="def")],
                      judgments=[Judgment("tone", "Plain tone", "ok", "para 2", "cut adverbs", "assessed")],
                      verified_revision={"draft": "a" * 64})
    assert TaskRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec
    old = make_record(tmp_path).to_dict()
    for k in ("references", "judgments", "verified_revision"):
        old.pop(k)
    back = TaskRecord.from_dict(old)
    assert (back.references, back.judgments, back.verified_revision) == ([], [], {})


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
    assert "reference src missing" in loaded.notes


def test_revision_budget_counts_edits_to_verified_artifacts(tmp_path):
    draft = tmp_path / "draft.md"
    draft.write_text("v1")
    state = started(tmp_path, "Begin.")
    rec, ctx = state.current, state.current.context
    ctx.artifacts["draft"] = draft
    ctx.revise(Requirement("lint", "Lint", "bash_exec", {"command": "lint"}, "exit_code", 0, ("draft",)))

    def lint(code, call):
        state.record_result("bash_exec", {"command": "lint"}, call, success=True,
                            metadata={"exit_code": code}, before=state.revisions())

    def edit(text, call):
        before = state.revisions()
        draft.write_text(text)
        state.record_result("write", {"path": "draft.md"}, call, success=True, metadata={}, before=before)

    lint(1, "c1")
    assert ctx.outcomes()["lint"] == "failed" and "draft" in rec.verified_revision
    state.record_result("read", {"path": "x"}, "c2", success=True, metadata={}, before=state.revisions())
    assert rec.revisions_used == 0
    edit("v2", "c3")
    assert rec.revisions_used == 1 and "draft" not in rec.verified_revision
    edit("v3", "c4")  # unverified since the last edit: not another revision
    assert rec.revisions_used == 1 and "Revision budget: used 1 of 1" in state.packet()
    assert BUDGET_EXHAUSTED not in state.packet()
    lint(0, "c5")
    edit("v4", "c6")
    assert rec.revisions_used == 2 and BUDGET_EXHAUSTED in state.packet()


def test_finalize_reports_open_judgments_only_on_unchanged_candidates(tmp_path):
    state = worked(started(tmp_path, "Begin."))
    state.current.context.requested_status = "complete"
    state.current.judgments = [Judgment("tone", "Plain tone"), Judgment("flow", "Flows", status="assessed")]
    assert state.finalize("Draft done.") == "Draft done.\n\nOpen editorial judgments (not assessed): tone."
    state.current.context.revise(Requirement("lint", "Lint", "bash_exec", {"command": "lint"}))
    out = state.finalize("Draft done.")
    assert out.startswith("Task remains unverified") and "editorial" not in out
    state.current.judgments[0].status = "waived"
    assert "Open editorial" not in state.finalize("Draft done.")


def test_completed_record_does_not_rewrite_later_replies(tmp_path):
    state = worked(started(tmp_path, "Skip the lint gate for this draft."))
    ctx = state.current.context
    ctx.requested_status = "complete"
    ctx.revise(Requirement("lint", "Lint", "bash_exec", {"command": "lint"}))
    ctx.waive("lint", human_decision="skip the lint gate")
    assert state.finalize("Done.").startswith("Task finished with a human waiver")
    assert state.status == "complete"
    assert state.finalize("4") == "4"  # a later aside is not rewritten


def test_show_lists_references_and_judgments(tmp_path):
    state = refs_state(tmp_path, Reference("voice", "voice.md"), Reference("pasted", handle="abc"))
    state.current.judgments = [Judgment("tone", "Plain tone", assessment="mostly plain", fix="cut adverbs")]
    text = state.show()
    assert "  voice: voice.md (pending)" in text and "  pasted: handle abc (pending)" in text
    assert "  tone: open — Plain tone | assessment: mostly plain | fix: cut adverbs" in text


# --- 0.16.5 slice 3: the record owns delegations; tool-less asides are never rewritten ---------

from localharness.agent.task_record import Delegation  # noqa: E402
from localharness.tools.base import ToolResult  # noqa: E402

REFUSAL = ("Delegation budget (2) used for this task; integrate the results you have or ask the "
           "user to raise it")


def ok_result(output="SUBAGENT RUN COMPLETE ... tail words", **meta):
    return ToolResult(output=output, metadata={"delegated_to": "reviewer", **meta})


def test_delegations_round_trip_and_legacy_records_load_empty(tmp_path):
    rec = make_record(tmp_path, delegations=[Delegation("d1", "reviewer", "review", "completed",
                                                        ["/w/r.md"], "f", "u", "r", "used", "c1")])
    assert TaskRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec
    legacy = rec.to_dict()
    del legacy["delegations"]
    assert legacy["format"] == 1 and TaskRecord.from_dict(legacy).delegations == []


def test_begin_delegation_inert_without_record_then_budgeted(tmp_path):
    bare = TaskState(tmp_path / "task.json", workspace=str(tmp_path))
    assert bare.begin_delegation("c1", "reviewer", "review") is None
    assert not (tmp_path / "task.json").exists()
    state = started(tmp_path)
    assert state.begin_delegation("c1", "reviewer", "independent review") is None
    assert state.begin_delegation("c2", "writer", "draft") is None
    assert state.begin_delegation("c3", "writer", "again") == REFUSAL
    saved = json.loads(state.path.read_text())["delegations"]
    assert [(d["id"], d["agent"], d["status"], d["call_id"]) for d in saved] == [
        ("d1", "reviewer", "running", "c1"), ("d2", "writer", "running", "c2")]


@pytest.mark.parametrize("result,status", [
    (ok_result(status="completed"), "completed"),
    (ok_result(), "completed"),
    (ok_result(status="no_result"), "no_result"),
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
    assert ("Delegations: d1 reviewer: completed, integrated; d2 writer: failed: execution_error"
            in state.packet())
    assert state.unresolved_delegations() == [state.current.delegations[1]]
    state.record_result("read", {}, "c9", success=True, metadata={}, before={})
    assert state.finalize("Done.") == (
        "Done.\n\nDelegated work unresolved: d2 writer: failed: execution_error.")


def test_show_lists_delegations(tmp_path):
    state = started(tmp_path)
    state.begin_delegation("c1", "reviewer", "independent review")
    state.record_delegation("c1", ok_result(status="completed", artifacts=["/w/review.md"]))
    text = state.show()
    assert "delegations:" in text and "d1 reviewer: completed — independent review" in text
    assert "/w/review.md" in text


def test_tool_less_turn_is_never_rewritten(tmp_path):
    """R8: an aside with no tool activity this turn keeps its reply, even with a failed check."""
    state = started(tmp_path)
    ctx = state.current.context
    ctx.requested_status = "complete"
    ctx.revise(Requirement("lint", "Lint passes", "bash_exec", {"command": "lint"}, "exit_code", 0))
    state.observe_human("go")
    state.record_result("bash_exec", {"command": "lint"}, "c1", success=True,
                        metadata={"exit_code": 1}, before={})
    state.observe_human("by the way, what is a tide pool?")
    assert state.finalize("A rocky pool left by the tide.") == "A rocky pool left by the tide."
    state.observe_human("nudge", new_turn=False)  # a mid-turn steering nudge keeps the count
    state.record_result("read", {}, "c2", success=True, metadata={}, before={})
    state.observe_human("nudge", new_turn=False)
    assert state.finalize("All done.").startswith("Task remains unverified.")
