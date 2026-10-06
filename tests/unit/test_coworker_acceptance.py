"""Acceptance map: the ten composed coworker cases and the deterministic tests that cover them.

Each entry names a test that exists (`test_every_mapped_test_exists` resolves them all).

1. Ordinary arithmetic and answer-only discussion (no additional preflight call, mutation, or
   delegation):
   tests/unit/test_task_tool.py::test_loop_without_record_is_an_ordinary_request
   tests/unit/test_task_context.py::test_ordinary_query_one_call_and_no_task_packet
   tests/unit/test_task_context.py::test_delegation_without_record_is_unchanged
   tests/unit/test_task_context.py::test_aside_keeps_its_words_and_gets_the_evidence_line
   tests/unit/test_coworker_acceptance.py::test_case1_ordinary_question_adds_no_call_with_a_task_state
2. Goal plus aside plus correction:
   tests/unit/test_task_workflow.py::test_correction_aside_checkpoint_and_restart_from_disk
   tests/unit/test_task_context.py::test_task_packet_survives_lossy_packing_and_human_steering
   tests/unit/test_task_context.py::test_checkpoint_retains_unfinished_asks_then_correction
3. Existing accepted decision plus one material ambiguity:
   tests/unit/test_coworker_acceptance.py::test_case3_open_question_resolved_and_earlier_human_decision_carried
   tests/unit/test_task_tool.py::test_decide_check_and_waive_cite_human_turns
4. Outline checkpoint and restart:
   tests/unit/test_task_workflow.py::test_correction_aside_checkpoint_and_restart_from_disk
   tests/unit/test_task_tool.py::test_checkpoint_requires_a_cited_human_turn
   tests/unit/test_task_record.py::test_finalize_checkpoint_keeps_packet
5. Source/voice dependencies under forced compaction, with overflow:
   tests/unit/test_task_references.py::test_a_references_survive_forced_eviction
   tests/unit/test_task_references.py::test_b_overflow_blocks_once_with_the_narrowing_notice
   tests/unit/test_task_references.py::test_unchanged_reference_is_redeclared_after_the_next_human_turn
   tests/unit/test_task_context.py::test_oversized_task_packet_blocks_without_model_call
6. Failed executable check, focused fix, rerun, with revision facts:
   tests/unit/test_task_references.py::test_e_checks_report_draft_revisions
   tests/unit/test_task_references.py::test_f_close_complete_stays_live_until_evidence_settles
   tests/unit/test_task_context.py::test_actual_structured_gate_result_not_prose
7. Successful, failed, missing, malformed, and interrupted delegated review:
   tests/unit/test_task_delegation.py::test_completed_delegation_integrated_then_closed
   tests/unit/test_task_delegation.py::test_failed_delegation_is_named_in_reply_and_keeps_record_live
   tests/unit/test_task_delegation.py::test_malformed_handoff_is_none_and_status_stays_runtime
   tests/unit/test_task_delegation.py::test_cancelled_delegation_becomes_interrupted
   tests/unit/test_task_delegation.py::test_child_out_of_budget_is_budget_exhausted
   tests/unit/test_task_record.py::test_record_delegation_status_table
   tests/unit/test_task_record.py::test_finalize_with_unintegrated_delegation_keeps_record_live
   tests/unit/test_task_context.py::test_delegation_opens_entry_before_dispatch_without_refusal
8. Changed source/draft after restart:
   tests/unit/test_task_references.py::test_c_changed_source_reaches_the_next_request
   tests/unit/test_task_references.py::test_d_restart_resnapshots_path_references
   tests/unit/test_task_record.py::test_reconcile_notes_changed_and_missing_artifacts
   tests/unit/test_task_record.py::test_reconcile_nulls_handles_and_notes_reference_changes
   tests/unit/test_task_context.py::test_only_changed_dependency_invalidates_receipt
9. Missing primary research: uncited claims stay assumptions; a waiver must cite a turn:
   tests/unit/test_coworker_acceptance.py::test_case9_uncited_research_is_an_assumption_and_a_waiver_needs_a_turn
   tests/unit/test_task_tool.py::test_decide_check_and_waive_cite_human_turns
10. Revision facts and workspace isolation:
   tests/unit/test_task_references.py::test_e_checks_report_draft_revisions
   tests/unit/test_task_record.py::test_artifact_revisions_and_run_facts_in_packet
   tests/unit/test_task_record.py::test_load_missing_corrupt_and_foreign
   tests/unit/test_task_workflow.py::test_correction_aside_checkpoint_and_restart_from_disk (the
   reload from another folder)

These tests prove the runtime's half: the runtime records and reports facts; it refuses only
malformed or over-capacity input, and never refuses or rewrites on the model's behalf. Whether a
model asks one useful question, avoids redrafting, or writes good prose is not deterministic. That
belongs to the preregistered live comparison, which this file does not replace.
"""
import re
from pathlib import Path

from localharness.agent.task_record import TaskState
from localharness.core.events import Observation
from localharness.tools.builtin.task_tool import TaskTool
from localharness.tools.registry import ToolRegistry
from tests.unit.test_task_context import make_loop
from tests.unit.test_task_workflow import Registry, capture, packet_of, scripted

ROOT = Path(__file__).resolve().parents[2]


def capture_with_tools(llm):
    seen, tools = [], []
    original = llm.stream_complete

    async def wrapped(messages, *a, **kw):
        seen.append(list(messages))
        tools.append(kw.get("tools", a[0] if a else None))
        return await original(messages, *a, **kw)
    llm.stream_complete = wrapped
    return seen, tools


def tool_names(schemas):
    return {(s.get("function") or s).get("name") if isinstance(s, dict) else getattr(s, "name", None)
            for s in schemas or []}


async def test_case1_ordinary_question_adds_no_call_with_a_task_state(tmp_path, bus, mock_llm_client):
    """With `task` advertised (so the act-guard is live), a TaskState costs no extra call."""
    state = TaskState(tmp_path / "a" / "task.json", workspace=str(tmp_path))
    runs = []
    for task in (None, state):
        llm = mock_llm_client([mock_llm_client.Response(content="42"),
                               mock_llm_client.Response(content="CONFIRMED")])
        seen, tools = capture_with_tools(llm)
        reg = ToolRegistry()
        await reg.register(TaskTool(task or TaskState(None, workspace=str(tmp_path))), scope="global")
        loop = make_loop(llm, bus, tmp_path, task, reg)
        runs.append((await loop.run_turn("Answer only: what is 17 + 25?"), seen, tools))
    (base, base_seen, base_tools), (with_state, seen, tools) = runs
    assert base == with_state == "42"
    assert len(base_seen) == len(seen) == 2  # the answer plus the pre-existing act-guard round-trip
    assert all("task" in tool_names(t) for t in base_tools + tools)
    assert not any(packet_of(request) for request in base_seen + seen)
    assert not state.path.exists() and state.current is None
    assert not [e for e in bus.history(event_types=[Observation]) if e.tool_name in {"task", "agent"}]


QUESTION = "Should the profile name the customer's region?"


async def test_case3_open_question_resolved_and_earlier_human_decision_carried(
        tmp_path, bus, mock_llm_client):
    state = TaskState(tmp_path / "agents" / "t" / "task.json", workspace=str(tmp_path))
    llm = scripted(
        mock_llm_client,
        ("task", {"action": "start", "objective": "Acme customer profile",
                  "assignment": "Profile from sources/"}),
        ("task", {"action": "update", "question": QUESTION}),
        "One question: should the profile name the region?",
        ("task", {"action": "update", "resolve_question": QUESTION}),
        ("task", {"action": "decide", "text": "Region stays anonymous", "human_turn": 2}),
        ("task", {"action": "decide", "text": "Interview notes are the primary source", "human_turn": 1}),
        "Noted.",
        "The outline.",
    )
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path))

    assert await loop.run_turn("Prepare the Acme customer profile from the sources folder. Use the "
                               "interview notes as the primary source.") == (
        "One question: should the profile name the region?")
    mark = len(seen)
    assert await loop.run_turn("Keep the region anonymous.") == "Noted."
    assert f"Open questions: {QUESTION}" in packet_of(seen[mark])
    mark = len(seen)
    assert await loop.run_turn("What is next?") == "The outline."
    packet = packet_of(seen[mark])
    assert "Open questions" not in packet
    assert '[human turn 2: "Keep the region anonymous."] Region stays anonymous' in packet
    assert ('[human turn 1: "Prepare the Acme customer profile from the sources folder. Use the '
            'interview notes as the primary source."] Interview notes are the primary source') in packet
    assert state.current.questions == []


async def test_case9_uncited_research_is_an_assumption_and_a_waiver_needs_a_turn(
        tmp_path, bus, mock_llm_client):
    state = TaskState(tmp_path / "agents" / "t" / "task.json", workspace=str(tmp_path))
    llm = scripted(
        mock_llm_client,
        ("task", {"action": "start", "objective": "Acme profile from sources/ only",
                  "assignment": "Draft the profile"}),
        ("task", {"action": "decide", "text": "Interviews confirm 40% growth"}),
        ("task", {"action": "artifact", "key": "draft", "path": "draft.md"}),
        ("task", {"action": "check", "key": "lint", "description": "Draft passes the lint",
                  "tool": "bash_exec", "arguments": {"command": "python3 checks/lint.py draft.md"},
                  "result_field": "exit_code", "expected": 0, "depends_on": ["draft"]}),
        ("task", {"action": "waive", "key": "lint"}),
        ("task", {"action": "waive", "key": "lint", "human_turn": 9}),
        "There are no interview notes; stopping.",
    )
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path))

    reply = await loop.run_turn("Write the Acme profile from sources/ only. If there are no "
                                "interview notes, say so and stop.")
    (decision,) = state.current.decisions
    assert decision.origin == "model"
    assert any("[assumption] Interviews confirm 40% growth" in packet_of(r) for r in seen[2:])
    tool_text = [str(m.get("content")) for r in seen for m in r if m.get("role") == "tool"]
    assert any("A waiver cites the human turn that waives it" in t for t in tool_text)
    assert any("human turn 9 is not on record; stored turns: 1-1" in t for t in tool_text)
    ctx = state.current.context
    assert "lint" not in ctx.waivers and ctx.outcomes()["lint"] == "unknown"
    assert reply == "There are no interview notes; stopping.\n\nTask evidence: lint: unknown."


def test_every_mapped_test_exists():
    entries = re.findall(r"(tests/unit/\w+\.py)::(\w+)", __doc__)
    assert len(entries) >= 30
    for rel, name in entries:
        source = (ROOT / rel).read_text()
        assert re.search(rf"^(async )?def {name}\(", source, re.M), f"{rel}::{name}"
