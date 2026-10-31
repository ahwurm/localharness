"""The task flow's tables and record: the stage table, the limits, the lines and the revision rule."""
from __future__ import annotations

from localharness.orchestrator.task_flow.record import (
    BOUNDARIES,
    STAGES,
    TOOL_STAGES,
    file_doc,
    note_run,
    open_task,
    panel_minutes,
    pointed,
    review_of,
    set_verdict,
)


def test_stages_are_keyed_by_the_six_boundaries_with_eight_fields_each():
    assert list(STAGES) == list(BOUNDARIES) == ["reply", "new", "research", "plan", "step", "task"]
    assert all(len(row) == 8 for row in STAGES.values())
    assert [row.revisions for row in STAGES.values()] == [1, 1, 0, 3, 1, 0]
    assert [row.next for row in STAGES.values()] == [None, "plan", "plan", "step", "task", "done"]


def test_the_tool_whitelist_is_derived_from_the_stage_table():
    assert TOOL_STAGES == ("new", "research", "plan", "run")


def test_new_boundary_opens_files_and_passes_without_using_a_revision():
    record = open_task(3, "fix login", 1)
    assert record.stage == "new"
    assert pointed(record) == "decisions.md"

    key = ("decisions.md", None)
    file_doc(record, "new", key, "d1")
    assert record.filed[key].verdict == "unreviewed"
    assert not record.counts["new"].revised

    set_verdict(record, key, "pass")
    note_run(record, "new", 0)
    assert record.counts["new"].used == 0
    assert record.counts["new"].blockers == 0
    assert review_of(record, "decisions.md") == "decisions-review.md"


def test_new_panel_is_one_critic_at_the_floor_budget():
    assert panel_minutes(STAGES["new"].floor) == 1
