"""The task flow's tables and record: the stage table, the limits, the lines and the revision rule."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from localharness.config.models import ChildBudget
from localharness.core.caps import Caps
from localharness.orchestrator.task_flow.record import (
    BOUNDARIES,
    CRITICS_AT_FLOOR,
    LIMITS,
    LINES,
    REPLY_RESERVE_MINUTES,
    SENSED,
    STAGES,
    TOOL_STAGES,
    Need,
    Unit,
    file_doc,
    limited,
    need,
    note_run,
    open_task,
    panel_critics,
    panel_minutes,
    pointed,
    printed_bound,
    review_of,
    set_verdict,
    unit_minutes,
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


def within(low, value, high) -> bool:
    if isinstance(value, ChildBudget):
        return all(getattr(low, f) <= getattr(value, f) <= getattr(high, f) for f in ("actions", "minutes"))
    return low <= value <= high


def test_every_limit_row_orders_floor_default_ceiling():
    assert all(within(row.floor, row.default, row.ceiling) for row in LIMITS.values())
    assert {"critic", "executor", "explorer", "additions", "flow_cap_minutes"} <= set(LIMITS)


def test_revisions_limit_rows_derive_from_the_stage_table_with_the_reply_ceiling_one():
    rows = {k.removeprefix("revisions."): v for k, v in LIMITS.items() if k.startswith("revisions.")}
    assert {b: r.ceiling for b, r in rows.items()} == {"reply": 1, "new": 3, "plan": 5, "step": 3}
    assert all(r.floor == 0 and r.default == STAGES[b].revisions for b, r in rows.items())


def test_limit_budgets_are_the_shipped_numbers():
    limits, _ = limited()
    assert (LIMITS["critic"].floor.actions, LIMITS["critic"].floor.minutes) == (3, 1)
    assert (LIMITS["critic"].ceiling.actions, LIMITS["critic"].ceiling.minutes) == (12, 6)
    assert CRITICS_AT_FLOOR == {"jtbd", "answer"}
    assert (limits.critic("jtbd").actions, limits.critic("answer").minutes) == (3, 1)
    assert (limits.critic("factual").actions, limits.critic("factual").minutes) == (12, 6)
    assert (limits.executor.actions, limits.executor.minutes) == (24, 25)
    assert (limits.explorer.actions, limits.explorer.minutes) == (12, 6)
    assert (LIMITS["executor"].ceiling.actions, LIMITS["executor"].ceiling.minutes) == (48, 40)
    assert (LIMITS["explorer"].ceiling.actions, LIMITS["explorer"].ceiling.minutes) == (24, 10)
    assert (LIMITS["additions"].floor, LIMITS["additions"].ceiling) == (0, 3)


def test_bound_of_every_unit_at_the_defaults_is_computed_from_the_table():
    limits, _ = limited()
    minutes = lambda *unit, sensed=(): unit_minutes(Unit(*unit), limits, sensed)  # noqa: E731
    assert minutes("new", "panel") == minutes("new", "file") == 1
    assert minutes("plan", "panel") == minutes("plan", "file") == 13
    assert minutes("research", "explore") == 6
    assert minutes("research", "panel") == 7
    assert minutes("step", "execute", 1) == minutes("step", "revise", 1) == 25
    assert minutes("step", "panel", 1) == 6
    assert minutes("task", "panel") == 7
    assert minutes("reply", "panel", sensed=("factual", "recency")) == 28


def test_bound_of_a_three_step_task_sums_to_114_minutes():
    limits, _ = limited()
    step = unit_minutes(Unit("step", "execute", 1), limits) + unit_minutes(Unit("step", "panel", 1), limits)
    total = (unit_minutes(Unit("new", "panel"), limits) + unit_minutes(Unit("plan", "panel"), limits)
             + 3 * step + unit_minutes(Unit("task", "panel"), limits))
    assert total == 114


def test_printed_bound_rounds_down_and_never_promises_more_than_the_remainder():
    assert printed_bound(25, 600) == 10
    assert printed_bound(13, 13 * 60) == 13
    assert printed_bound(25, 9.9 * 60) == 9
    assert printed_bound(25) == 25
    assert printed_bound(1.5) == 1
    assert isinstance(printed_bound(25, 600), int)


def test_reserve_and_flow_cap_hold_one_turns_children_to_276_and_456_minutes():
    assert REPLY_RESERVE_MINUTES == 96
    assert LIMITS["flow_cap_minutes"].default + REPLY_RESERVE_MINUTES == 276
    assert LIMITS["flow_cap_minutes"].ceiling + REPLY_RESERVE_MINUTES == 456
    assert SENSED == ("factual", "recency", "spec-compliance")


def test_limited_without_a_config_or_with_every_field_unset_gives_the_defaults():
    unset = SimpleNamespace(panels=None, revisions=None, critics=None, executor=None, explorer=None,
                            flow_cap_minutes=None)
    for cfg in (None, unset):
        limits, notes = limited(cfg)
        assert notes == ()
        assert limits.revisions == {"reply": 1, "new": 1, "plan": 3, "step": 1}
        assert limits.flow_cap_minutes == 180
        assert limits.additions == {} and limits.critics == {}


def test_limited_clamps_an_executor_budget_to_its_ceiling_with_one_note():
    limits, notes = limited(SimpleNamespace(executor=ChildBudget(actions=100, minutes=100)))
    assert (limits.executor.actions, limits.executor.minutes) == (48, 40)
    assert len(notes) == 1 and notes[0].startswith("task_flow.executor: ")


def test_limited_raises_a_critic_budget_below_the_floor_and_cannot_lower_it():
    limits, notes = limited(SimpleNamespace(critics={"answer": ChildBudget(actions=1, minutes=0.5)}))
    assert (limits.critic("answer").actions, limits.critic("answer").minutes) == (3, 1)
    assert len(notes) == 1


def test_limited_revisions_clamp_to_the_ceiling_and_drop_a_boundary_with_none():
    limits, notes = limited(SimpleNamespace(revisions={"plan": 9, "reply": 3, "research": 1}))
    assert limits.revisions["plan"] == 5 and limits.revisions["reply"] == 1
    assert len(notes) == 3 and any("research" in n and "dropped" in n for n in notes)


def test_limited_revisions_one_past_the_ceiling_and_one_below_the_floor_clamp():
    assert limited(SimpleNamespace(revisions={"plan": 6}))[0].revisions["plan"] == 5
    assert limited(SimpleNamespace(revisions={"plan": 5}))[1] == ()
    assert limited(SimpleNamespace(revisions={"plan": -1}))[0].revisions["plan"] == 0


def test_limited_keeps_three_panel_additions_and_names_the_fourth():
    limits, notes = limited(SimpleNamespace(panels={"plan": ["security", "quality", "factual", "recency"]}))
    assert limits.additions["plan"] == ("security", "quality", "factual")
    assert len(notes) == 1 and "recency" in notes[0]


def test_limited_drops_an_addition_in_the_floor_or_with_a_bad_name_and_notes_each():
    limits, notes = limited(SimpleNamespace(panels={"plan": ["jtbd", "Bad Name", "security"], "nowhere": ["x"]}))
    assert limits.additions == {"plan": ("security",)}
    assert len(notes) == 3


def test_limited_flow_cap_clamps_into_one_to_three_hundred_sixty():
    assert limited(SimpleNamespace(flow_cap_minutes=1000))[0].flow_cap_minutes == 360
    assert limited(SimpleNamespace(flow_cap_minutes=0))[0].flow_cap_minutes == 1
    assert limited(SimpleNamespace(flow_cap_minutes=90))[1] == ()


def test_limited_known_names_drop_an_addition_and_a_critic_budget_outside_them():
    cfg = SimpleNamespace(panels={"plan": ["security", "ghost"]}, critics={"ghost": ChildBudget(actions=5, minutes=2)})
    limits, notes = limited(cfg, known=frozenset({"security"}))
    assert limits.additions["plan"] == ("security",) and limits.critics == {}
    assert len(notes) == 2


def test_panel_critics_are_the_floor_then_the_sensed_then_the_additions_without_duplicates():
    limits, _ = limited(SimpleNamespace(panels={"reply": ["security", "factual"]}))
    assert panel_critics("reply", limits, sensed=("factual",)) == ("answer", "jtbd", "factual", "security")
    assert panel_critics("reply", limited()[0]) == ("answer", "jtbd")


def test_panel_critics_with_an_empty_configured_panel_still_yield_the_floor():
    limits, _ = limited(SimpleNamespace(panels={"reply": [], "plan": []}))
    assert panel_critics("reply", limits) == ("answer", "jtbd")
    assert panel_critics("plan", limits) == STAGES["plan"].floor


ROWS = [
    "armed", "running", "stopping", "stopping idle", "heard", "queued", "queued slash", "queued phone",
    "decisions filed", "pass", "reopened", "blocked", "unreviewed", "interview open", "research blocked",
    "research unreviewed", "plan unreviewed", "review blocked", "review unreviewed", "step pass",
    "step pass ended", "step pass last", "step blocked", "step unreviewed", "step accepted", "step stopped",
    "halted", "halted cap", "not started", "done", "closed", "paused", "run needs a plan", "decisions unfiled",
    "decisions blocked", "person decides", "nothing blocked", "nothing to revise", "research full",
    "nothing to review", "no task", "open and not blocked", "step not executed", "step not in the plan",
    "research closed", "done refused", "not adopted", "not the open task", "not a task doc",
    "edited since filed",
]


def test_lines_hold_every_row_of_the_table_with_both_done_rows_under_distinct_keys():
    assert set(LINES) == set(ROWS)
    assert LINES["done"] != LINES["done refused"]
    assert all(len(line) == 4 for line in LINES.values())


def test_lines_model_renderings_appear_verbatim():
    models = {line.model for line in LINES.values()}
    for text in (
        "run needs a passed or accepted plan; call stage=plan",
        "the interview is open; ask, or file stage=new first",
        "decisions.md is blocked; the person reopens or accepts it",
        "no unreviewed answer; next: plan",
        "no task is open; the person opens one with /task new",
        "plan.md is filed; research is closed",
        "the task is done; the person starts the next",
        "the doc changed after it was filed; the person accepts it",
        "the task review blocked the plan; call stage=plan for the findings, then revise it",
        "panel unfinished; the person re-runs it",
        "halted by the person",
        "the flow cap is spent for this turn; the person continues",
        f"{Caps.QUESTIONS} questions are filed; call stage=research to review them",
    ):
        assert text in models


def test_lines_model_only_rows_carry_no_commands():
    for row in ("decisions unfiled", "person decides", "research full", "nothing to review"):
        assert LINES[row].model and LINES[row].commands == ()


def test_lines_carry_no_literal_bound():
    assert not [t for line in LINES.values() for t in (line.tail, line.state) if "up to " in t and any(c.isdigit() for c in t)]


def test_need_builds_from_a_row_with_commands_reason_and_model_text():
    built = need("run needs a plan")
    assert isinstance(built, Need)
    assert built.commands == ("/task plan",)
    assert built.reason == "run needs a plan"
    assert built.model == "run needs a passed or accepted plan; call stage=plan"


def test_need_substitutes_placeholders_in_the_reason_commands_and_model():
    built = need("person decides", doc="plan.md", why="revisions used up")
    assert built.model == "plan.md is blocked, revisions used up; the person decides"
    edited = need("edited since filed", doc="plan.md")
    assert edited.reason == "plan.md edited since filed"
    assert edited.commands == ("/task accept plan.md (uses your edit)",)
    assert need("no task").commands == ("/task new <objective>",)
    assert need("open and not blocked", task="t3").commands[1] == "new <objective> (closes t3)"


def test_need_unknown_row_raises():
    with pytest.raises(KeyError):
        need("no such row")
