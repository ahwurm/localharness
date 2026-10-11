"""The action limit follows the live surface: an attended terminal root runs unbounded, a remote
unattended surface is capped by the machine knob (and a child by its fallback), and the failure
text names the key that bound."""
from __future__ import annotations

import pytest

from localharness.agent.loop import BudgetLabels, BudgetTracker, Session, _budget_note, _cap_limit


def _session(actions: int = 0) -> Session:
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = actions
    return s


# --------------------------------------------------------------------------- _cap_limit


@pytest.mark.parametrize(
    ("limit", "cap", "expected"),
    [(0, 8, 8), (16, 8, 8), (5, 8, 5), (8, 8, 8)],
)
def test_cap_limit_treats_zero_as_no_limit(limit, cap, expected):
    assert _cap_limit(limit, cap) == expected


def test_budget_labels_default_to_nothing():
    labels = BudgetLabels()
    assert (labels.actions, labels.minutes, labels.fallback) == (None, None, None)


# --------------------------------------------------------------------------- the tracker


def test_an_int_tracker_behaves_as_before():
    t = BudgetTracker(max_actions=5, max_duration_minutes=30.0)
    assert t.limit == 5 and t.bound_by is None
    assert t.check(_session(4)) is None
    v = t.check(_session(5))
    assert v is not None and v.reason == "actions" and v.limit == 5 and v.current == 5
    assert v.message == "Agent reached the maximum of 5 tool calls. Stopping to prevent runaway execution."


def test_a_labelled_supplier_names_the_key_that_bound():
    t = BudgetTracker(max_actions=lambda: (100, "channels.unattended_max_actions"), max_duration_minutes=0.0)
    assert t.limit == 100 and t.bound_by == "channels.unattended_max_actions"
    v = t.check(_session(100))
    assert v is not None
    assert v.message == (
        "Agent reached the maximum of 100 tool calls. Stopping to prevent runaway execution. "
        "(channels.unattended_max_actions)"
    )


def test_a_supplier_of_zero_never_trips():
    t = BudgetTracker(max_actions=lambda: (0, None), max_duration_minutes=0.0)
    assert t.check(_session(10_000)) is None


def test_the_supplier_is_re_read_on_every_check_and_once_per_check():
    reads: list[int] = []
    limits = iter([(0, None), (3, "k"), (3, "k")])

    def supplier():
        reads.append(1)
        return next(limits)

    t = BudgetTracker(max_actions=supplier, max_duration_minutes=0.0)
    assert t.check(_session(5)) is None
    assert len(reads) == 1
    v = t.check(_session(5))
    assert v is not None and v.limit == 3 and v.message.endswith("(k)")
    assert len(reads) == 2


def _old_session(minutes: float) -> Session:
    s = _session(2)
    s.start_time -= minutes * 60.0
    return s


def test_the_time_message_carries_its_label_when_one_is_set():
    labelled = BudgetTracker(max_actions=0, max_duration_minutes=1.0, minutes_bound_by="org.children_budget.default")
    v = labelled.check(_old_session(2.0))
    assert v is not None and v.reason == "time"
    assert v.message == (
        "Agent reached the time limit of 1.0 minutes. Stopping after 2 tool calls. (org.children_budget.default)"
    )
    plain = BudgetTracker(max_actions=0, max_duration_minutes=1.0).check(_old_session(2.0))
    assert plain is not None
    assert plain.message == "Agent reached the time limit of 1.0 minutes. Stopping after 2 tool calls."


def test_budget_note_reads_a_live_limit_and_is_silent_without_one():
    live = BudgetTracker(max_actions=lambda: (12, "k"), max_duration_minutes=0)
    assert "3/12 tool calls used" in _budget_note(_session(3), live)
    assert _budget_note(_session(3), BudgetTracker(max_actions=lambda: (0, None), max_duration_minutes=0)) == ""
