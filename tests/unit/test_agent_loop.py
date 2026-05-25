"""Tests for Session, StuckDetector, BudgetTracker, KillWatcher, and AgentLoop."""
import time
import pytest
from pathlib import Path

from localharness.agent.loop import (
    Session,
    StuckDetector,
    StuckState,
    BudgetTracker,
    BudgetViolation,
    KillWatcher,
    StepResult,
)


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

def test_session_initializes():
    s = Session(agent_id="a", session_id="s", messages=[])
    assert s.agent_id == "a"
    assert s.session_id == "s"
    assert s.messages == []
    assert s.iteration == 0
    assert s.actions_taken == 0
    assert s.summary == ""
    assert s.terminated_reason is None


def test_session_push_appends():
    s = Session(agent_id="a", session_id="s", messages=[])
    msg = {"role": "user", "content": "hello"}
    s.push(msg)
    assert len(s.messages) == 1
    assert s.messages[0] is msg


def test_session_elapsed_seconds_positive():
    s = Session(agent_id="a", session_id="s", messages=[])
    time.sleep(0.01)
    assert s.elapsed_seconds() > 0
    assert s.elapsed_minutes() > 0


def test_session_messages_append_only_via_push():
    """Direct modification of messages list is not via push — push is the correct interface."""
    s = Session(agent_id="a", session_id="s", messages=[])
    s.push({"role": "user", "content": "a"})
    s.push({"role": "user", "content": "b"})
    assert len(s.messages) == 2


# ---------------------------------------------------------------------------
# StuckDetector tests
# ---------------------------------------------------------------------------

def test_stuck_compute_signature_returns_16_chars():
    sd = StuckDetector()
    sig = sd.compute_signature("bash", {"cmd": "ls"})
    assert len(sig) == 16
    assert sig.isalnum()


def test_stuck_compute_signature_order_independent():
    sd = StuckDetector()
    sig1 = sd.compute_signature("tool", {"a": 1, "b": 2})
    sig2 = sd.compute_signature("tool", {"b": 2, "a": 1})
    assert sig1 == sig2


def test_stuck_clear_when_different_calls():
    sd = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "pwd"})
    sd.record("bash", {"cmd": "echo"})
    assert sd.check() == StuckState.CLEAR


def test_stuck_recovering_at_two_identical():
    sd = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "ls"})
    assert sd.check() == StuckState.RECOVERING


def test_stuck_escalate_at_three_identical():
    sd = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "ls"})
    assert sd.check() == StuckState.ESCALATE


def test_stuck_recovery_message_nonempty():
    sd = StuckDetector()
    msg = sd.recovery_message("abcdef1234567890")
    assert isinstance(msg, str) and len(msg) > 0


def test_stuck_most_repeated_signature():
    sd = StuckDetector(window_size=5)
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "pwd"})
    sig_ls = sd.compute_signature("bash", {"cmd": "ls"})
    assert sd.most_repeated_signature() == sig_ls


def test_stuck_clear_when_window_too_small():
    sd = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)
    sd.record("bash", {"cmd": "ls"})
    assert sd.check() == StuckState.CLEAR


# ---------------------------------------------------------------------------
# BudgetTracker tests
# ---------------------------------------------------------------------------

def test_budget_actions_exceeded():
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = 5
    tracker = BudgetTracker(max_actions=5, max_duration_minutes=30.0)
    v = tracker.check(s)
    assert isinstance(v, BudgetViolation)
    assert v.reason == "actions"


def test_budget_actions_not_exceeded():
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = 4
    tracker = BudgetTracker(max_actions=5, max_duration_minutes=30.0)
    assert tracker.check(s) is None


def test_budget_unlimited_actions():
    """max_actions=0 means unlimited — never trips."""
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = 999
    tracker = BudgetTracker(max_actions=0, max_duration_minutes=30.0)
    # Should not trip on actions, only check time (30 min, not elapsed)
    result = tracker.check(s)
    assert result is None or result.reason == "time"


def test_budget_time_exceeded():
    # Use very short duration to trigger time violation
    s = Session(agent_id="a", session_id="s", messages=[])
    time.sleep(0.05)  # short sleep; duration_minutes = 0.0001 (very small)
    tracker = BudgetTracker(max_actions=100, max_duration_minutes=0.0001)
    v = tracker.check(s)
    assert isinstance(v, BudgetViolation)
    assert v.reason == "time"


# ---------------------------------------------------------------------------
# KillWatcher tests
# ---------------------------------------------------------------------------

def test_kill_watcher_false_when_no_file(tmp_path):
    kw = KillWatcher(kill_file_path=tmp_path / "KILL")
    assert kw.is_killed() is False


def test_kill_watcher_true_when_file_exists(tmp_path):
    kill_path = tmp_path / "KILL"
    kill_path.touch()
    kw = KillWatcher(kill_file_path=kill_path)
    assert kw.is_killed() is True


# ---------------------------------------------------------------------------
# Task 2: AgentLoop tests (appended below after Task 1 commit)
# ---------------------------------------------------------------------------
