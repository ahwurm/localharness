"""Degenerate-repetition guard at the loop's tool-less acceptance seam (issue #152).

Live receipt (2026-09-02): a web-research delegation returned ONE narration line repeated
265 times ("\n\n"-separated, ~15 KB, exactly one unique line) with zero tool calls, and the
acceptance gate published it as a completed task — twice in a row, so the session burned its
time and the user got no answer. `StuckDetector` structurally cannot see this: it is only ever
fed `(tool_name, args)`, so a zero-tool-call turn never reaches it.

The guard is a pure line-based detector at the same seam the act-guard / baton gate live at:
one bounded corrective nudge, then an honest task failure — never a TaskComplete carrying the
repetition.
"""
from __future__ import annotations

import logging

import pytest

from localharness.agent.loop import detect_degenerate_repetition


# --- the shipped defaults, passed explicitly so detector tests don't drift with config ------
_T = {"min_lines": 10, "max_unique_ratio": 0.15}

# The live incident's exact shape.
_LIVE_LINE = "Gathering the latest news on the topic…"
_LIVE_REPEATS = 265
_LIVE_DEGENERATE = "\n\n".join([_LIVE_LINE] * _LIVE_REPEATS)


# --- detector: positives (degenerate — must NOT be accepted as an answer) -------------------

def test_live_incident_shape_is_degenerate():
    """The receipt: 265 identical lines, blank-line separated. Stats are returned for logging."""
    stats = detect_degenerate_repetition(_LIVE_DEGENERATE, **_T)
    assert stats.degenerate is True
    assert stats.total_lines == _LIVE_REPEATS      # blank separators are not lines
    assert stats.unique_lines == 1
    assert stats.ratio == pytest.approx(1 / _LIVE_REPEATS)


def test_single_newline_separated_repetition_is_degenerate():
    stats = detect_degenerate_repetition("\n".join(["Working on it."] * 40), **_T)
    assert stats.degenerate is True
    assert (stats.total_lines, stats.unique_lines) == (40, 1)


def test_trailing_whitespace_does_not_fake_uniqueness():
    """Lines are stripped before uniqueness — trailing spaces must not defeat the guard."""
    text = "\n".join([f"Searching the web.{' ' * (i % 3)}" for i in range(30)])
    assert detect_degenerate_repetition(text, **_T).degenerate is True


def test_mostly_repetition_with_a_few_distinct_lines_is_degenerate():
    """20 lines, 3 unique -> ratio exactly 0.15 (the boundary is inclusive)."""
    text = "\n".join(["Same line."] * 18 + ["A second line.", "A third line."])
    stats = detect_degenerate_repetition(text, **_T)
    assert (stats.total_lines, stats.unique_lines) == (20, 3)
    assert stats.ratio == pytest.approx(0.15)
    assert stats.degenerate is True


def test_boundary_exactly_min_lines_is_degenerate():
    stats = detect_degenerate_repetition("\n".join(["One line."] * 10), **_T)
    assert stats.total_lines == 10
    assert stats.degenerate is True


# --- detector: negatives (a false positive costs a real answer — precision matters) ---------

def test_one_line_short_of_min_lines_is_not_flagged():
    stats = detect_degenerate_repetition("\n".join(["One line."] * 9), **_T)
    assert stats.total_lines == 9
    assert stats.degenerate is False


def test_short_replies_are_never_flagged():
    for text in ("The answer is 42.", "Yes.\nYes.\nYes.", "a\na\na\na\na", "", "   ", None):
        assert detect_degenerate_repetition(text, **_T).degenerate is False, text


def test_healthy_long_prose_is_not_flagged():
    text = "\n".join(
        f"Paragraph {i}: a distinct sentence about a distinct part of the answer." for i in range(30)
    )
    stats = detect_degenerate_repetition(text, **_T)
    assert stats.unique_lines == 30
    assert stats.degenerate is False


def test_thirty_item_distinct_list_is_not_flagged():
    text = "Here are the files I found:\n\n" + "\n".join(f"- item_{i}.md" for i in range(30))
    assert detect_degenerate_repetition(text, **_T).degenerate is False


def test_just_above_the_ratio_is_not_flagged():
    """20 lines, 4 unique -> ratio 0.20 > 0.15: a repetitive but not degenerate reply stands."""
    text = "\n".join(["Same line."] * 17 + ["Second.", "Third.", "Fourth."])
    stats = detect_degenerate_repetition(text, **_T)
    assert stats.ratio == pytest.approx(0.20)
    assert stats.degenerate is False


def test_wall_of_text_without_newlines_is_not_flagged():
    """Documented limitation: the detector is LINE-based, so newline-free repetition is a miss
    (conservative by design — see the detector docstring)."""
    assert detect_degenerate_repetition("I am gathering the news. " * 200, **_T).degenerate is False


# --- config ---------------------------------------------------------------------------------
from localharness.config.models import AgentConfig, RepetitionGuardConfig  # noqa: E402


def test_repetition_guard_config_defaults():
    a = AgentConfig(name="x", role="y")
    assert a.repetition_guard.enabled is True
    assert a.repetition_guard.min_lines == 10
    assert a.repetition_guard.max_unique_ratio == 0.15


def test_repetition_guard_config_bounds():
    import pydantic

    RepetitionGuardConfig(min_lines=2, max_unique_ratio=0.01)
    RepetitionGuardConfig(min_lines=1000, max_unique_ratio=1.0)
    for bad in ({"min_lines": 1}, {"min_lines": 1001},
                {"max_unique_ratio": 0.0}, {"max_unique_ratio": 1.01}):
        with pytest.raises(pydantic.ValidationError):
            RepetitionGuardConfig(**bad)


# --- gate behaviour at the acceptance seam ---------------------------------------------------
from localharness.agent.context import ContextManager  # noqa: E402
from localharness.agent.loop import AgentLoop, Session  # noqa: E402
from localharness.agent.permissions import PermissionEvaluator  # noqa: E402
from localharness.core.events import TaskComplete, TurnCompleted, TurnFailed  # noqa: E402


class _ScriptedNoToolLLM:
    """Content-only script, one entry per LLM round-trip; never emits tool calls (mirrors the
    baton-gate suite's fake). tool_registry=None -> no tool_schemas -> the act-guard is out of
    the way, so the repetition guard is exercised in isolation."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0
        class _Cfg: pass
        self.config = _Cfg(); self.config.tool_call_mode = "native"; self.config.context_window = 128000

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        from types import SimpleNamespace as NS
        self.calls += 1
        idx = min(self.calls, len(self._contents)) - 1
        return NS(content=self._contents[idx], tool_calls=None), None


def _make_loop(llm, bus, *, repetition_guard=None):
    overrides = {"name": "rep-agent", "role": "Test agent."}
    if repetition_guard is not None:
        overrides["repetition_guard"] = repetition_guard
    cfg = AgentConfig.model_validate(overrides)
    return AgentLoop(config=cfg, llm=llm, bus=bus, context_manager=ContextManager(),
                     tool_registry=None, permission_evaluator=PermissionEvaluator())


def _rep_nudges(session):
    return [m for m in session.messages if m.get("role") == "user"
            and "same line repeated" in (m.get("content") or "")]


@pytest.mark.asyncio
async def test_degenerate_reply_is_nudged_then_the_healthy_answer_completes(bus, caplog):
    """One corrective nudge, then the recovered answer is what the user gets."""
    llm = _ScriptedNoToolLLM([_LIVE_DEGENERATE, "The top story is the port strike."])
    loop = _make_loop(llm, bus)
    session = Session(agent_id="rep-agent", session_id="s-recover", messages=[])

    with caplog.at_level(logging.WARNING, logger="localharness.agent.loop"):
        summary = await loop._execute_loop(session, "get the news", None)

    assert len(_rep_nudges(session)) == 1
    assert str(_LIVE_REPEATS) in _rep_nudges(session)[0]["content"], \
        "the nudge must name the repeat count"
    assert session.iteration == 2
    assert session.terminated_reason == "complete"
    assert summary == "The top story is the port strike."
    # The WARNING carries the stats, so a live session leaves a diagnosable trace.
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("265" in m and "1" in m for m in warned), warned


@pytest.mark.asyncio
async def test_degenerate_twice_fails_the_task_and_never_completes_with_the_garbage(bus):
    """The bug: the second degeneration must END the turn honestly — no TaskComplete, ever."""
    llm = _ScriptedNoToolLLM([_LIVE_DEGENERATE, _LIVE_DEGENERATE])
    loop = _make_loop(llm, bus)
    session = Session(agent_id="rep-agent", session_id="s-fail", messages=[])
    summary = await loop._execute_loop(session, "get the news", None)

    assert bus.history(event_types=[TaskComplete]) == [], "garbage must never be a completed task"
    assert len(_rep_nudges(session)) == 1, "exactly one nudge, then failure (no loop)"
    assert session.iteration == 2
    assert session.terminated_reason == "error"
    assert "repet" in summary.lower()
    assert summary.count(_LIVE_LINE) <= 1, "the failure notice must not re-ship the 15 KB of garbage"


@pytest.mark.asyncio
async def test_run_turn_surfaces_the_failure_as_turn_failed(bus):
    """Through the real entry point: TurnFailed, not TurnCompleted."""
    llm = _ScriptedNoToolLLM([_LIVE_DEGENERATE, _LIVE_DEGENERATE])
    loop = _make_loop(llm, bus)
    summary = await loop.run_turn("get the news")

    failed = bus.history(event_types=[TurnFailed])
    assert len(failed) == 1 and failed[0].reason == "llm_error"
    assert bus.history(event_types=[TurnCompleted]) == []
    assert bus.history(event_types=[TaskComplete]) == []
    assert "repet" in summary.lower()


@pytest.mark.asyncio
async def test_healthy_answer_is_unaffected(bus):
    """Regression rail: a normal tool-less reply still completes verbatim, no extra round-trip."""
    llm = _ScriptedNoToolLLM(["The top story is the port strike."])
    loop = _make_loop(llm, bus)
    session = Session(agent_id="rep-agent", session_id="s-healthy", messages=[])
    summary = await loop._execute_loop(session, "get the news", None)
    assert session.iteration == 1
    assert _rep_nudges(session) == []
    assert summary == "The top story is the port strike."


@pytest.mark.asyncio
async def test_guard_disabled_restores_verbatim_accept(bus):
    """Kill switch: OFF -> the pre-fix behavior (the repetition is accepted), for A/B work."""
    llm = _ScriptedNoToolLLM([_LIVE_DEGENERATE])
    loop = _make_loop(llm, bus, repetition_guard={"enabled": False})
    session = Session(agent_id="rep-agent", session_id="s-off", messages=[])
    summary = await loop._execute_loop(session, "get the news", None)
    assert session.iteration == 1
    assert _rep_nudges(session) == []
    assert summary == _LIVE_DEGENERATE
