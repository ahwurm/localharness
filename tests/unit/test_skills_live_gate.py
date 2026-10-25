"""The no-model checks of the live skills gate: the bar, the fixture skills, the request sets and the first-step reader."""
from __future__ import annotations

import pytest

from localharness.core.events import Action
from localharness.tools.builtin.skill_tool import render_listing
from tests.integration.test_injection_redteam import MockLLMClient
from tests.integration.test_skills_model_live import (
    CREATOR_BAR, FALSE_BAR, MATCHING_BAR, RUNS, TEMPERATURE, Step, first_step, first_step_run, install, root_config)

SKILL = {"name": "skill", "arguments": {"name": "release-notes"}, "id": "c1"}
TASK = {"name": "task", "arguments": {"action": "show"}, "id": "c0"}


def test_the_bar_is_the_architects():
    assert RUNS == 3
    assert MATCHING_BAR == (24, 30)
    assert FALSE_BAR == (3, 60)
    assert CREATOR_BAR == (2, 3)
    assert TEMPERATURE == 0.6


async def scripted(tmp_path, monkeypatch, *responses):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    skill_set = install(tmp_path / "cfg")
    client = MockLLMClient([*responses, {"content": "second request", "tool_calls": None}])
    step = await first_step_run(client, skill_set, root_config(tmp_path / "cfg"), "Draft the release notes.", work)
    return step, skill_set, client


async def test_a_scripted_first_step_that_loads_the_skill_is_read_from_a_real_root_turn(tmp_path, monkeypatch):
    step, skill_set, client = await scripted(tmp_path, monkeypatch, {"content": None, "tool_calls": [SKILL]})
    assert step.calls == (("skill", {"name": "release-notes"}),)
    assert render_listing(skill_set.skills) in step.system
    assert client._index == 1


async def test_a_first_step_that_answers_in_text_calls_nothing(tmp_path, monkeypatch):
    step, _, client = await scripted(tmp_path, monkeypatch, {"content": "Here are the notes.", "tool_calls": None})
    assert step.calls == ()
    assert isinstance(step, Step)
    assert client._index == 1


async def test_the_first_step_keeps_its_calls_in_dispatch_order(tmp_path, monkeypatch):
    step, _, _ = await scripted(tmp_path, monkeypatch, {"content": None, "tool_calls": [TASK, SKILL]})
    assert step.calls == (("task", {"action": "show"}), ("skill", {"name": "release-notes"}))


def test_a_turn_the_model_never_answered_is_not_an_outcome():
    events = [Action(agent_id="a", session_id="s", action_type="tool_call", tool_name="skill", tool_params={"name": "x"})]
    with pytest.raises(AssertionError, match="never answered"):
        first_step(events)
