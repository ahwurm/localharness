"""The no-model checks of the live skills gate: the bar, the fixture skills, the request sets and the first-step reader."""
from __future__ import annotations

import re

import pytest

from localharness.config.skills import LISTING_BUDGET_CHARS, shown
from localharness.core.events import Action
from localharness.tools.builtin.skill_tool import render_listing
from tests.integration.test_injection_redteam import MockLLMClient
from tests.integration.test_skills_model_live import (
    CREATOR_BAR, FALSE_BAR, FIXTURES, MATCHING_BAR, RUNS, TEMPERATURE, Step, first_step, first_step_run, install,
    prompts, root_config)
from tests.unit.test_image_hygiene import FORBIDDEN
from tests.unit.test_skills_misc import lures

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


LISTED = {
    "bug-repro-test": "machine", "commit-message": "machine", "docstring-pass": "machine",
    "incident-summary": "machine", "release-notes": "machine", "skill-creator": "bundled",
}
KINDS = ("matching", "near_miss", "unrelated")


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def requests() -> list[tuple[str, str, str]]:
    sets = prompts()["skills"]
    return [(name, kind, text) for name, groups in sets.items() for kind in KINDS for text in groups[kind]] + [
        ("creator", "creator", prompts()["creator"])]


def distinctive(skill_set) -> dict[str, set[str]]:
    own = {s.name: {w for w in words(s.description) if len(w) >= 3} for s in skill_set.skills}
    return {name: {w for w in found if all(w not in other for n, other in own.items() if n != name)}
            for name, found in own.items()}


def test_the_fixture_skills_and_the_bundled_creator_list_with_every_description(tmp_path):
    skill_set = install(tmp_path / "cfg")
    assert {s.name: s.origin for s in skill_set.skills} == LISTED
    assert skill_set.skipped == ()
    assert all(description is not None for _, _, description, _ in shown(skill_set.skills))
    assert sum(len(s.description) for s in skill_set.skills) <= LISTING_BUDGET_CHARS


def test_every_listed_skill_has_ten_requests_of_each_kind_and_no_request_repeats(tmp_path):
    sets = prompts()["skills"]
    assert set(sets) == set(LISTED)
    assert all(len(groups[kind]) == 10 for groups in sets.values() for kind in KINDS)
    creator = prompts()["creator"]
    assert isinstance(creator, str) and creator.strip() and "\n" not in creator.strip()
    texts = [text.strip().casefold() for _, _, text in requests()]
    assert len(texts) == 181 and len(set(texts)) == 181


def test_near_misses_share_a_word_only_their_skill_uses(tmp_path):
    own = distinctive(install(tmp_path / "cfg"))
    bare = [text for name, kind, text in requests() if kind == "near_miss" and not words(text) & own[name]]
    assert bare == []


def test_matching_requests_vary_their_wording(tmp_path):
    own = distinctive(install(tmp_path / "cfg"))
    for name, groups in prompts()["skills"].items():
        assert sum(not words(text) & own[name] for text in groups["matching"]) >= 3, name


def test_unrelated_requests_share_no_word_with_any_description(tmp_path):
    seen = {w for s in install(tmp_path / "cfg").skills for w in words(s.description) if len(w) >= 5}
    shared = {text: sorted(found) for name, kind, text in requests() if kind == "unrelated"
              if (found := {w for w in words(text) if len(w) >= 5} & seen)}
    assert shared == {}


def test_no_request_names_a_skill(tmp_path):
    for name, kind, text in requests():
        assert not any(listed in text.casefold() for listed in LISTED), text
        if name not in ("skill-creator", "creator"):
            assert not words(text) & {"skill", "skills"}, text


def test_no_fixture_skill_carries_an_install_or_fetch_step():
    files = [*(FIXTURES / "skills").glob("*/SKILL.md"), FIXTURES / "prompts.yaml"]
    assert len(files) == 6
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert lures(text) == [], path
        assert not [needle for needle in FORBIDDEN if needle in text], path
