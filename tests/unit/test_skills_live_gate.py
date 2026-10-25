"""The no-model checks of the live skills gate: the bar, the fixture skills, the request sets and the first-step reader."""
from __future__ import annotations

import re

import pytest

from localharness.config.skills import LISTING_BUDGET_CHARS, shown
from localharness.core.events import Action
from localharness.tools.builtin.skill_tool import render_listing
from tests.integration.test_injection_redteam import MockLLMClient
import tests.integration.test_skills_model_live as live
from tests.integration.test_skills_model_live import (
    CREATOR_BAR, FALSE_BAR, FIXTURES, MATCHING_BAR, RUNS, TEMPERATURE, VARIANT_A, CreatorRun, Run, Step,
    first_step, first_step_run, hit, install, passed, prompts, render_report, root_config, write_report)
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


DAY = "2026-10-24"
NAMES = sorted(LISTED)
META = {
    "day": DAY, "commit": "abc1234", "model": "qwen-27b-swift-1.5", "base_url": "http://localhost:8080/v1", "window": 131072,
    "temperature": 0.6, "tools": ["agent", "skill", "task"], "listing": "- release-notes: Draft release notes.", "minutes": 42}


def step_of(skill, kind, called) -> Step:
    calls = (("skill", {"name": skill}),) if called else ()
    return Step(calls, "tool_calls" if called else "stop", "system")


def synthetic(hits, leg="native", repeats=RUNS) -> list[Run]:
    """Runs for the six listed names; hits(skill, kind, n) says whether the n-th run of that skill and kind called skill."""
    seen: dict[tuple[str, str], int] = {}
    runs = []
    for index in range(1, repeats + 1):
        for skill in NAMES:
            for kind in KINDS:
                for prompt in prompts()["skills"][skill][kind]:
                    n = seen[skill, kind] = seen.get((skill, kind), -1) + 1
                    runs.append(Run(leg, skill, kind, prompt, index, step_of(skill, kind, hits(skill, kind, n))))
    return runs


def creator_runs(*valid) -> list[CreatorRun]:
    return [CreatorRun(prompts()["creator"], (("write", "write: a skill file", ok),), "completed", "listing", ok) for ok in valid]


def test_synthetic_outcomes_are_judged_and_written_as_a_dated_report(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "REPORTS", tmp_path / "reports")
    runs = synthetic(lambda skill, kind, n: kind == "matching")
    creator = creator_runs(True, True, True)
    assert passed(runs, [c.valid for c in creator]) is True
    text = render_report(META, runs, creator)
    assert "Outcome: pass" in text and VARIANT_A in text
    assert text.count("30 of 30") == 6 and text.count("0 of 60") == 6
    assert META["listing"] in text
    assert all(text.count(request) >= RUNS for request in prompts()["skills"]["release-notes"]["matching"])
    path = write_report(DAY, text)
    assert path == tmp_path / "reports" / f"skills-live-gate-{DAY}.md"
    assert path.read_text(encoding="utf-8") == text


def test_a_matching_run_passes_only_on_the_matching_name():
    def matching(called):
        return Run("native", "release-notes", "matching", "p", 1, Step((("skill", {"name": called}),), "tool_calls", ""))

    assert hit(matching("release-notes")) is True
    assert hit(matching("commit-message")) is False
