"""The no-model checks of the live skills gate: the bar, the fixture skills, the request sets and the first-step reader."""
from __future__ import annotations

import dataclasses
import re

import pytest

from localharness.config.skills import LISTING_BUDGET_CHARS, shown
from localharness.core.events import Action
from localharness.tools.builtin.skill_tool import render_listing
from tests.integration.test_injection_redteam import MockLLMClient
import tests.integration.test_skills_model_live as live
from tests.integration.test_skills_model_live import (
    CREATOR_BAR, FALSE_BAR, FIXTURES, MATCHING_BAR, RUNS, TEMPERATURE, NOT_FINISHED, NOT_SERVING, NOT_SUBJECT, VARIANT_A, VARIANT_B, CreatorRun, Run, Step,
    creator_run, first_step, first_step_run, hit, install, not_run_report, out_of_list, passed, prompts, render_report, root_config,
    tally, write_report)
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


def at(skill, matching, others):
    """Every skill hits all its matching runs and no other; `skill` hits the first `matching` and `others` of its near-misses."""
    def hits(name, kind, n):
        if name != skill:
            return kind == "matching"
        return n < matching if kind == "matching" else kind == "near_miss" and n < others
    return hits


def verdict(hits, valid=(True, True, True), extra=()):
    return passed([*synthetic(hits), *extra], list(valid))


def test_the_bar_holds_at_its_thresholds():
    assert verdict(at("release-notes", 24, 0)) is True
    assert verdict(at("release-notes", 23, 0)) is False
    assert verdict(at("release-notes", 30, 3)) is True
    assert verdict(at("release-notes", 30, 4)) is False
    assert verdict(at("release-notes", 24, 3)) is True
    assert verdict(at("release-notes", 30, 0), valid=(True, True, False)) is True
    assert verdict(at("release-notes", 30, 0), valid=(True, False, False)) is False
    assert verdict(at("release-notes", 30, 0), valid=(False, False, False)) is False


def test_one_skill_below_the_bar_misses_the_gate():
    assert tally(synthetic(at("release-notes", 23, 0)), "release-notes") == (23, 30, 0, 60)
    assert verdict(at("release-notes", 23, 0)) is False
    assert verdict(at("commit-message", 30, 0)) is True


def test_xml_runs_never_enter_the_bar():
    native = synthetic(at("release-notes", 23, 0))
    for xml in (synthetic(lambda s, k, n: True, "xml", 1), synthetic(lambda s, k, n: False, "xml", 1)):
        assert passed([*native, *xml], [True, True, True]) is False
        assert tally([*native, *xml], "release-notes") == tally(native, "release-notes")
    good = synthetic(at("release-notes", 30, 0))
    assert passed([*good, *synthetic(lambda s, k, n: k != "matching", "xml", 1)], [True, True, True]) is True


def test_a_set_of_the_wrong_size_is_refused():
    runs = synthetic(at("release-notes", 30, 0))
    with pytest.raises(AssertionError):
        passed(runs[1:], [True, True, True])
    with pytest.raises(AssertionError):
        passed(runs, [True, True, True, True])
    with pytest.raises(AssertionError):
        passed(runs, [True, True])
    with pytest.raises(AssertionError):
        passed([], [True, True, True])


def odd_run(leg, called, kind="matching", prompt="p") -> Run:
    return Run(leg, "release-notes", kind, prompt, 1, Step((("skill", called),), "tool_calls", ""))


ODD = [{"name": "Release-Notes"}, {"name": " release-notes"}, {"name": "made-up"}, {}, {"name": 7}]
ODD_REPR = ["'Release-Notes'", "' release-notes'", "'made-up'", "None", "7"]


def test_out_of_list_names_are_kept_per_leg():
    native = [odd_run("native", called, prompt=f"p{i}") for i, called in enumerate(ODD)]
    assert out_of_list(native, NAMES) == {"native": [(f"p{i}", r) for i, r in enumerate(ODD_REPR)], "xml": []}
    xml = [odd_run("xml", called, prompt=f"p{i}") for i, called in enumerate(ODD)]
    assert out_of_list(xml, NAMES) == {"native": [], "xml": [(f"p{i}", r) for i, r in enumerate(ODD_REPR)]}
    assert out_of_list([odd_run("native", {"name": "release-notes"})], NAMES) == {"native": [], "xml": []}
    assert [hit(odd_run("native", called)) for called in ODD[:2]] == [False, False]


def test_a_skill_call_beside_another_in_the_first_step_counts():
    beside = Run("native", "release-notes", "matching", "p", 1,
                 Step((("read", {"path": "notes.md"}), ("skill", {"name": "release-notes"})), "tool_calls", ""))
    assert hit(beside) is True
    own = Run("native", "commit-message", "near_miss", "p", 1, Step((("skill", {"name": "commit-message"}),), "tool_calls", ""))
    assert hit(own) is True
    assert tally([beside, own], "commit-message") == (0, 0, 1, 1)
    assert tally([beside, own], "release-notes") == (1, 1, 0, 0)


@pytest.mark.parametrize("reason", [NOT_SERVING, NOT_SUBJECT, NOT_FINISHED])
def test_a_report_that_did_not_run_holds_no_number(reason):
    text = not_run_report("2026-10-24", "qwen3.8-27b", "http://localhost:8080/v1", reason)
    assert "Outcome: not run" in text and reason in text and VARIANT_B in text
    assert not [line for line in text.splitlines() if re.search(r"\d", line) and not line.startswith(("Date:", "Model:", "Endpoint:"))]
    assert not any(needle in text for needle in ("of 30", "of 60", "%"))
    assert "Model: none configured" in not_run_report("2026-10-24", None, None, reason)
    assert "Endpoint: none configured" in not_run_report("2026-10-24", None, None, reason)


def test_the_fixed_sentences_hold_no_digit():
    assert not any(re.search(r"\d", s) for s in (NOT_SERVING, NOT_SUBJECT, NOT_FINISHED, VARIANT_B))


def test_a_report_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "REPORTS", tmp_path / "reports")
    path = write_report(DAY, "first")
    with pytest.raises(FileExistsError):
        write_report(DAY, "second")
    assert path.read_text(encoding="utf-8") == "first"


def test_a_table_cell_cannot_break_its_row():
    runs = synthetic(lambda s, k, n: k == "matching")
    runs[0] = dataclasses.replace(runs[0], prompt="a | b\nc")
    text = render_report(META, runs, creator_runs(True, True, True))
    rows = [line for line in text.splitlines() if "a \\| b c" in line]
    assert len(rows) == 1 and rows[0].startswith("| native |") and rows[0].endswith(" |")
    assert "a | b" not in text and "b\nc" not in text


def test_a_report_names_the_shape_of_the_root():
    text = render_report(META, synthetic(lambda s, k, n: k == "matching"), creator_runs(True, True, True))
    for fragment in ("no plugins", "no memory", "no guardrails file", "no project instructions", "web verbs are denied"):
        assert fragment in text
    assert "Window: 131072" in text
    assert "Window: not exposed by the server" in render_report(
        {**META, "window": None}, synthetic(lambda s, k, n: k == "matching"), creator_runs(True, True, True))


WEEKLY = (
    "---\nname: weekly-status\n"
    "description: Turns pasted notes into a weekly status update under Done, Doing and Blocked. Use when the person pastes notes for a status.\n"
    "---\n# Weekly status\n\n1. Group each item under Done, Doing or Blocked.\n2. End with one line naming the biggest risk.\n")


def call(n, tool, **arguments):
    return {"name": tool, "arguments": arguments, "id": f"c{n}"}


async def creator_turn(tmp_path, monkeypatch, fake_home, *calls):
    home = fake_home(tmp_path / "home")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    client = MockLLMClient([{"content": None, "tool_calls": [call(0, "skill", name="skill-creator")]},
                            *({"content": None, "tool_calls": [c]} for c in calls),
                            {"content": "Done.", "tool_calls": None}])
    return home, await creator_run(client, home, prompts()["creator"], work)


async def test_a_scripted_creator_turn_leaves_a_skill_the_verb_lists(tmp_path, monkeypatch, fake_home):
    outside = tmp_path / "outside.txt"
    home, run = await creator_turn(
        tmp_path, monkeypatch, fake_home,
        call(1, "write", path="~/.localharness/skills/weekly-status/SKILL.md", content=WEEKLY, mode="overwrite"),
        call(2, "write", path=str(outside), content="x", mode="overwrite"))
    assert run.valid is True
    assert (home / ".localharness" / "skills" / "weekly-status" / "SKILL.md").read_text(encoding="utf-8") == WEEKLY
    assert not outside.exists()
    assert "weekly-status (machine, " in run.listing and "Skipped:" not in run.listing
    allowed = [ask for ask in run.asks if ask[2]]
    assert [ask[0] for ask in allowed] == ["write"] and "weekly-status" in allowed[0][1]
    assert [ask[0] for ask in run.asks if not ask[2]] == ["write"]


async def test_a_creator_turn_that_writes_nothing_is_not_valid(tmp_path, monkeypatch, fake_home):
    home, run = await creator_turn(tmp_path, monkeypatch, fake_home)
    assert run.valid is False
    assert "skill-creator (bundled, " in run.listing and " (machine, " not in run.listing
    assert run.asks == ()
    assert not (home / ".localharness" / "skills").exists()
