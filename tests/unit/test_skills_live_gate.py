"""The no-model checks of the live skills gate: the bar, the fixture skills, the request sets and the first-step reader."""
from __future__ import annotations

import dataclasses
import getpass
import json
import os
import re
from pathlib import Path

import pytest

from localharness.agent.gate import derive_session_boundary
from localharness.agent.gate_types import PermissionRequest
from localharness.config import skills as skill_config
from localharness.config.skills import LISTING_BUDGET_CHARS, shown
from localharness.core.events import Action
from localharness.provider.client import ProviderAPIError, ProviderConnectionError
from localharness.tools.builtin.skill_tool import render_listing
import tests.integration.test_skills_model_live as live
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient
from tests.integration.test_skills_model_live import (
    CREATOR_BAR, CREATOR_MAX_ACTIONS, CREATOR_MAX_MINUTES, FALSE_BAR, FIXTURES, KINDS, MATCHING_BAR, MAX_ERROR_STREAK, NOT_FINISHED, NOT_SERVING,
    NOT_SUBJECT, RUNS, TEMPERATURE, VARIANT_A, VARIANT_B, CreatorRun, Run, RunStopped, Step, count, creator_asker, creator_run, first_step, first_step_run, hit,
    flaw_of, install, judge, loop_shape, meets, neutral_sitting, not_run_report, out_of_list, passed, prompts, render_report, root_config, run_number, settled,
    sizes, stop_on_failing_server, subject_served, write_report, xml_mode)
from tests.unit.test_image_hygiene import FORBIDDEN
from tests.unit.test_skills_misc import lures

REAL_HOME = Path.home()

URL = "http://localhost:8080/v1"
SKILL = FakeToolCall(id="c1", name="skill", arguments={"name": "release-notes"})
TASK = FakeToolCall(id="c0", name="task", arguments={"action": "show"})


def test_the_bar_is_the_architects():
    assert RUNS == 3
    assert MATCHING_BAR == (24, 30)
    assert FALSE_BAR == (3, 60)
    assert CREATOR_BAR == (2, 3)
    assert TEMPERATURE == 0.6
    assert (CREATOR_MAX_ACTIONS, CREATOR_MAX_MINUTES) == (40, 30.0)
    assert MAX_ERROR_STREAK == 10


def reply(content=None, calls=(), finish="stop", reasoning=None) -> FakeLLMResponse:
    response = FakeLLMResponse(content=content, tool_calls=list(calls), finish_reason=finish)
    response.reasoning_content = reasoning
    return response


async def scripted(tmp_path, monkeypatch, *responses, served=None, mode="native"):
    work = tmp_path / "work"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    skill_set = install(tmp_path / "cfg")
    cfg = root_config(tmp_path / "cfg")
    client = MockLLMClient([*responses, reply("second request")])
    client.config.tool_call_mode = mode
    step = await first_step_run(client, skill_set, cfg, loop_shape(tmp_path / "cfg", cfg, URL, served), "Draft the release notes.", work)
    return step, skill_set, client


async def test_a_scripted_first_step_that_loads_the_skill_is_read_from_a_real_root_turn(tmp_path, monkeypatch):
    step, skill_set, client = await scripted(tmp_path, monkeypatch, reply(calls=[SKILL]))
    assert step.calls == (("skill", {"name": "release-notes"}),)
    assert render_listing(skill_set.skills) in step.system
    assert client._index == 1
    assert step.flaw is None and step.error is None


async def test_a_first_step_that_answers_in_text_calls_nothing(tmp_path, monkeypatch):
    step, _, client = await scripted(tmp_path, monkeypatch, reply("Here are the notes."))
    assert step.calls == ()
    assert isinstance(step, Step)
    assert client._index == 1
    assert step.flaw is None


async def test_the_first_step_keeps_its_calls_in_dispatch_order(tmp_path, monkeypatch):
    step, _, _ = await scripted(tmp_path, monkeypatch, reply(calls=[TASK, SKILL]))
    assert step.calls == (("task", {"action": "show"}), ("skill", {"name": "release-notes"}))


async def test_a_first_response_that_is_no_answer_is_named(tmp_path, monkeypatch):
    cut, _, _ = await scripted(tmp_path / "a", monkeypatch, reply("The release notes are", finish="length"))
    assert (cut.calls, cut.finish_reason, cut.flaw) == ((), "length", "length")
    empty, _, _ = await scripted(tmp_path / "b", monkeypatch, reply())
    assert (empty.calls, empty.flaw) == ((), "empty")
    thought, _, _ = await scripted(tmp_path / "c", monkeypatch, reply(reasoning="Let me think about release notes."))
    assert (thought.calls, thought.flaw) == ((), "reasoning only")


XML_SKILL = '<tool_call>{"name": "skill", "arguments": {"name": "release-notes"}}</tool_call>'


async def test_the_xml_leg_teaches_the_syntax_and_records_the_call_the_loop_dispatched(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    skill_set = install(tmp_path / "cfg")
    cfg = root_config(tmp_path / "cfg")
    registry, _ = await live.root_registry(skill_set, cfg)
    client = MockLLMClient([reply(XML_SKILL), reply("second request")])
    with xml_mode(client):
        taught = client._fn_converter.build_system_injection(registry.global_schemas())
        step = await first_step_run(client, skill_set, cfg, loop_shape(tmp_path / "cfg", cfg, URL, None), "Draft the release notes.", work)
    assert "<tool_call>" in taught and "skill" in taught
    assert step.calls == (("skill", {"name": "release-notes"}),)
    assert step.flaw is None and client._index == 1 and client.config.tool_call_mode == "native"


async def test_a_server_that_answers_natively_in_xml_mode_is_named_not_recorded_as_no_calls(tmp_path, monkeypatch):
    step, _, _ = await scripted(tmp_path, monkeypatch, reply(calls=[SKILL], finish="tool_calls"), mode="xml")
    assert (step.calls, step.finish_reason, step.flaw) == ((), "tool_calls", "native-in-xml")


def test_a_tool_call_finish_with_nothing_dispatched_is_named_by_mode():
    response = Action(agent_id="a", session_id="s", action_type="llm_response", content="", finish_reason="tool_calls")
    assert (flaw_of(response, (), "xml"), flaw_of(response, (), "native")) == ("native-in-xml", "calls not dispatched")
    assert flaw_of(response, (("skill", {}),), "xml") is None


def test_the_xml_leg_teaches_the_syntax_and_restores_the_client():
    client = MockLLMClient([])
    with xml_mode(client):
        assert client.config.tool_call_mode == "xml" and client._fn_converter is not None
    assert client.config.tool_call_mode == "native" and client._fn_converter is None


async def test_the_measured_loop_sends_what_a_real_start_sends(tmp_path, monkeypatch):
    _, _, wide = await scripted(tmp_path / "a", monkeypatch, reply("ok"), served=None)
    _, _, narrow = await scripted(tmp_path / "b", monkeypatch, reply("ok"), served=20000)
    assert wide.calls[0]["effort"] == narrow.calls[0]["effort"] == "high"
    assert 0 < narrow.calls[0]["reasoning_budget"] < wide.calls[0]["reasoning_budget"]


def test_the_loop_shape_is_the_one_start_resolves(tmp_path):
    cfg = root_config(tmp_path / "cfg")
    wide, narrow = loop_shape(tmp_path / "cfg", cfg, URL, None), loop_shape(tmp_path / "cfg", cfg, URL, 20000)
    assert wide.window == cfg.context.max_context_tokens and narrow.window == 20000
    assert wide.effort == ("high", "default") and wide.ratios == (0.125, 0.5)


def test_a_turn_the_model_never_answered_is_not_an_outcome():
    events = [Action(agent_id="a", session_id="s", action_type="tool_call", tool_name="skill", tool_params={"name": "x"})]
    with pytest.raises(AssertionError, match="never answered"):
        first_step(events)


LISTED = {
    "bug-repro-test": "machine", "commit-message": "machine", "docstring-pass": "machine",
    "incident-summary": "machine", "release-notes": "machine", "skill-creator": "bundled",
}

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


QUESTION_WORDS = {
    "what", "who", "whom", "whose", "which", "why", "how", "when", "where", "is", "are", "was", "were", "does", "did", "has", "have", "should"}


def is_question(text: str) -> bool:
    """Opens with a question word; a leading can, could or would is a request."""
    return text.split()[0].lower() in QUESTION_WORDS


def leans(text: str, name: str, own: dict[str, set[str]]) -> bool:
    """Does a request carry the skill's name, hyphen or space, or a word only its description uses?"""
    return name.replace("-", " ") in re.sub(r"[-\s]+", " ", text.casefold()) or bool(words(text) & own[name])


def test_matching_requests_vary_their_wording(tmp_path):
    own = distinctive(install(tmp_path / "cfg"))
    for name, groups in prompts()["skills"].items():
        assert sum(not leans(text, name, own) for text in groups["matching"]) >= 5, name


def test_near_misses_and_matching_requests_are_not_told_apart_by_their_form():
    for name, groups in prompts()["skills"].items():
        assert sum(is_question(text) for text in groups["matching"]) >= 3, name
        assert sum(is_question(text) for text in groups["near_miss"]) <= 4, name


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
    "day": DAY, "run": 1, "commit": "abc1234", "tree": "clean", "sha": "f" * 64, "model": "qwen-27b-swift-1.5", "base_url": URL,
    "window": 131072, "temperature": 0.6, "effort": ("high", "default"), "loop_window": 131072, "tools": ["agent", "skill", "task"],
    "listing": "- release-notes: Draft release notes.", "minutes": 42}


def step_of(skill, kind, called) -> Step:
    calls = (("skill", {"name": skill}),) if called else ()
    return Step(calls, "tool_calls" if called else "stop", "system")


def failed(run: Run, **step) -> Run:
    return dataclasses.replace(run, step=Step((), None, "", **step))


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
    assert count(synthetic(at("release-notes", 23, 0)), "native", "release-notes") == (23, 30, 0, 60)
    assert verdict(at("release-notes", 23, 0)) is False
    assert verdict(at("commit-message", 30, 0)) is True


def test_xml_runs_never_enter_the_bar():
    native = synthetic(at("release-notes", 23, 0))
    for xml in (synthetic(lambda s, k, n: True, "xml", 1), synthetic(lambda s, k, n: False, "xml", 1)):
        assert passed([*native, *xml], [True, True, True]) is False
        assert count([*native, *xml], "native", "release-notes") == count(native, "native", "release-notes")
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
    assert count([beside, own], "native", "commit-message") == (0, 0, 1, 1)
    assert count([beside, own], "native", "release-notes") == (1, 1, 0, 0)


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
    return FakeToolCall(id=f"c{n}", name=tool, arguments=arguments)


class Flaky(MockLLMClient):
    def __init__(self, fails, *responses):
        super().__init__(list(responses))
        self.fails = fails

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kwargs):
        if self.fails:
            self.fails -= 1
            raise RuntimeError("the server went away")
        return await super().stream_complete(messages, tools, on_token, **kwargs)


class Raises(MockLLMClient):
    def __init__(self, exc):
        super().__init__([])
        self.exc = exc

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kwargs):
        raise self.exc


async def settle(tmp_path, monkeypatch, client):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    skill_set = install(tmp_path / "cfg")
    cfg = root_config(tmp_path / "cfg")
    return await settled(client, skill_set, cfg, loop_shape(tmp_path / "cfg", cfg, URL, None), "Draft the release notes.", work)


async def test_a_request_that_fails_once_is_tried_again(tmp_path, monkeypatch):
    step = await settle(tmp_path, monkeypatch, Flaky(1, reply(calls=[SKILL])))
    assert step.error is None and step.calls == (("skill", {"name": "release-notes"}),)


async def test_a_request_that_fails_twice_is_an_error_step_and_the_run_goes_on(tmp_path, monkeypatch):
    client = Flaky(2, reply(calls=[SKILL]))
    step = await settle(tmp_path, monkeypatch, client)
    assert step.calls == () and step.flaw is None
    assert "the server went away" in step.error and "AssertionError" not in step.error
    assert client.fails == 0 and client._index == 0


@pytest.mark.parametrize("exc", [
    ProviderConnectionError("All connection attempts failed"),
    ProviderAPIError("HTTP 400: This model's maximum context length is 32768 tokens.", 400)], ids=["lost connection", "context overflow"])
async def test_a_request_the_loop_retries_internally_records_the_error_it_raised(tmp_path, monkeypatch, exc):
    step = await settle(tmp_path, monkeypatch, Raises(exc))
    assert f"{type(exc).__name__}: {exc}" in step.error and "AssertionError" not in step.error
    assert step.calls == () and step.flaw is None
    matching = Run("native", "release-notes", "matching", "p", 1, step)
    assert (hit(matching), hit(dataclasses.replace(matching, kind="near_miss"))) == (False, True)


def test_an_error_can_only_make_the_bar_harder():
    matching = Run("native", "release-notes", "matching", "p", 1, Step((), None, "", error="ProviderTimeoutError"))
    assert hit(matching) is False
    assert hit(dataclasses.replace(matching, kind="near_miss")) is True
    assert hit(dataclasses.replace(matching, kind="unrelated")) is True
    runs = synthetic(at("release-notes", 24, 0))
    first_hit = next(i for i, r in enumerate(runs) if r.skill == "release-notes" and r.kind == "matching" and hit(r))
    assert passed(runs, [True, True, True]) is True
    assert passed([*runs[:first_hit], failed(runs[first_hit], error="X"), *runs[first_hit + 1:]], [True, True, True]) is False
    miss = next(i for i, r in enumerate(runs) if r.skill == "release-notes" and r.kind == "matching" and not hit(r))
    assert passed([*runs[:miss], failed(runs[miss], error="X"), *runs[miss + 1:]], [True, True, True]) is True
    quiet = synthetic(at("release-notes", 30, 3))
    spare = next(i for i, r in enumerate(quiet) if r.skill == "release-notes" and r.kind == "unrelated" and not hit(r))
    assert passed(quiet, [True, True, True]) is True
    assert passed([*quiet[:spare], failed(quiet[spare], error="X"), *quiet[spare + 1:]], [True, True, True]) is False


async def test_ten_failed_requests_in_a_row_stop_the_run_and_a_good_one_in_between_does_not(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    skill_set = install(tmp_path / "cfg")
    cfg = root_config(tmp_path / "cfg")
    shape = loop_shape(tmp_path / "cfg", cfg, URL, None)
    runs: list[Run] = []
    for n in range(MAX_ERROR_STREAK + 1):
        step = await settled(Flaky(2, reply(calls=[SKILL])), skill_set, cfg, shape, "Draft the release notes.", work)
        runs.append(Run("native", "release-notes", "matching", "p", n, step))
        if n < MAX_ERROR_STREAK - 1:
            stop_on_failing_server(runs)
        else:
            with pytest.raises(RunStopped, match="10 requests in a row errored.*the server went away"):
                stop_on_failing_server(runs)
            break
    assert len(runs) == MAX_ERROR_STREAK and all(r.step.error for r in runs)
    runs[4] = Run("native", "release-notes", "matching", "p", 4, step_of("release-notes", "matching", True))
    stop_on_failing_server(runs)


def test_a_run_the_server_stopped_renders_as_incomplete_with_its_reason_and_no_verdict():
    runs = synthetic(lambda s, k, n: k == "matching")[:40]
    text = render_report(META, runs, [], "stopped after 10 requests in a row errored; the last: boom")
    assert "Outcome: incomplete" in text and "Incomplete because: stopped after 10 requests in a row errored; the last: boom" in text
    assert "Outcome: pass" not in text and "Outcome: miss" not in text and VARIANT_B in text
    assert "| yes |" not in text.split("## XML")[0] and "| n/a |" in text


def test_the_bar_is_written_once():
    assert [meets(m, f) for m, f in [(24, 3), (23, 3), (24, 4), (30, 0)]] == [True, False, False, True]
    assert judge(synthetic(at("release-notes", 24, 3)), [True, True, False]) is True
    assert sizes(synthetic(at("release-notes", 24, 3)), [True, True, False]) == []
    assert sizes([], []) == ["no runs to judge"]


def test_a_report_of_a_short_set_renders_as_incomplete_and_never_raises():
    runs = synthetic(lambda s, k, n: k == "matching")[1:]
    text = render_report(META, runs, creator_runs(True, True))
    assert "Outcome: incomplete" in text and "Incomplete because:" in text and VARIANT_B in text
    assert "Outcome: incomplete" in render_report(META, [], [])


def test_the_report_counts_errors_and_no_answers_per_leg_and_lists_them():
    runs = synthetic(lambda s, k, n: k == "matching")
    runs[0] = failed(runs[0], error="ProviderTimeoutError")
    runs[1] = dataclasses.replace(runs[1], step=Step((), "length", "system", flaw="length"))
    xml = synthetic(lambda s, k, n: k == "matching", "xml", 1)
    xml[2] = dataclasses.replace(xml[2], step=Step((), "stop", "system", flaw="reasoning only"))
    text = render_report(META, [*runs, *xml], creator_runs(True, True, True))
    assert text.count("Count: native: 1, xml: 0") == 1
    assert text.count("Count: native: 1, xml: 1") == 1
    assert "ProviderTimeoutError" in text and "reasoning only" in text
    assert "| bug-repro-test | 28 of 30 | 0 of 60 | yes | 1 | 1 |" in text


def test_the_header_names_the_run_the_tree_the_module_and_the_loop():
    text = render_report({**META, "run": 2, "tree": "dirty"}, synthetic(lambda s, k, n: k == "matching"), creator_runs(True, True, True))
    assert "Run: 2 · Repo: abc1234 (dirty tree) · Gate module sha256: " + "f" * 64 in text
    assert "Loop: effort high (default rung) · context window 131072" in text


def test_the_creator_evidence_reaches_the_report():
    skill = "---\nname: weekly-status\n---\nWrite it up.\n```\ncode\n```\n"
    runs = [CreatorRun(prompts()["creator"], (), "completed", "weekly-status (machine, ok)", True,
                       (("weekly-status/SKILL.md", skill), ("weekly-status/notes.txt", "n")))]
    text = render_report(META, synthetic(lambda s, k, n: k == "matching"), [*runs, *creator_runs(True, False)])
    assert "weekly-status (machine, ok)" in text
    assert "Written `weekly-status/SKILL.md`:" in text and skill in text
    assert "````text" in text
    assert "Other files written: weekly-status/notes.txt" in text and "Other files written: none" in text


def test_the_report_of_a_second_run_never_replaces_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "REPORTS", tmp_path / "reports")
    assert run_number(DAY) == 1
    first = write_report(DAY, "first", 1)
    assert (first.name, run_number(DAY)) == (f"skills-live-gate-{DAY}.md", 2)
    second = write_report(DAY, "second", 2)
    assert (second.name, run_number(DAY)) == (f"skills-live-gate-{DAY}-2.md", 3)
    with pytest.raises(FileExistsError):
        write_report(DAY, "again", 1)
    assert first.read_text(encoding="utf-8") == "first" and second.read_text(encoding="utf-8") == "second"


def test_only_a_qwen_model_is_the_subject_whatever_its_case_and_the_report_names_what_was_served():
    assert [subject_served(m) for m in ("qwen-27b-swift-1.5", "Qwen3.6-27B", "QWEN-27B", "llama-3.3-70b")] == [True, True, True, False]
    assert "Model: llama-3.3-70b" in not_run_report(DAY, "llama-3.3-70b", URL, NOT_SUBJECT)


async def creator_turn(fake_home, *calls, client=None):
    client = client or MockLLMClient([reply(calls=[call(0, "skill", name="skill-creator")]),
                                      *(reply(calls=[c]) for c in calls), reply("Done.")])
    with neutral_sitting(fake_home) as (home, work):
        return await creator_run(client, home, prompts()["creator"], work, URL, None)


async def test_a_scripted_creator_turn_leaves_a_skill_the_verb_lists(tmp_path, fake_home):
    outside = tmp_path / "outside.txt"
    run = await creator_turn(
        fake_home,
        call(1, "write", path="~/.localharness/skills/weekly-status/SKILL.md", content=WEEKLY, mode="overwrite"),
        call(2, "write", path=str(outside), content="x", mode="overwrite"))
    assert run.valid is True
    assert not outside.exists()
    assert "weekly-status (machine, " in run.listing and "Skipped:" not in run.listing
    assert run.files == (("weekly-status/SKILL.md", WEEKLY),)
    allowed = [ask for ask in run.asks if ask[2]]
    assert [ask[0] for ask in allowed] == ["write"] and "weekly-status" in allowed[0][1]
    assert [ask[0] for ask in run.asks if not ask[2]] == ["write"]


async def test_a_creator_turn_that_writes_nothing_is_not_valid(fake_home):
    run = await creator_turn(fake_home)
    assert run.valid is False
    assert "skill-creator (bundled, " in run.listing and " (machine, " not in run.listing
    assert run.asks == () and run.files == ()


class Spy(MockLLMClient):
    def __init__(self, *responses):
        super().__init__(list(responses))
        self.sent: list[str] = []

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kwargs):
        self.sent.append(json.dumps([messages, tools], default=str))
        return await super().stream_complete(messages, tools, on_token, **kwargs)


async def test_nothing_the_creator_is_sent_names_the_real_user_and_every_path_it_sees_is_neutral(fake_home):
    client = Spy(reply(calls=[call(0, "skill", name="skill-creator")]), reply("Done."))
    with neutral_sitting(fake_home) as (home, work):
        assert {os.environ["USER"], os.environ["LOGNAME"]} == {"user"}
        reference = home / "pkg" / "assets" / "skills" / "skill-creator" / "references" / "format.md"
        client._responses.insert(1, reply(calls=[call(1, "read", path=str(reference))]))
        run = await creator_run(client, home, prompts()["creator"], work, URL, None)
        sent = "\n".join(client.sent)
        directory = re.search(r"directory '([^']+)'", sent)
        assert directory and Path(directory.group(1)).is_relative_to(home)
        assert reference.is_file() and run.asks == ()
    assert len(client.sent) == 3 and "The skill format" in client.sent[2]
    for real in {getpass.getuser(), REAL_HOME.name, str(REAL_HOME)}:
        assert real not in sent


async def test_a_creator_turn_that_dies_is_an_invalid_run_not_a_lost_report(fake_home):
    run = await creator_turn(fake_home, client=Flaky(99))
    assert run.valid is False and run.terminated == "internal_error: the server went away"


async def test_a_path_in_the_real_home_is_denied_and_reported_as_written_and_as_resolved(fake_home, tmp_path):
    target = tmp_path / "real-home" / ".localharness" / "skills" / "never-written-probe" / "SKILL.md"
    run = await creator_turn(fake_home, call(1, "write", path=str(target), content=WEEKLY, mode="overwrite"))
    assert run.valid is False and run.files == ()
    [(tool, shown, allowed)] = run.asks
    assert (tool, allowed) == ("write", False)
    assert f"the model wrote {str(target)!r}, resolved to {target}" in shown
    assert not target.exists()


def test_the_sitting_has_a_workspace_in_a_fake_home_and_no_user_name_in_any_path(fake_home):
    here = Path.cwd()
    with neutral_sitting(fake_home) as (home, work):
        base = home.parent
        assert Path.home() == home and Path.cwd() == work and home in work.parents
        assert (work / ".localharness").is_dir()
        assert derive_session_boundary(cwd=work, local_dir=work / ".localharness") == work
        assert REAL_HOME.name not in {*home.parts, *work.parts}
        assert REAL_HOME.name not in str(base)
    assert not base.exists() and Path.cwd() == here


def test_the_sitting_restores_the_package_directory_the_cwd_and_the_environment_when_its_body_raises(fake_home):
    before = Path.cwd(), skill_config.PACKAGE_DIR, os.environ.get("USER"), os.environ.get("LOGNAME")
    with pytest.raises(RuntimeError), neutral_sitting(fake_home) as (home, work):
        assert skill_config.PACKAGE_DIR == home / "pkg"
        raise RuntimeError
    assert (Path.cwd(), skill_config.PACKAGE_DIR, os.environ.get("USER"), os.environ.get("LOGNAME")) == before


def ask(tool, **params) -> PermissionRequest:
    return PermissionRequest(tool_name=tool, tool_params=params, klass="x", key=None, grantable=False, reason="", display=tool)


@pytest.fixture
def skills_dir(tmp_path):
    root = tmp_path / "home" / ".localharness" / "skills"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available")
    return root.resolve()


def denied_asks(skills: Path) -> list[PermissionRequest]:
    home = skills.parent.parent
    return [
        ask("write", path=str(skills / ".." / "config.yaml"), content="x", mode="overwrite"),
        ask("write", path=str(skills / "x" / ".." / ".." / "config.yaml"), content="x", mode="overwrite"),
        ask("write", path=str(skills / "link" / "evil.md"), content="x", mode="overwrite"),
        ask("write", path=str(skills), content="x", mode="overwrite"),
        ask("write", path=str(home / ".bashrc"), content="x", mode="overwrite"),
        ask("write", path=str(home.parent / "outside.txt"), content="x", mode="overwrite"),
        ask("write", path=str(home / ".localharness" / "config.yaml"), content="x", mode="append"),
        ask("write", path="~nosuchuser/x.md", content="x", mode="overwrite"),
        ask("write", path="", content="x", mode="overwrite"),
        ask("write", path=None, content="x", mode="overwrite"),
        ask("write", content="x", mode="overwrite"),
        ask("edit", path=str(skills / "x" / "SKILL.md"), old_string="a", new_string="b"),
        ask("bash_exec", command="rm -rf ~"),
        ask("read", path=str(REAL_HOME / ".localharness" / "config.yaml")),
    ]


async def test_the_creator_asker_denies_everything_but_a_write_under_the_skills_directory(skills_dir):
    asks: list = []
    asker = creator_asker(skills_dir, asks)
    for request in denied_asks(skills_dir):
        assert (await asker(request)).kind == "reject_once", request
    allowed = ask("write", path=str(skills_dir / "x" / "SKILL.md"), content="x", mode="overwrite")
    assert (await asker(allowed)).kind == "allow_once"
    assert [a[2] for a in asks] == [False] * len(denied_asks(skills_dir)) + [True]
