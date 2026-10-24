"""Slash rows for skills: the dispatch seam and late rows."""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.cli.skill_session import SkillSession, skill_row
from localharness.cli.slash_commands import (
    SLASH_COMMANDS,
    SlashCommand,
    add_plugin_row,
    all_rows,
    set_plugin_rows,
)
from localharness.config import skills
from localharness.config.skills import SkillSet
from localharness.core.bus import EventBus
from localharness.core.events import Action, TurnCompleted, TurnFailed
from tests.unit.test_repl_input_box import _pending_turn, _repl as _box_repl
from tests.unit.test_repl_unknown_slash import RecordingChannel, _build_repl
from tests.unit.test_skills_discovery import _skill as write_skill


@pytest.fixture
def rows():
    assert all_rows() == SLASH_COMMANDS, "a plugin row leaked in from an earlier test"
    yield set_plugin_rows
    set_plugin_rows(())


async def test_a_plugin_table_row_with_a_method_name_is_called_with_its_row_and_args(rows):
    row = SlashCommand("/example", "an example skill", "_slash_example", takes_args=True, plugin="skill")
    assert rows([row]) == []
    repl, agent, _ = _build_repl(RecordingChannel([]))
    repl._slash_example = AsyncMock()
    assert await repl._handle_slash("/EXAMPLE  Hi There ") is True
    repl._slash_example.assert_awaited_once_with(row, "Hi There")
    repl._slash_example.reset_mock()
    assert await repl._handle_slash("/example") is True
    repl._slash_example.assert_awaited_once_with(row, "")
    agent.run_turn.assert_not_called()


def _skill(name: str) -> SlashCommand:
    return SlashCommand(f"/{name}", "x", "_slash_skill", takes_args=True, plugin="skill")


def _plugin(name: str, plugin: str = "example") -> SlashCommand:
    return SlashCommand(f"/{name}", "x", AsyncMock(), plugin=plugin)


def test_warnings_are_worded_by_the_rows_kind(rows):
    assert rows([_skill("help"), _plugin("help", "evil")]) == [
        'skill help: slash command /help is already taken — say "use the help skill", or rename it',
        "plugin evil: slash command /help is already taken — skipped",
    ]
    assert all_rows() == SLASH_COMMANDS


def test_empty_rows_leave_only_the_core_table(rows):
    rows([_plugin("example")])
    assert rows(()) == [] and all_rows() == SLASH_COMMANDS


def test_a_late_row_is_appended_after_every_earlier_plugin_row(rows):
    example, release, third = _plugin("example"), _skill("release-notes"), _skill("deploy")
    rows([example])
    assert add_plugin_row(release) is None
    assert all_rows() == SLASH_COMMANDS + (example, release)
    assert add_plugin_row(third) is None
    assert all_rows() == SLASH_COMMANDS + (example, release, third)


def test_a_late_row_with_no_plugin_rows_installed_is_the_only_one(rows):
    row = _skill("release-notes")
    assert add_plugin_row(row) is None
    assert all_rows() == SLASH_COMMANDS + (row,)


@pytest.mark.parametrize(("late", "warning"), [
    (_skill("example"), 'skill example: slash command /example is already taken — say "use the example skill", or rename it'),
    (_skill("help"), 'skill help: slash command /help is already taken — say "use the help skill", or rename it'),
    (_plugin("model", "twin"), "plugin twin: slash command /model is already taken — skipped"),
    (SlashCommand("/odd", "x", None, plugin="skill"), "skill odd: slash command /odd has no callable handler — skipped"),
    (SlashCommand("/odd", "x", None, plugin="twin"), "plugin twin: slash command /odd has no callable handler — skipped"),
])
def test_a_refused_late_row_returns_its_warning_and_changes_nothing(rows, late, warning):
    rows([_plugin("example")])
    before = all_rows()
    assert add_plugin_row(late) == warning
    assert all_rows() == before
    assert not warning.startswith("plugin ") or late.plugin != "skill"


def test_the_taken_name_check_is_exact(rows):
    rows([_plugin("example")])
    assert add_plugin_row(_skill("example-two")) is None
    assert [row.name for row in all_rows()][-2:] == ["/example", "/example-two"]


@pytest.mark.parametrize("handler", ["_slash_skill", AsyncMock()])
def test_a_row_naming_no_plugin_is_skipped_in_both_entry_points(rows, handler):
    row = SlashCommand("/orphan", "x", handler)
    warning = "slash command /orphan names no plugin — skipped"
    assert rows([row]) == [warning]
    assert add_plugin_row(row) == warning
    assert all_rows() == SLASH_COMMANDS


async def test_a_callable_on_a_late_row_gets_the_stripped_text_only(rows):
    handler = AsyncMock(return_value=None)
    assert add_plugin_row(SlashCommand("/late", "x", handler, takes_args=True, plugin="late")) is None
    repl, _, _ = _build_repl(RecordingChannel([]))
    assert await repl._handle_slash("/late  Hi There ") is True
    handler.assert_awaited_once_with("Hi There")


def test_the_dispatcher_does_not_key_on_the_kind():
    tree = ast.parse(textwrap.dedent(inspect.getsource(OrchestratorREPL._handle_slash)))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "skill"]


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "PACKAGE_DIR", tmp_path / "pkg")
    return tmp_path / "pkg"


class Sitting:
    def __init__(self, cfg, names=("notes",)):
        self.cfg = cfg
        self.paths = {n: write_skill(cfg / "skills", n) for n in names}
        self.found = SkillSet.discover(cfg, None)
        self.session = SkillSession(self.found)
        self.bus = EventBus()
        self.said = []
        self.handles = self.session.subscribe(self.bus, "orchestrator", self._say)

    async def _say(self, text):
        self.said.append(text)

    def record(self, name="notes"):
        return next(s for s in self.found.skills if s.name == name)

    async def call(self, name, agent="orchestrator", tool="skill"):
        await self.bus.publish(Action(
            agent_id=agent, session_id="s1", action_type="tool_call", tool_name=tool, tool_params={"name": name},
        ))

    async def act(self, tool, **params):
        await self.bus.publish(Action(
            agent_id="orchestrator", session_id="s1", action_type="tool_call", tool_name=tool, tool_params=params,
        ))

    async def completed(self, agent="orchestrator"):
        await self.bus.publish(TurnCompleted(
            agent_id=agent, session_id="s1", iterations=1, duration_seconds=0.1, elapsed_tokens=0, summary="",
        ))

    async def failed(self):
        await self.bus.publish(TurnFailed(
            agent_id="orchestrator", session_id="s1", reason="llm_error", detail="x", iterations=1, duration_seconds=0.1,
        ))


@pytest.fixture
def sitting(tmp_path, pkg):
    return Sitting(tmp_path / "cfg")


async def test_a_slash_name_the_model_did_not_load_is_said_once(sitting):
    assert len(sitting.handles) == 3
    line, notices = sitting.session.invoke(sitting.record(), "")
    assert (line, notices) == ("The person invoked /notes. Call skill('notes') and follow it.", ())
    assert "Body of notes." not in line
    assert sitting.session.invoke(sitting.record(), "draft v2")[0] == (
        "The person invoked /notes with: draft v2. Call skill('notes') and follow it."
    )
    sitting.session.arm()
    await sitting.completed()
    await sitting.completed()
    assert sitting.said == ['the model did not load /notes; try again, or say "use the notes skill"']


async def test_a_skill_call_or_a_failed_turn_says_nothing(sitting):
    sitting.session.invoke(sitting.record(), "")
    sitting.session.arm()
    await sitting.call("notes")
    await sitting.completed()
    sitting.session.invoke(sitting.record(), "")
    sitting.session.arm()
    await sitting.failed()
    await sitting.completed()
    assert sitting.said == []


def _text(name, mark):
    return f"---\nname: {name}\ndescription: Use {name} when asked.\n---\nBody {mark}.\n"


def _changed(path):
    return (f"changed since first use: {str(path)!r}; using it",)


def test_each_distinct_body_is_reported_once_as_changed(sitting):
    path, record = sitting.paths["notes"], sitting.record()
    assert sitting.session.invoke(record, "")[1] == ()
    path.write_text(_text("notes", "B"))
    assert sitting.session.invoke(record, "")[1] == _changed(path)
    assert sitting.session.invoke(record, "")[1] == ()


def test_the_first_body_seen_again_is_not_reported(sitting):
    path, record = sitting.paths["notes"], sitting.record()
    first = path.read_text()
    sitting.session.invoke(record, "")
    path.write_text(_text("notes", "B"))
    assert sitting.session.invoke(record, "")[1] == _changed(path)
    path.write_text(first)
    assert sitting.session.invoke(record, "")[1] == ()
    path.write_text(_text("notes", "C"))
    assert sitting.session.invoke(record, "")[1] == _changed(path)


def test_a_skill_found_after_start_is_added_once(sitting):
    path = write_skill(sitting.cfg / "skills", "fresh")
    record = sitting.session.find("fresh")
    assert sitting.session.invoke(record, "")[1] == (f"added since start: fresh (machine, {str(path)!r})",)
    assert sitting.session.invoke(record, "")[1] == ()


@pytest.mark.parametrize("params", [{}, {"mode": "overwrite"}])
async def test_an_overwrite_write_of_the_new_text_keeps_the_changed_notice_quiet(sitting, params):
    path, record = sitting.paths["notes"], sitting.record()
    sitting.session.invoke(record, "")
    path.write_text(_text("notes", "B"))
    await sitting.act("write", path=str(path), content=_text("notes", "B"), **params)
    assert sitting.session.invoke(record, "")[1] == ()
    path.write_text(_text("notes", "C"))
    assert sitting.session.invoke(record, "")[1] == _changed(path)


async def test_a_creator_write_followed_by_the_name_is_silent(sitting):
    path = sitting.cfg / "skills" / "fresh" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(_text("fresh", "F"))
    await sitting.act("write", path=str(path), content=_text("fresh", "F"))
    assert sitting.session.invoke(sitting.session.find("fresh"), "")[1] == ()


@pytest.mark.parametrize(
    "tool, params",
    [
        ("write", {"content": _text("notes", "B2")}),
        ("write", {"content": _text("notes", "B"), "mode": "append"}),
        ("write", {"content": _text("notes", "B"), "mode": "Overwrite"}),
        ("edit", {"old_string": "Body of notes.", "new_string": "Body B."}),
    ],
)
async def test_a_write_of_other_content_an_append_or_an_edit_suppresses_nothing(sitting, tool, params):
    path, record = sitting.paths["notes"], sitting.record()
    sitting.session.invoke(record, "")
    path.write_text(_text("notes", "B"))
    await sitting.act(tool, path=str(path), **params)
    assert sitting.session.invoke(record, "")[1] == _changed(path)


async def test_an_equal_write_under_another_directory_suppresses_nothing(tmp_path, pkg):
    sitting = Sitting(tmp_path / "cfg", ("notes", "notes-two"))
    path, record = sitting.paths["notes"], sitting.record()
    sitting.session.invoke(record, "")
    path.write_text(_text("notes", "B"))
    await sitting.act("write", path=str(sitting.paths["notes-two"]), content=_text("notes", "B"))
    assert sitting.session.invoke(record, "")[1] == _changed(path)
    await sitting.act("write", path=str(path.parent / "references" / "x.md"), content=_text("notes", "B"))
    path.write_text(_text("notes", "D"))
    await sitting.act("write", path=str(path.parent / "references" / "x.md"), content=_text("notes", "D"))
    assert sitting.session.invoke(record, "")[1] == ()


async def test_a_write_of_the_old_text_while_the_file_holds_another_suppresses_nothing(sitting):
    path, record = sitting.paths["notes"], sitting.record()
    first = path.read_text()
    sitting.session.invoke(record, "")
    path.write_text(_text("notes", "B"))
    await sitting.act("write", path=str(path), content=first)
    assert sitting.session.invoke(record, "")[1] == _changed(path)


async def test_a_relative_write_path_is_resolved_against_the_working_directory(sitting, monkeypatch):
    path, record = sitting.paths["notes"], sitting.record()
    sitting.session.invoke(record, "")
    path.write_text(_text("notes", "B"))
    monkeypatch.chdir(sitting.cfg)
    await sitting.act("write", path="skills/notes/SKILL.md", content=_text("notes", "B"))
    assert sitting.session.invoke(record, "")[1] == ()


async def test_a_leading_bom_does_not_defeat_the_digest(sitting):
    path, record = sitting.paths["notes"], sitting.record()
    sitting.session.invoke(record, "")
    path.write_bytes(b"\xef\xbb\xbf" + _text("notes", "B").encode())
    await sitting.act("write", path=str(path), content=_text("notes", "B"))
    assert sitting.session.invoke(record, "")[1] == ()


async def test_a_model_triggered_first_load_records_no_digest(sitting):
    path, record = sitting.paths["notes"], sitting.record()
    await sitting.call("notes")
    path.write_text(_text("notes", "B"))
    assert sitting.session.invoke(record, "")[1] == ()


async def test_a_deleted_skill_is_skipped_and_arms_nothing(sitting):
    path, record = sitting.paths["notes"], sitting.record()
    path.unlink()
    assert isinstance(sitting.session.invoke(record, ""), skills.Skipped)
    sitting.session.arm()
    await sitting.completed()
    assert sitting.said == []


async def test_a_write_from_an_earlier_turn_still_suppresses(sitting):
    path, record = sitting.paths["notes"], sitting.record()
    sitting.session.invoke(record, "")
    path.write_text(_text("notes", "B"))
    await sitting.act("write", path=str(path), content=_text("notes", "B"))
    sitting.session.arm()
    await sitting.completed()
    assert sitting.session.invoke(record, "")[1] == ()


def _row(description, origin="machine", name="notes"):
    return skill_row(skills.Skill(name, description, origin, Path(f"/x/{name}/SKILL.md"), origin == "project"))


def test_a_skill_row_carries_the_first_sentence_and_takes_args():
    row = _row("Use notes when asked. Then more.")
    assert row == SlashCommand("/notes", "Use notes when asked.", "_slash_skill", takes_args=True, plugin="skill")
    assert row.terminal_only is False
    assert _row("Use notes when asked. Then more.", "project").description == "Use notes when asked. (project)"


@pytest.mark.parametrize(
    "description, shown",
    [("Ship it! Now.", "Ship it!"), ("Is it? Yes.", "Is it?"), ("Uses v1.2 notes", "Uses v1.2 notes"), ("Ends here.", "Ends here.")],
)
def test_a_skill_row_cuts_at_a_sentence_end_followed_by_space_or_the_end(description, shown):
    assert _row(description).description == shown


def test_a_skill_row_is_cut_to_eighty_characters_and_the_label_is_not_counted():
    assert _row("é" * 100).description == "é" * 80
    assert _row("a" * 79 + ". tail").description == "a" * 79 + "."
    assert _row("a" * 79 + "!" + " tail").description == "a" * 79 + "!"
    assert _row("a" * 80 + ". tail").description == "a" * 80
    assert _row("b" * 90, "project").description == "b" * 80 + " (project)"


def test_the_session_is_the_one_caller_of_the_set_find_in_cli():
    def finds(path):
        return [
            n for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "find"
            and (getattr(n.func.value, "attr", None) or getattr(n.func.value, "id", "")).endswith("skill_set")
        ]

    cli = Path(inspect.getfile(SkillSession)).parent
    assert len(finds(cli / "skill_session.py")) == 1
    assert [p.name for p in cli.glob("*.py") if p.name != "skill_session.py" and finds(p)] == []


def test_the_session_find_returns_what_the_set_returns(sitting):
    write_skill(sitting.cfg / "skills", "fresh")
    assert sitting.session.find("notes") is sitting.record()
    assert sitting.session.find("fresh") is sitting.found.skills[-1]
    assert sitting.session.find("../x") is None
    assert sitting.session.find("absent") is None


async def test_events_from_a_child_reach_none_of_the_session_handlers(sitting):
    sitting.session.invoke(sitting.record(), "")
    sitting.session.arm()
    await sitting.completed("explore")
    assert sitting.said == []
    await sitting.call("notes", agent="explore")
    await sitting.completed()
    assert sitting.said == ['the model did not load /notes; try again, or say "use the notes skill"']


async def test_a_skill_call_for_another_name_does_not_satisfy_the_arm(tmp_path, pkg):
    sitting = Sitting(tmp_path / "cfg", ("notes", "notes-two"))
    sitting.session.invoke(sitting.record(), "")
    sitting.session.arm()
    await sitting.call("notes-two")
    await sitting.completed()
    assert sitting.said == ['the model did not load /notes; try again, or say "use the notes skill"']


async def test_a_cancelled_turn_leaves_nothing_armed_for_the_next(sitting):
    sitting.session.invoke(sitting.record(), "")
    sitting.session.arm()
    sitting.session.arm()
    await sitting.completed()
    assert sitting.said == []


async def test_a_mid_turn_arm_by_name_ignores_calls_recorded_before_it(sitting):
    await sitting.call("notes")
    sitting.session.arm()
    sitting.session.invoke(sitting.record(), "")
    sitting.session.arm("notes")
    await sitting.completed()
    assert sitting.said == ['the model did not load /notes; try again, or say "use the notes skill"']


async def test_nothing_armed_says_nothing_on_either_turn_end(sitting):
    sitting.session.arm()
    await sitting.completed()
    await sitting.failed()
    assert sitting.said == []


async def test_hostile_action_params_record_nothing_and_raise_nothing(sitting):
    path = sitting.paths["notes"]

    def bus_action(**kw):
        return sitting.bus.publish(Action(agent_id="orchestrator", session_id="s1", **kw))

    await bus_action(action_type="tool_call", tool_name="skill", tool_params=None)
    await bus_action(action_type="tool_call", tool_name="skill", tool_params={"name": 3})
    await bus_action(action_type="llm_response", tool_name="skill", tool_params={"name": "notes"})
    await sitting.act("write", path="a\x00b", content="x")
    await sitting.act("write", path=3, content="x")
    await sitting.act("write", path=str(path), content=None)
    assert sitting.session._called == set() and sitting.session._written == set()
    await sitting.act("write", path=str(path), content="\ud800")
    assert len(sitting.session._written) == 1


def test_only_the_turn_end_handler_awaits_and_only_the_say():
    def awaits(fn):
        return [n for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))) if isinstance(n, ast.Await)]

    for name in ("invoke", "arm", "find", "on_action", "on_turn_failed"):
        assert awaits(getattr(SkillSession, name)) == []
    (only,) = awaits(SkillSession.on_turn_completed)
    assert ast.unparse(only.value.func) == "self._say"


NOT_LOADED_NOTES = ('the model did not load /notes; try again, or say "use the notes skill"', {"style": "system.info"})
NOTES_LINE = "The person invoked /notes. Call skill('notes') and follow it."


class Terminal:
    def __init__(self, tmp_path, repl, channel, agent, names=("notes",)):
        self.channel, self.repl, self.agent = channel, repl, agent
        self.repl._bus = EventBus()
        self.cfg = tmp_path / "cfg"
        self.paths = {n: write_skill(self.cfg / "skills", n) for n in names}
        self.found = SkillSet.discover(self.cfg, None)
        self.session = self.repl._skills = SkillSession(self.found)
        self.bus = self.repl._bus
        set_plugin_rows([skill_row(s) for s in self.found.skills])
        self.repl._subscribe_pending()

    async def call(self, name, agent="orchestrator"):
        await self.bus.publish(Action(
            agent_id=agent, session_id="s1", action_type="tool_call", tool_name="skill", tool_params={"name": name},
        ))

    async def completed(self):
        await self.bus.publish(TurnCompleted(
            agent_id="orchestrator", session_id="s1", iterations=1, duration_seconds=0.1, elapsed_tokens=0, summary="",
        ))

    def said(self):
        return [m for m in self.channel.sent if m[0].startswith("the model did not load")]


@pytest.fixture
def terminal(tmp_path, pkg, rows):
    channel = RecordingChannel([])
    repl, agent, _ = _build_repl(channel)
    yield Terminal(tmp_path, repl, channel, agent)


@pytest.fixture
async def box(tmp_path, pkg, rows):
    repl, channel, agent = _box_repl()
    yield Terminal(tmp_path, repl, channel, agent)
    if repl._turn_task is not None:
        repl._turn_task.cancel()


async def test_slash_name_at_an_idle_session_is_one_line_the_model_must_load(terminal):
    task = await terminal.repl._dispatch_input("/notes")
    await task
    terminal.agent.run_turn.assert_called_once()
    assert terminal.agent.run_turn.call_args.kwargs["task"] == NOTES_LINE
    assert terminal.repl._slash_followup is None
    await terminal.completed()
    await terminal.completed()
    assert terminal.channel.sent[-1] == NOT_LOADED_NOTES
    assert terminal.channel.sent.count(NOT_LOADED_NOTES) == 1


async def test_a_skill_call_keeps_the_classic_path_silent(terminal):
    await (await terminal.repl._dispatch_input("/notes"))
    await terminal.call("notes")
    await terminal.completed()
    assert terminal.said() == []


async def test_unsubscribing_removes_the_session_handlers(terminal):
    terminal.repl._unsubscribe_pending()
    await (await terminal.repl._dispatch_input("/notes"))
    await terminal.completed()
    assert terminal.said() == []


async def test_a_repl_without_a_session_refuses_an_unknown_word_as_before(rows):
    channel = RecordingChannel([])
    repl, agent, _ = _build_repl(channel)
    assert await repl._handle_slash("/frobnicate") is True
    assert channel.sent == [("Unknown command: /frobnicate — /help lists commands.", {"style": "system.error"})]
    assert await repl._handle_slash("/frobnicate more") is False
    agent.run_turn.assert_not_called()


async def test_the_box_starts_the_turn_on_the_verbatim_line_and_says_a_miss_once(box):
    await box.repl._handle_box_event("submit", "/notes")
    box.agent.run_turn.assert_called_once()
    assert box.agent.run_turn.call_args.kwargs["task"] == NOTES_LINE
    await box.completed()
    await box.completed()
    assert box.channel.sent.count(NOT_LOADED_NOTES) == 1


async def test_a_slash_name_during_a_turn_nudges_it_once_after_arming_the_name(terminal):
    order = []
    arm = terminal.session.arm
    terminal.session.arm = lambda *a: (order.append("arm"), arm(*a))[1]
    terminal.agent.push_user_nudge = MagicMock(side_effect=lambda line: order.append("nudge"))
    await _pending_turn(terminal.repl)
    try:
        assert await terminal.repl._handle_slash("/notes now") is True
        line = "The person invoked /notes with: now. Call skill('notes') and follow it."
        terminal.agent.push_user_nudge.assert_called_once_with(line)
        assert order == ["arm", "nudge"]
        assert terminal.repl._slash_followup is None
        terminal.agent.run_turn.assert_not_called()
        await terminal.completed()
        assert terminal.channel.sent.count(NOT_LOADED_NOTES) == 1
        terminal.session.arm("notes")
        await terminal.call("notes")
        terminal.channel.sent.clear()
        await terminal.completed()
        assert terminal.said() == []
    finally:
        terminal.repl._turn_task.cancel()


async def test_a_cancelled_slash_turn_leaves_the_next_ordinary_turn_unarmed(terminal):
    task = await terminal.repl._dispatch_input("/notes")
    task.cancel()
    await (await terminal.repl._dispatch_input("hello"))
    await terminal.completed()
    assert terminal.said() == []


async def test_a_changed_body_is_noticed_before_the_turn_starts(terminal):
    await (await terminal.repl._dispatch_input("/notes"))
    path = terminal.paths["notes"]
    path.write_text(path.read_text().replace("Body of notes.", "Body of notes, edited."))
    terminal.channel.sent.clear()
    await (await terminal.repl._dispatch_input("/notes"))
    assert terminal.channel.sent == [(f"changed since first use: {str(path)!r}; using it", {"style": "system.info"})]
    assert terminal.agent.run_turn.call_count == 2
    assert "Body of notes" not in terminal.agent.run_turn.call_args.kwargs["task"]


async def test_an_unreadable_body_prints_the_skip_line_and_starts_nothing(terminal):
    terminal.paths["notes"].unlink()
    assert await terminal.repl._handle_slash("/notes") is True
    (text, meta), = terminal.channel.sent
    assert text.startswith("skipped: ") and meta == {"style": "system.error"}
    assert terminal.repl._slash_followup is None
    assert await terminal.repl._dispatch_input("/notes") is None
    terminal.agent.run_turn.assert_not_called()


def fresh_line(args="go"):
    return f"The person invoked /fresh with: {args}. Call skill('fresh') and follow it."


async def test_a_skill_written_after_start_gets_its_notice_and_its_row_at_its_first_name(terminal):
    path = write_skill(terminal.cfg / "skills", "fresh")
    assert await terminal.repl._handle_slash("/fresh go") is True
    assert terminal.channel.sent == [(f"added since start: fresh (machine, {str(path)!r})", {"style": "system.info"})]
    fresh = next(s for s in terminal.session.skill_set.skills if s.name == "fresh")
    assert all_rows()[-1] == skill_row(fresh)
    assert terminal.repl._slash_followup == fresh_line()
    terminal.channel.sent.clear()
    await terminal.repl._dispatch_input("/fresh again")
    assert terminal.channel.sent == []
    assert [r.name for r in all_rows()].count("/fresh") == 1


async def test_a_skill_that_cannot_load_after_start_prints_its_skip_line_and_adds_no_row(terminal):
    write_skill(terminal.cfg / "skills", "broken", "no frontmatter")
    before = all_rows()
    assert await terminal.repl._handle_slash("/broken") is True
    (text, meta), = terminal.channel.sent
    assert text.startswith("skipped: ") and meta == {"style": "system.error"}
    assert all_rows() == before
    assert terminal.repl._slash_followup is None


async def test_only_a_bare_unknown_word_is_refused(terminal):
    assert await terminal.repl._handle_slash("/nothing") is True
    assert terminal.channel.sent == [("Unknown command: /nothing — /help lists commands.", {"style": "system.error"})]
    assert await terminal.repl._handle_slash("/nothing more words") is False


@pytest.mark.parametrize("word, handled", [
    ("/tmp/x", False), ("/../x", False), ("/a/b", False), ("/Bad_Name", True), ("/", False), ("/ x", False), ("/nötes", False),
])
async def test_a_hostile_word_never_reaches_the_disk(terminal, monkeypatch, word, handled):
    def boom(*_):
        raise AssertionError("a path was built")

    monkeypatch.setattr(skills, "_hits", boom)
    assert await terminal.repl._handle_slash(word) is handled


async def test_the_word_is_lower_cased_on_the_live_path_too(terminal):
    write_skill(terminal.cfg / "skills", "fresh")
    assert await terminal.repl._handle_slash("/FRESH go") is True
    assert terminal.repl._slash_followup == fresh_line()
    terminal.repl._slash_followup = None
    assert await terminal.repl._handle_slash("/NOTES") is True
    assert terminal.repl._slash_followup == NOTES_LINE


async def test_a_word_longer_than_a_row_name_is_looked_up_and_a_row_backed_word_is_not(terminal, monkeypatch):
    asked = []
    find = terminal.session.find
    monkeypatch.setattr(terminal.session, "find", lambda name: (asked.append(name), find(name))[1])
    assert await terminal.repl._handle_slash("/notes x") is True
    assert asked == ["notes"]
    asked.clear()
    assert await terminal.repl._handle_slash("/notesx") is True
    assert asked == ["notesx"]
    asked.clear()
    write_skill(terminal.cfg / "skills", "help")
    assert await terminal.repl._handle_slash("/help x") is False
    assert asked == []
    terminal.agent.run_turn.assert_not_called()


async def test_the_listing_and_the_enum_stay_as_start_fixed_them(terminal, monkeypatch, tmp_path):
    from localharness.agent.context import ContentStore
    from localharness.agent.task_record import TaskState
    from localharness.tools.builtin import skill_tool

    tool = skill_tool.SkillTool(
        terminal.found, ContentStore(), TaskState(tmp_path / "task.json", workspace=str(tmp_path)), lambda: 10_000,
    )
    assert tool.info().parameters["properties"]["name"]["enum"] == ["notes"]
    write_skill(terminal.cfg / "skills", "fresh")
    with monkeypatch.context() as m:
        m.setattr(skill_tool, "render_listing", lambda *_: pytest.fail("the listing was rebuilt"))
        m.setattr(skill_tool, "skill_schema", lambda *_: pytest.fail("the schema was rebuilt"))
        assert await terminal.repl._handle_slash("/fresh go") is True
    assert tool.info().parameters["properties"]["name"]["enum"] == ["notes"]


def _watch_run(monkeypatch, *lines):
    seen = {"followups": []}

    async def run(self):
        seen.update(rows=all_rows(), session=self._skills, tool=self._agent._tools._tools["global"].get("skill"))
        for line in lines:
            await self._handle_slash(line)
            seen["followups"].append(self._slash_followup)
            self._slash_followup = None

    monkeypatch.setattr(OrchestratorREPL, "run", run)
    return seen


async def test_a_skill_at_start_is_a_row_that_asks_the_model_to_load_it(tmp_path, pkg, rows, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_skills_listing import _start_world

    seen_set, _ = _start_world(tmp_path, monkeypatch)
    seen = _watch_run(monkeypatch, "/notes draft")
    await _start_async(None, False, False, str(tmp_path))
    (notes,) = seen_set[0]
    assert seen["rows"][-1] == skill_row(notes)
    assert seen["followups"] == ["The person invoked /notes with: draft. Call skill('notes') and follow it."]
    assert seen["session"].skill_set is seen["tool"]._skills
    assert all_rows() == SLASH_COMMANDS


async def test_a_start_with_no_skill_has_no_skill_row_and_no_session(tmp_path, pkg, rows, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_skills_listing import _start_world

    _start_world(tmp_path, monkeypatch, skill=False)
    seen = _watch_run(monkeypatch)
    await _start_async(None, False, False, str(tmp_path))
    assert seen["session"] is None and not [r for r in seen["rows"] if r.plugin == "skill"]
