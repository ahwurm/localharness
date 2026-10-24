"""Slash rows for skills: the dispatch seam and late rows."""
from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import AsyncMock

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.cli.skill_session import SkillSession
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
