"""/effort in the REPL (spec 16 §7.1/§7.2, 03-CONTEXT D-01..D-05, D-10, D-14, D-29, D-30, D-33).

Drives the real OrchestratorREPL with a scripted channel and a MagicMock loop, faking only the
process boundaries: the persistence call is replaced by a recorder, the picker is a recording double.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from localharness.cli import model_ops
from localharness.cli import repl as repl_mod
from localharness.cli.repl import (
    EFFORT_LIST_TEMPLATE,
    EFFORT_NONE_LINE,
    EFFORT_PERSIST_FAILED_TEMPLATE,
    EFFORT_PICKER_PROMPT,
    EFFORT_SET_TEMPLATE,
    EFFORT_UNENFORCED_TEMPLATE,
    EFFORT_UNKNOWN_TEMPLATE,
)
from localharness.config.models import LEVELS
from tests.unit.effort_support import PRIMARY_URL, RecordingLLM, harness_with
from tests.unit.test_repl_unknown_slash import RecordingChannel, _build_repl

INFO = {"style": "system.info"}
ERROR = {"style": "system.error"}
PEER_URL = "http://127.0.0.1:8082/v1"


class Chan(RecordingChannel):
    channel_id = "recording"


class TerminalDouble(Chan):
    """A terminal with a running input box: it can draw the picker."""

    def __init__(self, answer):
        super().__init__([])
        self.answer = answer
        self.picker_calls: list[tuple] = []

    async def run_picker(self, options, current):
        self.picker_calls.append((options, current))
        return self.answer


class MobileDouble(Chan):
    channel_id = "mobile"

    def __init__(self, answer, attached=True):
        super().__init__([])
        self.answer = answer
        self.attached = attached
        self.picker_calls: list[tuple] = []

    def can_run_picker(self):
        return self.attached

    async def run_picker(self, options, current, *, prompt=None):
        self.picker_calls.append((options, current, prompt))
        return self.answer


class Persist:
    """Replaces model_ops.persist_effort: records the keywords; returns `result` or raises it."""

    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple] = []

    async def __call__(self, harness, level, **kw):
        self.calls.append((level, kw))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _repl(channel=None, *, server_effort="max", file_rung="inherit", llm=None, **harness_kw):
    channel = channel if channel is not None else Chan([])
    repl, agent, _ = _build_repl(channel)
    agent._config.effort = file_rung
    agent._llm = llm or RecordingLLM()
    agent.effective_effort = "high"
    agent.effort_rung = "endpoint"
    repl._harness = harness_with(server_effort=server_effort, **harness_kw)
    return repl, agent, channel


@pytest.fixture
def persist(monkeypatch):
    rec = Persist()
    monkeypatch.setattr(model_ops, "persist_effort", rec)
    return rec


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------------- the tracer

def test_effort_low_sets_the_loop_prints_and_persists_with_the_channel_id(persist):
    repl, agent, channel = _repl()
    assert run(repl._handle_slash("/effort low")) is True
    agent.set_effort.assert_called_once_with("low", "session")
    assert repl._session_effort == "low"
    assert channel.sent[0] == (EFFORT_SET_TEMPLATE.format(level="low", rung="session"), INFO)
    assert channel.sent[0][0] == "effort: low (session)"
    ((level, kw),) = persist.calls
    assert level == "low"
    assert kw["base_url"] == PRIMARY_URL and kw["channel_id"] == "recording"
    agent.run_turn.assert_not_called()


def test_the_argument_is_lowered_and_stripped_by_the_dispatcher(persist):
    repl, agent, _ = _repl()
    run(repl._handle_slash("  /Effort  MAX "))
    agent.set_effort.assert_called_once_with("max", "session")


def test_a_persist_failure_is_reported_and_never_undoes_the_session_change(persist):
    persist.result = RuntimeError("boom")
    repl, agent, channel = _repl()
    run(repl._handle_effort_cmd("low"))
    agent.set_effort.assert_called_once_with("low", "session")
    assert (EFFORT_PERSIST_FAILED_TEMPLATE.format(exc="boom"), ERROR) in channel.sent
    assert "Effort set for this session, but persisting it failed: boom" == channel.sent[-1][0]
    assert repl._session_effort == "low"


def test_a_persist_note_follows_the_set_line_as_info(persist):
    persist.result = model_ops.EFFORT_SESSION_ONLY_NOTE
    repl, _, channel = _repl()
    run(repl._handle_effort_cmd("high"))
    assert channel.sent == [
        ("effort: high (session)", INFO), (model_ops.EFFORT_SESSION_ONLY_NOTE, INFO)]


def test_an_unknown_argument_prints_the_unknown_line_and_nothing_else(persist):
    repl, agent, channel = _repl()
    run(repl._handle_effort_cmd("medium"))
    assert channel.sent == [(EFFORT_UNKNOWN_TEMPLATE.format(arg="medium"), ERROR)]
    assert channel.sent[0][0] == (
        "Unknown effort 'medium'. Levels: off, low, high, max — /effort show reports the current level.")
    agent.set_effort.assert_not_called()
    assert persist.calls == [] and repl._session_effort is None


def test_the_agent_file_rung_is_never_written(persist):
    repl, agent, _ = _repl()
    for arg in ("low", "max", "show", "", "nope"):
        run(repl._handle_effort_cmd(arg))
    assert agent._config.effort == "inherit"


# ----------------------------------------------------------------------------- bare /effort

def test_bare_effort_on_a_scripted_channel_prints_the_list_with_the_current_bracketed(persist):
    repl, agent, channel = _repl()
    run(repl._handle_effort_cmd(""))
    line = EFFORT_LIST_TEMPLATE.format(levels="off  low  [high]  max")
    assert channel.sent == [(line, INFO)]
    assert line == "effort: off  low  [high]  max — /effort <level> sets it, /effort show names the rung"
    agent.set_effort.assert_not_called()
    assert persist.calls == []


def test_bare_effort_with_no_level_brackets_nothing(persist):
    repl, agent, channel = _repl()
    agent.effective_effort = None
    run(repl._handle_effort_cmd(""))
    assert channel.sent == [(EFFORT_LIST_TEMPLATE.format(levels="off  low  high  max"), INFO)]


def test_the_terminal_picker_gets_the_levels_and_the_current_and_its_answer_is_applied(
        persist, monkeypatch):
    channel = TerminalDouble("max")
    repl, agent, _ = _repl(channel)
    monkeypatch.setattr(repl, "_use_input_box", lambda: True)
    run(repl._handle_effort_cmd(""))
    assert channel.picker_calls == [(list(LEVELS), "high")]
    agent.set_effort.assert_called_once_with("max", "session")
    assert channel.sent == [("effort: max (session)", INFO)]
    assert persist.calls[0][0] == "max"


def test_the_terminal_pickers_none_changes_and_prints_nothing(persist, monkeypatch):
    channel = TerminalDouble(None)
    repl, agent, _ = _repl(channel)
    monkeypatch.setattr(repl, "_use_input_box", lambda: True)
    run(repl._handle_effort_cmd(""))
    assert channel.picker_calls == [(list(LEVELS), "high")]
    agent.set_effort.assert_not_called()
    assert channel.sent == [] and persist.calls == [] and repl._session_effort is None


def test_a_terminal_without_a_running_box_prints_the_list_not_the_picker(persist):
    channel = TerminalDouble("max")
    repl, agent, _ = _repl(channel)  # _use_input_box() is False for a double
    run(repl._handle_effort_cmd(""))
    assert channel.picker_calls == []
    assert channel.sent[0][0].startswith("effort: off  low  [high]  max")
    agent.set_effort.assert_not_called()


def test_the_phone_picker_carries_the_prompt_and_its_answer_is_applied_like_a_typed_level(persist):
    channel = MobileDouble("low")
    repl, agent, _ = _repl(channel)
    run(repl._handle_effort_cmd(""))
    assert channel.picker_calls == [(list(LEVELS), "high", EFFORT_PICKER_PROMPT)]
    agent.set_effort.assert_called_once_with("low", "session")
    assert channel.sent == [("effort: low (session)", INFO)]
    assert persist.calls[0][0] == "low" and persist.calls[0][1]["channel_id"] == "mobile"


def test_the_phone_pickers_none_changes_and_prints_nothing(persist):
    channel = MobileDouble(None)
    repl, agent, _ = _repl(channel)
    run(repl._handle_effort_cmd(""))
    assert len(channel.picker_calls) == 1
    agent.set_effort.assert_not_called()
    assert channel.sent == [] and persist.calls == []


def test_a_phone_with_no_client_attached_gets_the_list(persist):
    channel = MobileDouble("low", attached=False)
    repl, agent, _ = _repl(channel)
    run(repl._handle_effort_cmd(""))
    assert channel.picker_calls == []
    assert channel.sent[0][0].startswith("effort: off  low  [high]  max")
    agent.set_effort.assert_not_called()


# ----------------------------------------------------------------------------- /effort show

def test_show_names_level_rung_agents_and_rejected_keys(persist, monkeypatch, tmp_path):
    monkeypatch.setattr(model_ops, "agents_with_effort",
                        lambda cfg, local_config_dir=None: [("critic", "low"), ("writer", "max")])
    llm = RecordingLLM(rejected=("reasoning_budget_tokens", "chat_template_kwargs"))
    repl, agent, channel = _repl(llm=llm)
    repl._config_dir = tmp_path
    run(repl._handle_effort_cmd("show"))
    assert channel.sent == [(
        "effort: high (endpoint)\n"
        "  critic: low\n"
        "  writer: max\n"
        "effort unenforced on this server: chat_template_kwargs rejected\n"
        "effort unenforced on this server: reasoning_budget_tokens rejected", INFO)]
    assert EFFORT_UNENFORCED_TEMPLATE.format(key="chat_template_kwargs") in channel.sent[0][0]
    agent.set_effort.assert_not_called()


def test_show_on_a_loop_with_no_level_prints_the_none_line(persist, monkeypatch):
    monkeypatch.setattr(model_ops, "agents_with_effort", lambda cfg, local_config_dir=None: [])
    repl, agent, channel = _repl()
    agent.effective_effort = None
    agent.effort_rung = None
    run(repl._handle_effort_cmd("show"))
    assert channel.sent == [(EFFORT_NONE_LINE, INFO)]
    assert EFFORT_NONE_LINE == "effort: none (no effort keys are sent)"


def test_show_reads_the_real_agent_files(persist, tmp_path):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "critic.yaml").write_text(
        "name: critic\ndescription: d\nsystem_prompt: p\neffort: low\n")
    (tmp_path / "agents" / "plain.yaml").write_text(
        "name: plain\ndescription: d\nsystem_prompt: p\neffort: inherit\n")
    repl, _, channel = _repl()
    repl._config_dir = tmp_path
    run(repl._handle_effort_cmd("show"))
    assert channel.sent[0][0] == "effort: high (endpoint)\n  critic: low"


# ----------------------------------------------------------------------------- the table

def test_effort_is_a_core_row_after_reasoning_and_never_terminal_only():
    from localharness.cli.slash_commands import SLASH_COMMANDS, find_row

    names = [row.name for row in SLASH_COMMANDS]
    assert names.index("/effort") == names.index("/reasoning") + 1
    row = find_row("/effort low")
    assert row is not None and row.takes_args and not row.terminal_only
    assert row.handler == "_slash_effort"


def test_reasoning_is_untouched(persist):
    channel = Chan([])
    repl, _, _ = _build_repl(channel)
    repl._handle_reasoning_cmd = MagicMock(side_effect=lambda arg: asyncio.sleep(0))
    run(repl._handle_slash("/reasoning ON"))
    repl._handle_reasoning_cmd.assert_called_once_with("on")


def test_the_module_still_exports_the_effort_templates():
    for name in ("EFFORT_SET_TEMPLATE", "EFFORT_NONE_LINE", "EFFORT_LIST_TEMPLATE",
                 "EFFORT_UNKNOWN_TEMPLATE", "EFFORT_AGENT_LINE", "EFFORT_UNENFORCED_TEMPLATE",
                 "EFFORT_PERSIST_FAILED_TEMPLATE", "EFFORT_PICKER_PROMPT"):
        assert isinstance(getattr(repl_mod, name), str), name
