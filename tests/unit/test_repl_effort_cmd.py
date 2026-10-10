"""/effort in the REPL (spec 16 §7.1/§7.2, 03-CONTEXT D-01..D-05, D-10, D-14, D-29, D-30, D-33).

Drives the real OrchestratorREPL with a scripted channel and a MagicMock loop, faking only the
process boundaries: the persistence call is replaced by a recorder, the picker is a recording double.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import LEVELS
from localharness.provider import speed_stats
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


# ============================================================================ /model re-resolution

class FakeLoop:
    """The slice of an AgentLoop `_refresh_token_counter` and `/effort` read, with `set_effort` recorded.
    No context manager and no counter, so the refresh touches nothing but the effort and the ledger."""

    def __init__(self, llm, file_rung="inherit"):
        self._llm = llm
        self._ctx = None
        self._config = SimpleNamespace(
            effort=file_rung, context=None, max_tokens=None, name="orchestrator")
        self.set_calls: list[tuple[str, str]] = []
        self.effective_effort = None
        self.effort_rung = None

    def set_effort(self, level, rung):
        self.set_calls.append((level, rung))
        self.effective_effort, self.effort_rung = level, rung


def _loop_repl(tmp_path, *, harness, llm=None, file_rung="inherit"):
    loop = FakeLoop(llm or RecordingLLM(), file_rung)
    channel = Chan([])
    repl = OrchestratorREPL(
        orchestrator=MagicMock(), agent_loop=loop, channel=channel, bus=AsyncMock(),
        config_dir=tmp_path, harness_config=harness,
    )
    return repl, loop, channel


def test_a_model_switch_to_a_peer_with_its_own_effort_moves_the_loop_and_not_the_file(tmp_path):
    harness = harness_with(server_effort="high", peer=PEER_URL, peer_effort="max")
    repl, loop, _ = _loop_repl(tmp_path, harness=harness)
    run(repl._refresh_token_counter("m", base_url=PEER_URL, provider_type="llamacpp"))
    assert loop.set_calls == [("max", "endpoint")]
    run(repl._refresh_token_counter("m", base_url=PRIMARY_URL))
    assert loop.set_calls == [("max", "endpoint"), ("high", "endpoint")]
    assert loop._config.effort == "inherit"


def test_a_session_level_survives_both_switches(tmp_path):
    harness = harness_with(server_effort="high", peer=PEER_URL, peer_effort="max")
    repl, loop, _ = _loop_repl(tmp_path, harness=harness)
    repl._session_effort = "low"
    run(repl._refresh_token_counter("m", base_url=PEER_URL, provider_type="llamacpp"))
    run(repl._refresh_token_counter("m", base_url=PRIMARY_URL))
    assert loop.set_calls == [("low", "session"), ("low", "session")]


def test_an_agent_file_level_outranks_the_endpoint_after_a_switch(tmp_path):
    harness = harness_with(server_effort="high", peer=PEER_URL, peer_effort="max")
    repl, loop, _ = _loop_repl(tmp_path, harness=harness, file_rung="low")
    run(repl._refresh_token_counter("m", base_url=PEER_URL, provider_type="llamacpp"))
    assert loop.set_calls == [("low", "agent")] and loop._config.effort == "low"


def test_no_harness_means_no_re_resolve_and_no_error(tmp_path):
    repl, loop, _ = _loop_repl(tmp_path, harness=None)
    assert run(repl._refresh_token_counter("m", base_url=PEER_URL, provider_type="llamacpp")) == ""
    assert loop.set_calls == []


# ============================================================================ the rejected-key ledger

KEY = ("llamacpp", "m")
RBT = "reasoning_budget_tokens"
CTK = "chat_template_kwargs"


def _ledger(tmp_path):
    return speed_stats.default_speed_stats_path(tmp_path)


def _seed(tmp_path, keys):
    speed_stats.record_rejected_effort_keys(_ledger(tmp_path), *KEY, keys)


def _recorded(tmp_path):
    return speed_stats.last_rejected_effort_keys(_ledger(tmp_path), *KEY)


@pytest.fixture
def writes(monkeypatch):
    """Counts every record_rejected_effort_keys call (and passes it through)."""
    calls: list[list[str]] = []
    real = speed_stats.record_rejected_effort_keys

    def counting(path, provider_type, model, keys):
        calls.append(sorted(keys))
        return real(path, provider_type, model, keys)

    monkeypatch.setattr(speed_stats, "record_rejected_effort_keys", counting)
    return calls


def _session(tmp_path, turns, *, llm=None):
    """Drive repl.run() through `len(turns)` ordinary user turns. `turns` is a list of
    (rejected, accepted) pairs the client holds AFTER that turn."""
    llm = llm or RecordingLLM()
    script = list(turns)

    async def run_turn(**kw):
        rejected, accepted = script.pop(0)
        llm.rejected_effort_keys = frozenset(rejected)
        llm.accepted_effort_keys = frozenset(accepted)

    channel = Chan(["hello"] * len(turns))
    repl, agent, _ = _build_repl(channel, run_turn=run_turn)
    agent._llm = llm
    repl._config_dir = tmp_path
    asyncio.run(repl.run())
    return channel, repl


def _said(channel, key=RBT):
    return [t for t, _ in channel.sent if t == EFFORT_UNENFORCED_TEMPLATE.format(key=key)]


def _unenforced(channel):
    return [t for t, _ in channel.sent if t.startswith("effort unenforced")]


def test_a_first_turn_with_an_empty_live_set_leaves_a_pre_seeded_ledger_alone(tmp_path, writes):
    _seed(tmp_path, [RBT])
    del writes[:]
    channel, _ = _session(tmp_path, [((), ())])
    assert _recorded(tmp_path) == [RBT]
    assert _unenforced(channel) == [] and writes == []


def test_ten_max_or_off_turns_write_nothing_and_say_nothing(tmp_path, writes):
    _seed(tmp_path, [RBT])
    del writes[:]
    channel, _ = _session(tmp_path, [((), ())] * 10)
    assert _recorded(tmp_path) == [RBT]
    assert _unenforced(channel) == [] and writes == []


def test_a_new_rejection_is_said_once_and_recorded(tmp_path, writes):
    channel, _ = _session(tmp_path, [({RBT}, ()), ({RBT}, ())])
    assert _said(channel) == [EFFORT_UNENFORCED_TEMPLATE.format(key=RBT)]
    assert EFFORT_UNENFORCED_TEMPLATE.format(key=RBT) == (
        "effort unenforced on this server: reasoning_budget_tokens rejected")
    assert _recorded(tmp_path) == [RBT]
    assert writes == [[RBT]]
    assert ([m for t, m in channel.sent if t.startswith("effort unenforced")]
            == [INFO])


def test_the_same_rejected_key_over_ten_turns_prints_once_and_writes_once(tmp_path, writes):
    channel, _ = _session(tmp_path, [({RBT}, ())] * 10)
    assert len(_said(channel)) == 1 and writes == [[RBT]]


def test_a_new_key_joins_the_recorded_ones_and_only_it_is_said(tmp_path, writes):
    _seed(tmp_path, [CTK])
    del writes[:]
    channel, _ = _session(tmp_path, [({RBT}, ())])
    assert _unenforced(channel) == [EFFORT_UNENFORCED_TEMPLATE.format(key=RBT)]
    assert _recorded(tmp_path) == [CTK, RBT]


def test_a_key_the_ledger_already_holds_is_not_said_again_after_a_restart(tmp_path, writes):
    _seed(tmp_path, [RBT])
    del writes[:]
    channel, _ = _session(tmp_path, [({RBT}, ())])
    assert _unenforced(channel) == [] and writes == []
    assert _recorded(tmp_path) == [RBT]


def test_a_key_the_server_accepted_leaves_the_ledger_and_is_said_again_if_refused_later(
        tmp_path, writes):
    _seed(tmp_path, [CTK, RBT])
    del writes[:]
    channel, _ = _session(tmp_path, [((), {RBT})])
    assert _recorded(tmp_path) == [CTK] and _unenforced(channel) == []
    channel, _ = _session(tmp_path, [({RBT}, ())])
    assert _said(channel) == [EFFORT_UNENFORCED_TEMPLATE.format(key=RBT)]
    assert _recorded(tmp_path) == [CTK, RBT]


def test_an_accepted_key_that_was_never_recorded_changes_nothing(tmp_path, writes):
    channel, _ = _session(tmp_path, [((), {RBT})] * 3)
    assert writes == [] and _recorded(tmp_path) is None and _unenforced(channel) == []


def test_a_ledger_write_error_is_swallowed_and_the_key_is_not_said_twice(tmp_path, monkeypatch):
    attempts: list[int] = []

    def refuse(*a, **k):
        attempts.append(1)
        raise OSError("read-only")

    monkeypatch.setattr(speed_stats, "record_rejected_effort_keys", refuse)
    channel, _ = _session(tmp_path, [({RBT}, ())] * 3)
    assert len(_said(channel)) == 1
    assert attempts  # the write was tried; the turns still completed


def test_a_client_without_a_key_is_skipped_quietly(tmp_path, writes):
    llm = RecordingLLM()
    del llm.config
    channel, _ = _session(tmp_path, [({RBT}, ())], llm=llm)
    assert writes == [] and _unenforced(channel) == []


# ---------------------------------------------------------------------------- rebind

def test_a_switch_to_a_different_endpoint_clears_the_targets_entry_and_the_announced_set(
        tmp_path, writes):
    _seed(tmp_path, [RBT])
    del writes[:]
    repl, loop, _ = _loop_repl(tmp_path, harness=harness_with(server_effort="high", peer=PEER_URL))
    repl._announced_effort_keys.add(RBT)
    loop._llm.config.base_url = PEER_URL  # the client was rebound before the refresh runs
    run(repl._refresh_token_counter("m", base_url=PEER_URL, provider_type="llamacpp"))
    assert _recorded(tmp_path) == [] and repl._announced_effort_keys == set()
    assert writes == [[]]
    assert repl._effort_endpoint == PEER_URL


def test_a_same_endpoint_refresh_leaves_the_entry_and_the_announced_set_alone(tmp_path, writes):
    _seed(tmp_path, [RBT])
    del writes[:]
    repl, loop, _ = _loop_repl(tmp_path, harness=harness_with(server_effort="high"))
    repl._announced_effort_keys.add(RBT)
    run(repl._refresh_token_counter("m", base_url=PRIMARY_URL + "/", provider_type="llamacpp"))
    assert _recorded(tmp_path) == [RBT] and repl._announced_effort_keys == {RBT}
    assert writes == []


def test_a_failed_clear_never_breaks_the_swap(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(speed_stats, "record_rejected_effort_keys", refuse)
    repl, loop, _ = _loop_repl(tmp_path, harness=harness_with(server_effort="high", peer=PEER_URL))
    loop._llm.config.base_url = PEER_URL
    assert run(repl._refresh_token_counter("m", base_url=PEER_URL, provider_type="llamacpp")) == ""
    assert repl._effort_endpoint == PEER_URL


def test_the_announce_after_a_rebind_says_a_refusal_by_the_new_server(tmp_path, writes):
    """A rejection belongs to the server that made it: after the clear the same key is news again."""
    _seed(tmp_path, [RBT])
    repl, loop, channel = _loop_repl(
        tmp_path, harness=harness_with(server_effort="high", peer=PEER_URL))
    loop._llm.rejected_effort_keys = frozenset({RBT})
    run(repl._announce_rejected_effort_keys())
    assert _unenforced(channel) == []  # the old server's ledger already has it
    loop._llm.config.base_url = PEER_URL
    run(repl._refresh_token_counter("m", base_url=PEER_URL, provider_type="llamacpp"))
    assert _recorded(tmp_path) == []
    run(repl._announce_rejected_effort_keys())  # the new server refuses it too
    assert _said(channel) == [EFFORT_UNENFORCED_TEMPLATE.format(key=RBT)]
    assert _recorded(tmp_path) == [RBT]
