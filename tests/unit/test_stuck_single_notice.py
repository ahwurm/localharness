"""A stuck turn is announced once (STUCK-04, D-04, D-40).

The reply is the partial work and carries no stuck text; TurnFailed.detail is that reply; the
default ChannelAdapter.on_turn_failed adds no stuck line for stuck_detected, so Escalation,
delivered by on_escalation, is the one notice. It does deliver a ROOT turn's detail as a plain
message (CR-51): a stuck turn publishes no TaskComplete, so that is the only way the wrap-up the
loop spent a model call on reaches the terminal, Discord or (when nothing streamed) Zed. The bus
still carries both events.
"""
from __future__ import annotations

import pytest

from localharness.channels.base import ChannelAdapter
from localharness.core.events import Escalation, TurnFailed


class _Recorder(ChannelAdapter):
    def __init__(self, bus):
        super().__init__(bus=bus, config=object())
        self.sent: list[str] = []
        self.errors: list[tuple[str, str | None]] = []

    async def read_input(self): return None
    async def send_message(self, content, agent_id=None, **kw): self.sent.append(content)
    async def send_error(self, error, detail=None, agent_id=None, **kw):
        self.errors.append((error, detail))
    async def send_streaming(self, *a, **kw): pass
    async def send_tool_call(self, *a, **kw): pass
    async def send_tool_result(self, *a, **kw): pass
    async def start(self): pass
    async def stop(self): pass


def _subscribe(bus):
    chan = _Recorder(bus)
    bus.subscribe(TurnFailed, chan.on_turn_failed)
    bus.subscribe(Escalation, chan.on_escalation)
    return chan


class _StuckLLM:
    class config:
        tool_call_mode = "native"
        context_window = 128_000

    def __init__(self, response_cls, tool_call, wrapup):
        self._resp, self._tc, self._wrapup = response_cls, tool_call, wrapup

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kw):
        if tools is None:
            if self._wrapup is None:
                raise RuntimeError("provider boom")
            return self._resp(content=self._wrapup), None
        return self._resp(content=None, tool_calls=[self._tc]), None


async def _stuck_turn(mock_llm_client, bus, wrapup):
    from tests.unit.test_agent_loop import _SameResultRegistry, _make_agent_loop

    loop = _make_agent_loop(mock_llm_client, [], bus, tool_registry=_SameResultRegistry())
    tc = mock_llm_client.ToolCall(id="a", name="bash", arguments={"cmd": "ls"})
    loop._llm = _StuckLLM(mock_llm_client.Response, tc, wrapup)
    return await loop.run_turn("task")


@pytest.mark.asyncio
async def test_a_stuck_turns_reply_is_the_wrapup_and_carries_no_stuck_text(mock_llm_client, bus):
    reply = await _stuck_turn(mock_llm_client, bus, "Wrote the file; parser still TODO.")
    assert reply == "Wrote the file; parser still TODO."
    for needle in ("[Agent stuck", "Escalating to orchestrator", "partial completion"):
        assert needle not in reply
    failed = bus.history(event_types=[TurnFailed])
    assert [e.detail for e in failed] == [reply]
    assert [e.reason for e in failed] == ["stuck_detected"]


@pytest.mark.asyncio
async def test_a_failed_wrapup_leaves_the_last_assistant_content_or_nothing(mock_llm_client, bus):
    reply = await _stuck_turn(mock_llm_client, bus, None)
    assert reply == ""
    assert "[Agent stuck" not in reply
    assert [e.detail for e in bus.history(event_types=[TurnFailed])] == [""]


@pytest.mark.asyncio
async def test_a_recording_channel_sees_exactly_one_error_for_a_stuck_turn(mock_llm_client, bus):
    chan = _subscribe(bus)
    await _stuck_turn(mock_llm_client, bus, "partial work")

    assert len(chan.errors) == 1
    error, _detail = chan.errors[0]
    assert "escalated: stuck_detected" in error
    assert not any("turn failed" in e for e, _ in chan.errors)
    # CR-51: the reply reaches the person as a plain message, once, with no stuck text in it.
    assert chan.sent == ["partial work"]
    # the bus still carries both events
    assert len(bus.history(event_types=[Escalation])) == 1
    assert len(bus.history(event_types=[TurnFailed])) == 1


@pytest.mark.asyncio
async def test_a_child_stuck_turn_posts_the_escalation_and_no_failed_status_line():
    chan = _Recorder(bus=object())
    await chan.on_turn_failed(TurnFailed(
        agent_id="child", session_id="s", parent_id="p", reason="stuck_detected",
        detail="partial", iterations=4, duration_seconds=1.0))
    assert chan.sent == [] and chan.errors == []
    await chan.on_escalation(Escalation(
        agent_id="child", session_id="s", parent_id="p", reason="stuck_detected",
        detail="Repeated identical tool calls at iteration 4", iteration_at_escalation=4))
    assert len(chan.errors) == 1 and chan.sent == []


@pytest.mark.asyncio
async def test_other_turn_failed_reasons_behave_as_before():
    chan = _Recorder(bus=object())
    await chan.on_turn_failed(TurnFailed(
        agent_id="a", session_id="s", reason="internal_error", detail="boom " * 200,
        iterations=4, duration_seconds=1.0))
    assert chan.errors == [("turn failed — internal_error", ("boom " * 200).strip()[:400])]
    await chan.on_turn_failed(TurnFailed(
        agent_id="kid", session_id="s", parent_id="p", reason="budget_exceeded", detail="x",
        iterations=4, duration_seconds=1.0))
    assert len(chan.sent) == 1 and "subagent kid failed (budget_exceeded)" in chan.sent[0]


# IN-73: a root budget stop is one error line naming the bound, then the wrap-up reply WHOLE. The
# budget row above used to pin `send_error(..., detail[:400])` for budget_exceeded.

_KNOB_NOTE = ("[Budget limit reached: Agent reached the maximum of 100 tool calls. Stopping to "
              "prevent runaway execution. (channels.unattended_max_actions) Completed 100 tool "
              "calls in 12.0 minutes.]")


@pytest.mark.asyncio
async def test_a_root_budget_stop_names_the_bound_once_and_sends_the_reply_whole():
    chan = _Recorder(bus=object())
    reply = "Found the three call sites. " * 40  # well past 400 characters
    await chan.on_turn_failed(TurnFailed(
        agent_id="a", session_id="s", reason="budget_exceeded",
        detail=f"{reply.strip()}\n\n{_KNOB_NOTE}", iterations=4, duration_seconds=1.0))
    (error,) = chan.errors
    assert error[0] == "turn failed — budget_exceeded"
    assert "(channels.unattended_max_actions)" in error[1] and "Found the three" not in error[1]
    assert chan.sent == [reply.strip()]


@pytest.mark.asyncio
async def test_a_root_budget_stop_with_no_reply_sends_only_the_error_line():
    chan = _Recorder(bus=object())
    await chan.on_turn_failed(TurnFailed(
        agent_id="a", session_id="s", reason="budget_exceeded", detail=_KNOB_NOTE,
        iterations=4, duration_seconds=1.0))
    assert len(chan.errors) == 1 and chan.sent == []


@pytest.mark.asyncio
async def test_a_budget_detail_without_the_loops_notice_is_all_reply():
    chan = _Recorder(bus=object())
    await chan.on_turn_failed(TurnFailed(
        agent_id="a", session_id="s", reason="budget_exceeded", detail="Max actions reached.",
        iterations=4, duration_seconds=1.0))
    assert chan.errors == [("turn failed — budget_exceeded", None)]
    assert chan.sent == ["Max actions reached."]


@pytest.mark.asyncio
async def test_zed_gets_the_budget_wrapup_only_when_nothing_streamed(tmp_path):
    failed = TurnFailed(agent_id="root", session_id="s", reason="budget_exceeded",
                        detail=f"partial work\n\n{_KNOB_NOTE}", iterations=4, duration_seconds=1.0)
    ch = _acp(tmp_path)
    ch._streamed_this_turn = False
    await ch.on_turn_failed(failed)
    assert "partial work" in ch._conn.texts

    streamed = _acp(tmp_path)
    streamed._streamed_this_turn = True
    await streamed.on_turn_failed(failed)
    assert "partial work" not in streamed._conn.texts


@pytest.mark.asyncio
async def test_a_real_budget_stop_reaches_a_channel_whole(mock_llm_client, bus):
    """End to end through the loop: the wrap-up the loop generated arrives whole, and the error
    line carries the loop's notice."""
    from tests.unit.test_budget_root_uncapped import _calls, _loop, _Registry

    chan = _Recorder(bus)
    bus.subscribe(TurnFailed, chan.on_turn_failed)
    loop = _loop(mock_llm_client, bus, _calls(mock_llm_client, 5), max_actions=2, local=True,
                 registry=_Registry())
    await loop.run_turn("go")
    (error,) = chan.errors
    assert error[0] == "turn failed — budget_exceeded" and "maximum of 2 tool calls" in error[1]
    assert all("[Budget limit reached" not in m for m in chan.sent)


@pytest.mark.asyncio
async def test_an_empty_root_reply_sends_nothing():
    chan = _Recorder(bus=object())
    await chan.on_turn_failed(TurnFailed(
        agent_id="a", session_id="s", reason="stuck_detected", detail="  ",
        iterations=4, duration_seconds=1.0))
    assert chan.sent == [] and chan.errors == []


@pytest.mark.asyncio
async def test_the_terminal_renders_a_stuck_turns_wrapup(tmp_path):
    """CR-51 at the terminal: the REPL never prints run_turn's return value and does not stream
    answer text, so the TurnFailed handler is the reply's only way onto the screen."""
    from io import StringIO

    from rich.console import Console

    from localharness.channels.terminal import TERMINAL_THEME, TerminalChannel
    from localharness.core.bus import EventBus

    ch = TerminalChannel(EventBus(), {}, history_file=str(tmp_path / ".hist"))
    ch._console = Console(file=StringIO(), force_terminal=False, width=120, theme=TERMINAL_THEME)
    await ch.on_turn_failed(TurnFailed(
        agent_id="root", session_id="s", reason="stuck_detected",
        detail="Wrote parser.py; tests still failing on the date case.",
        iterations=6, duration_seconds=1.0))
    out = ch._console.file.getvalue()
    assert "Wrote parser.py; tests still failing on the date case." in out
    assert "turn failed" not in out and "stuck_detected" not in out


def _acp(tmp_path):
    from localharness.channels.acp import AcpChannel

    class _Conn:
        def __init__(self):
            self.texts: list[str] = []

        async def session_update(self, session_id, update, **kw):
            self.texts.append(update.content.text)

    ch = AcpChannel(config_dir=str(tmp_path / "config"))
    ch._conn, ch._session_id = _Conn(), "sess"
    return ch


@pytest.mark.asyncio
async def test_zed_gets_the_wrapup_when_nothing_streamed_and_never_twice(tmp_path):
    failed = TurnFailed(agent_id="root", session_id="s", reason="stuck_detected",
                        detail="partial work", iterations=4, duration_seconds=1.0)
    ch = _acp(tmp_path)
    ch._streamed_this_turn = False
    await ch.on_turn_failed(failed)
    assert ch._conn.texts == ["partial work"]

    streamed = _acp(tmp_path)
    streamed._streamed_this_turn = True  # the wrap-up was typed into the panel as it generated
    await streamed.on_turn_failed(failed)
    assert streamed._conn.texts == []
