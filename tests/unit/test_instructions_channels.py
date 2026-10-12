"""Every channel delivers the project-instructions line as a receipt and renders the new events.

Delivery is the receipt: a channel calls the handle's ``on_delivered`` (here a recorded list)
only after its own send proved a person was reached. The base never does, so a plugin channel that
knows nothing about instructions writes no record.
"""
from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any

import pytest

from localharness.channels.acp import AcpChannel
from localharness.channels.base import ChannelAdapter, _call_receipt
from localharness.core.bus import EventBus
from localharness.core.events import InstructionsChanged, NarrowingNotice

pytestmark = pytest.mark.asyncio


def _changed(cause: str = "start", text: str = "Using CLAUDE.md (5.1 KB); /instructions off",
             **over: Any) -> InstructionsChanged:
    base = dict(agent_id="a", session_id="s", path="/r/CLAUDE.md", name="CLAUDE.md", size=5,
                state="yes", cause=cause, text=text)
    base.update(over)
    return InstructionsChanged(**base)


def _narrowing(text: str = "workspace value ignored; accept it in ~/.localharness/config.yaml"):
    return NarrowingNotice(agent_id="a", session_id="s", key="k", file="f", accept_file="g",
                           text=text)


class Receipts:
    """Stands in for the handle's ``on_delivered``."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


# ------------------------------------------------------------------ the base defaults


class PluginStyleChannel(ChannelAdapter):
    """A plugin channel written before instructions existed: only the abstract methods."""

    channel_id = "plugin"

    def __init__(self, bus: EventBus) -> None:
        super().__init__(bus, {})
        self.sent: list[tuple[str, dict | None]] = []

    async def start(self) -> None:
        self.bus.subscribe(InstructionsChanged, self.on_instructions_changed)
        self.bus.subscribe(NarrowingNotice, self.on_narrowing_notice)

    async def stop(self) -> None:
        pass

    async def send_message(self, content, agent_id=None, metadata=None) -> None:
        self.sent.append((content, metadata))

    async def send_streaming(self, token_stream, agent_id=None) -> str:
        return ""

    async def send_tool_call(self, tool_name, arguments, agent_id=None) -> None:
        pass

    async def send_tool_result(self, tool_name, result, is_error, agent_id=None) -> None:
        pass

    async def send_error(self, error, detail=None, agent_id=None) -> None:
        pass

    async def read_input(self, prompt: str = "> ") -> str:
        return ""


async def test_a_plugin_channel_keeps_the_start_line_and_never_writes_a_record():
    bus = EventBus()
    channel = PluginStyleChannel(bus)
    receipts = Receipts()
    channel.instructions_on_delivered = receipts
    await channel.start()

    await bus.publish(_changed("start", text="T"))

    assert channel.sent == []
    assert channel._first_contact_notice == "T"
    assert receipts.calls == 0


async def test_a_too_large_line_is_first_contact_too():
    bus = EventBus()
    channel = PluginStyleChannel(bus)
    receipts = Receipts()
    channel.instructions_on_delivered = receipts
    await channel.start()

    await bus.publish(_changed("too_large", text="CLAUDE.md is too large"))

    assert channel.sent == []
    assert channel._first_contact_notice == "CLAUDE.md is too large"
    assert receipts.calls == 0


async def test_a_change_line_is_one_system_info_message():
    bus = EventBus()
    channel = PluginStyleChannel(bus)
    await channel.start()

    await bus.publish(_changed("adopted", text="CLAUDE.md changed; using it"))

    assert channel.sent == [("CLAUDE.md changed; using it", {"style": "system.info"})]
    assert channel._first_contact_notice is None


async def test_a_child_stamped_change_line_is_not_filtered():
    bus = EventBus()
    channel = PluginStyleChannel(bus)
    await channel.start()

    await bus.publish(_changed("shadowed", text="shadow", parent_id="child-1"))

    assert [c for c, _ in channel.sent] == ["shadow"]


async def test_a_narrowing_notice_is_one_system_info_message():
    bus = EventBus()
    channel = PluginStyleChannel(bus)
    await channel.start()

    await bus.publish(_narrowing("narrowed"))

    assert channel.sent == [("narrowed", {"style": "system.info"})]


async def test_the_class_attributes_exist_without_init():
    class NoSuper(PluginStyleChannel):
        def __init__(self) -> None:  # deliberately skips super().__init__
            self.sent = []

    channel = NoSuper()
    assert channel.instructions_on_delivered is None
    assert channel._first_contact_notice is None
    await channel.notice_on_first_contact("x", None)
    assert channel._first_contact_notice == "x"


async def test_a_raising_receipt_is_swallowed():
    def boom() -> None:
        raise RuntimeError("store down")

    _call_receipt(boom)  # never raises
    _call_receipt(None)


async def test_the_plugin_api_version_is_unchanged():
    from localharness.plugins.api import PLUGIN_API_VERSION

    assert PLUGIN_API_VERSION == "1"


# ------------------------------------------------------------------ ACP


class FakeConn:
    """The SDK connection: records ``session_update`` and can be told to fail."""

    def __init__(self) -> None:
        self.updates: list[Any] = []
        self.fail = False

    async def session_update(self, session_id: str, update: Any, **kw: Any) -> None:
        if self.fail:
            raise ConnectionError("editor went away")
        self.updates.append(update)

    def texts(self) -> list[str]:
        return [u.content.text for u in self.updates if hasattr(u, "content")]


def _acp(bus: EventBus | None = None, *, connected: bool = True) -> tuple[AcpChannel, FakeConn]:
    channel = AcpChannel()
    channel.bus = bus or EventBus()
    conn = FakeConn()
    if connected:
        channel._conn = conn
        channel._session_id = "sess"
    return channel, conn


async def test_acp_a_start_line_reaches_the_editor_and_only_then_calls_back():
    bus = EventBus()
    channel, conn = _acp(bus)
    receipts = Receipts()
    channel.instructions_on_delivered = receipts
    await channel.start()

    await bus.publish(_changed("start", text="T"))

    assert conn.texts() == ["T"]
    assert receipts.calls == 1


async def test_acp_without_a_connection_queues_and_the_next_prompt_delivers():
    channel, conn = _acp(connected=False)
    receipts = Receipts()

    await channel.notice_on_first_contact("T", receipts)

    assert receipts.calls == 0
    assert len(channel._pending_notices) == 1
    channel._conn, channel._session_id = conn, "sess"

    await channel._drain_pending_notices()

    assert conn.texts() == ["T"]
    assert receipts.calls == 1
    assert channel._pending_notices == []


async def test_acp_a_failing_update_keeps_the_pair_queued_and_never_raises():
    channel, conn = _acp()
    receipts = Receipts()
    conn.fail = True

    await channel.notice_on_first_contact("T", receipts)

    assert receipts.calls == 0
    assert channel._pending_notices == [("T", receipts)]
    conn.fail = False
    await channel._drain_pending_notices()
    assert conn.texts() == ["T"] and receipts.calls == 1


async def test_acp_update_checked_is_false_without_a_connection():
    channel, _ = _acp(connected=False)
    assert await channel._update_checked(object()) is False


async def test_acp_the_mcp_notice_still_reaches_the_first_prompt_with_no_callback():
    channel, conn = _acp()
    channel._pending_notices.append(("MCP not connected", None))

    await channel._drain_pending_notices()

    assert conn.texts() == ["MCP not connected"]


async def test_acp_a_raising_callback_is_swallowed():
    channel, conn = _acp()

    def boom() -> None:
        raise RuntimeError("store down")

    await channel.notice_on_first_contact("T", boom)

    assert conn.texts() == ["T"]
    assert channel._pending_notices == []


async def test_acp_a_change_line_goes_through_send_message():
    bus = EventBus()
    channel, conn = _acp(bus)
    await channel.start()

    await bus.publish(_changed("adopted", text="changed"))
    await bus.publish(_narrowing("narrowed"))

    assert conn.texts() == ["changed", "narrowed"]


async def test_acp_after_stop_a_published_event_reaches_nothing():
    bus = EventBus()
    channel, conn = _acp(bus)
    receipts = Receipts()
    channel.instructions_on_delivered = receipts
    await channel.start()
    await channel.stop()

    await bus.publish(_changed("start"))
    await bus.publish(_changed("adopted"))
    await bus.publish(_narrowing())

    assert conn.updates == []
    assert receipts.calls == 0


# ------------------------------------------------------------------ terminal

START_LINE = "Using CLAUDE.md (5.1 KB); /instructions off"
CHANGE_LINE = "CLAUDE.md changed by the model; not used — /instructions on"


def _terminal(tmp_path):
    from rich.console import Console

    from localharness.channels.terminal import TERMINAL_THEME, TerminalChannel

    bus = EventBus()
    channel = TerminalChannel(bus, {}, history_file=str(tmp_path / ".hist"))
    channel._console = Console(file=StringIO(), force_terminal=True, width=120,
                               theme=TERMINAL_THEME)
    return bus, channel


def _capture_bubble(monkeypatch) -> list[str]:
    from localharness.channels import terminal as term_mod

    seen: list[str] = []

    class _FakeApp:
        async def run_async(self):
            return "typed"

    def _fake_build(history, prompt, hint, context_pct=None, model_names_fn=None, **_kw):
        seen.append(hint)
        return _FakeApp()

    monkeypatch.setattr(term_mod, "_build_input_app", _fake_build)
    return seen


async def test_terminal_the_first_bubble_hint_is_the_receipt_once(monkeypatch, tmp_path):
    _, ch = _terminal(tmp_path)
    await ch.start()
    seen = _capture_bubble(monkeypatch)
    receipts = Receipts()
    ch.instructions_on_delivered = receipts
    ch.instructions_hint = START_LINE
    ch.first_prompt_hint = "/help for commands. · " + START_LINE

    await ch.read_input()
    await ch.read_input()
    await ch.stop()

    assert seen[0] == "/help for commands. · " + START_LINE and seen[1] == ""
    assert receipts.calls == 1
    assert ch.instructions_hint == ""


async def test_terminal_the_line_alone_is_also_a_receipt(monkeypatch, tmp_path):
    _, ch = _terminal(tmp_path)
    await ch.start()
    _capture_bubble(monkeypatch)
    receipts = Receipts()
    ch.instructions_on_delivered = receipts
    ch.instructions_hint = START_LINE
    ch.first_prompt_hint = START_LINE

    await ch.read_input()
    await ch.stop()

    assert receipts.calls == 1


async def test_terminal_a_hint_without_the_line_is_no_receipt(monkeypatch, tmp_path):
    _, ch = _terminal(tmp_path)
    await ch.start()
    _capture_bubble(monkeypatch)
    receipts = Receipts()
    ch.instructions_on_delivered = receipts
    ch.instructions_hint = START_LINE
    ch.first_prompt_hint = "/help for commands."

    await ch.read_input()
    await ch.stop()

    assert receipts.calls == 0
    assert ch.instructions_hint == START_LINE


async def test_terminal_an_empty_instructions_hint_is_no_receipt(monkeypatch, tmp_path):
    _, ch = _terminal(tmp_path)
    await ch.start()
    _capture_bubble(monkeypatch)
    receipts = Receipts()
    ch.instructions_on_delivered = receipts
    ch.first_prompt_hint = "/help for commands."

    await ch.read_input()
    await ch.stop()

    assert receipts.calls == 0


async def test_terminal_a_raising_receipt_never_breaks_the_bubble(monkeypatch, tmp_path):
    _, ch = _terminal(tmp_path)
    await ch.start()
    _capture_bubble(monkeypatch)

    def boom() -> None:
        raise RuntimeError("store down")

    ch.instructions_on_delivered = boom
    ch.instructions_hint = START_LINE
    ch.first_prompt_hint = START_LINE

    assert await ch.read_input() == "typed"
    await ch.stop()


async def test_terminal_the_box_path_consumes_the_hint_as_the_receipt(tmp_path):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    _, ch = _terminal(tmp_path)
    ch._history = InMemoryHistory()
    receipts = Receipts()
    ch.instructions_on_delivered = receipts
    ch.instructions_hint = START_LINE
    ch.first_prompt_hint = START_LINE
    q: asyncio.Queue = asyncio.Queue()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        await ch.start_input_box(q, on_interrupt=lambda: None)
        try:
            assert receipts.calls == 1
            assert START_LINE in ch._box_placeholder()
        finally:
            await ch.stop_input_box()


async def test_terminal_a_start_event_changes_nothing(tmp_path, capsys):
    bus, ch = _terminal(tmp_path)
    await ch.start()
    receipts = Receipts()
    ch.instructions_on_delivered = receipts

    await bus.publish(_changed("start", text=START_LINE))
    await ch.stop()

    assert ch._instructions_status == ""
    assert receipts.calls == 0
    assert ch._console.file.getvalue() == ""


async def test_terminal_a_change_line_shows_in_the_next_bubble_only(monkeypatch, tmp_path):
    bus, ch = _terminal(tmp_path)
    await ch.start()
    seen = _capture_bubble(monkeypatch)

    await bus.publish(_changed("harness_written", text=CHANGE_LINE))
    await ch.read_input()
    await ch.read_input()
    await ch.stop()

    assert CHANGE_LINE in seen[0]
    assert seen[1] == ""


async def test_terminal_the_change_line_joins_the_first_hint(monkeypatch, tmp_path):
    bus, ch = _terminal(tmp_path)
    await ch.start()
    seen = _capture_bubble(monkeypatch)
    ch.first_prompt_hint = "/help for commands."

    await bus.publish(_changed("adopted", text="changed"))
    await ch.read_input()
    await ch.stop()

    assert seen == ["/help for commands. · changed"]


async def test_terminal_the_box_shows_the_line_until_the_next_submit(tmp_path):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    bus, ch = _terminal(tmp_path)
    ch._history = InMemoryHistory()
    q: asyncio.Queue = asyncio.Queue()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        await ch.start_input_box(q, on_interrupt=lambda: None)
        try:
            await ch.on_instructions_changed(_changed("shadowed", text=CHANGE_LINE))
            assert CHANGE_LINE in ch._box_placeholder()
            inp.send_text("hello\r")
            assert await asyncio.wait_for(q.get(), timeout=10.0) == ("submit", "hello")
            assert CHANGE_LINE not in ch._box_placeholder()
        finally:
            await ch.stop_input_box()


async def test_terminal_clear_instructions_status_removes_it(monkeypatch, tmp_path):
    bus, ch = _terminal(tmp_path)
    await ch.start()
    seen = _capture_bubble(monkeypatch)

    await bus.publish(_changed("harness_written", text=CHANGE_LINE))
    ch.clear_instructions_status()
    await ch.read_input()
    await ch.stop()

    assert seen == [""]


async def test_terminal_a_narrowing_notice_lands_in_the_same_slot_joined(monkeypatch, tmp_path):
    bus, ch = _terminal(tmp_path)
    await ch.start()
    seen = _capture_bubble(monkeypatch)

    await bus.publish(_changed("refused", text="CLAUDE.md not used (unreadable)"))
    await bus.publish(_narrowing("narrowed"))
    await ch.read_input()
    await ch.stop()

    assert seen == ["CLAUDE.md not used (unreadable) · narrowed"]


async def test_terminal_control_sequences_are_stripped_and_brackets_stay_literal(
    monkeypatch, tmp_path
):
    bus, ch = _terminal(tmp_path)
    await ch.start()
    seen = _capture_bubble(monkeypatch)

    await bus.publish(_changed("refused", text="\x1b[2Jx [b]y[/b]"))
    await ch.read_input()
    await ch.stop()

    assert seen == ["x [b]y[/b]"]
    assert "\\" not in seen[0] and "\x1b" not in seen[0]


async def test_terminal_subscribes_both_events_in_start_and_drops_them_in_stop(tmp_path):
    bus, ch = _terminal(tmp_path)
    await ch.start()
    await bus.publish(_changed("adopted", text="one"))
    await bus.publish(_narrowing("two"))
    assert ch._instructions_status == "one · two"

    await ch.stop()
    ch._instructions_status = ""
    await bus.publish(_changed("adopted", text="three"))
    await bus.publish(_narrowing("four"))

    assert ch._instructions_status == ""
