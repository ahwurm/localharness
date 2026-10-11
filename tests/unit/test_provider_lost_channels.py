"""ProviderLost reaches every channel: the base default, the terminal's own status slot, Discord
and ACP through an explicit subscription (a base default alone reaches nobody).

The text is the contract (D-08): `model server unreachable — waiting up to {N}s`, with an em dash,
N = math.ceil(wait_seconds) (IN-11: the phone page agrees); only the terminal's override adds `; Esc stops`. A child loop may
publish the event, so a ProviderLost carrying a parent_id is shown everywhere (D-44). The phone's
rows are in tests/unit/channels/test_mobile_reference_page.py and test_mobile_channel.py.
"""
from __future__ import annotations

from io import StringIO
from typing import Any

import pytest

from localharness.channels.acp import AcpChannel
from localharness.channels.base import ChannelAdapter
from localharness.channels.terminal import TERMINAL_THEME, TerminalChannel
from localharness.core.bus import EventBus
from localharness.core.events import Heartbeat, ProviderLost
from localharness.dispatch.adapters.discord import DiscordAdapter
from localharness.dispatch.channel import DispatchChannel
from tests.dispatch_support import install_fake_discord

EM_DASH = "—"
BASE_1800 = "model server unreachable — waiting up to 1800s"
BASE_30 = "model server unreachable — waiting up to 30s"
TERMINAL_1800 = BASE_1800 + "; Esc stops"


def _lost(wait: float = 1800.0, *, detail: str = "boom", parent_id: str | None = None) -> ProviderLost:
    return ProviderLost(
        agent_id="a", session_id="s", detail=detail, wait_seconds=wait, parent_id=parent_id,
    )


def _beat() -> Heartbeat:
    return Heartbeat(
        agent_id="a", session_id="s", iteration=1, context_utilization_pct=0.1,
    )


# ---------------------------------------------------------------- the base default


class _Recording(ChannelAdapter):
    """The smallest channel: records what send_message gets and inherits everything else."""

    def __init__(self) -> None:
        super().__init__(EventBus(), {})
        self.sent: list[tuple[str, Any, Any]] = []

    async def send_message(self, content, agent_id=None, metadata=None, **kw):
        self.sent.append((content, agent_id, metadata))

    async def send_streaming(self, token_stream, agent_id=None, **kw):
        return ""

    async def send_tool_call(self, *a, **kw):
        pass

    async def send_tool_result(self, *a, **kw):
        pass

    async def send_error(self, *a, **kw):
        pass

    async def request_permission(self, *a, **kw):
        return None

    async def request_input(self, *a, **kw):
        return ""

    async def read_input(self, *a, **kw):
        return ""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _recording() -> _Recording:
    return _Recording()


async def test_the_base_default_sends_the_bound_as_a_system_line():
    ch = _recording()
    await ch.on_provider_lost(_lost(1800.0))
    await ch.on_provider_lost(_lost(30.0))
    assert ch.sent == [
        (BASE_1800, "a", {"style": "system.info"}),
        (BASE_30, "a", {"style": "system.info"}),
    ]
    assert ord(ch.sent[0][0][len("model server unreachable ")]) == 0x2014  # an em dash, by code point


async def test_the_base_default_shows_a_child_stamped_event_and_never_the_detail():
    ch = _recording()
    await ch.on_provider_lost(_lost(30.0, detail="secret-detail", parent_id="parent-session"))
    assert [c for c, _, _ in ch.sent] == [BASE_30]
    assert "secret-detail" not in ch.sent[0][0]
    assert "Esc" not in ch.sent[0][0]


async def test_an_empty_detail_prints_the_same_bound_only_sentence():
    ch = _recording()
    await ch.on_provider_lost(_lost(30.0, detail=""))
    assert [c for c, _, _ in ch.sent] == [BASE_30]


# ---------------------------------------------------------------- the terminal


def _terminal() -> TerminalChannel:
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(
        file=StringIO(), force_terminal=True, width=120, theme=TERMINAL_THEME, highlight=False,
    )
    return ch


def _frags_text(ch: TerminalChannel) -> str:
    return "".join(text for _style, text in ch._box_status_frags())


async def test_terminal_box_mode_shows_the_line_first_even_while_a_turn_runs():
    ch = _terminal()
    ch._box_active = True
    ch.box_notify_working(True)
    assert ch._box_working is True and TERMINAL_1800 not in _frags_text(ch)
    await ch.on_provider_lost(_lost(1800.0))
    assert TERMINAL_1800 in _frags_text(ch)
    assert "working" not in _frags_text(ch), "the dropped-server sentence is the whole row"


async def test_terminal_box_line_is_cleared_by_heartbeat_stop_thinking_and_turn_end():
    ch = _terminal()
    ch._box_active = True
    ch.box_notify_working(True)

    await ch.on_provider_lost(_lost())
    assert TERMINAL_1800 in _frags_text(ch)
    await ch.on_heartbeat(_beat())
    assert ch._provider_lost_text == "" and TERMINAL_1800 not in _frags_text(ch)

    await ch.on_provider_lost(_lost())
    assert TERMINAL_1800 in _frags_text(ch)
    ch._stop_thinking()
    assert ch._provider_lost_text == "" and TERMINAL_1800 not in _frags_text(ch)

    await ch.on_provider_lost(_lost())
    ch.box_notify_working(False)
    assert ch._provider_lost_text == "" and _frags_text(ch) == ""


async def test_terminal_the_last_event_wins():
    ch = _terminal()
    ch._box_active = True
    await ch.on_provider_lost(_lost())
    await ch.on_heartbeat(_beat())
    assert TERMINAL_1800 not in _frags_text(ch)
    await ch.on_heartbeat(_beat())
    await ch.on_provider_lost(_lost())
    assert TERMINAL_1800 in _frags_text(ch)


async def test_terminal_shows_a_child_stamped_event():
    ch = _terminal()
    ch._box_active = True
    await ch.on_provider_lost(_lost(1800.0, parent_id="parent-session"))
    assert TERMINAL_1800 in _frags_text(ch)


async def test_terminal_empty_detail_and_whole_second_bounds():
    ch = _terminal()
    ch._box_active = True
    await ch.on_provider_lost(_lost(30.0, detail=""))
    assert "model server unreachable — waiting up to 30s; Esc stops" in _frags_text(ch)


@pytest.mark.parametrize(("seconds", "shown"), [(30.0, 30), (2.5, 3), (3.5, 4), (0.4, 1), (29.2, 30)])
async def test_every_python_surface_rounds_the_bound_up(seconds, shown):
    """IN-11: the same number as the phone page (Math.ceil), never round-half-even, and never "0s"."""
    ch = _recording()
    await ch.on_provider_lost(_lost(seconds))
    assert ch.sent[0][0] == f"model server unreachable — waiting up to {shown}s"
    term = _terminal()
    term._box_active = True
    await term.on_provider_lost(_lost(seconds))
    assert f"waiting up to {shown}s; Esc stops" in _frags_text(term)


class _FakeStatus:
    def __init__(self, label: str) -> None:
        self.labels = [label]
        self.stopped = False

    def start(self) -> None:
        pass

    def update(self, label: str, **kw) -> None:
        self.labels.append(label)

    def stop(self) -> None:
        self.stopped = True


async def test_terminal_classic_mode_puts_the_escaped_text_in_the_thinking_label():
    ch = _terminal()
    made: list[_FakeStatus] = []

    def status(label, **kw):
        made.append(_FakeStatus(label))
        return made[-1]

    ch._console.status = status  # type: ignore[assignment]
    assert not ch._box_active
    await ch.on_provider_lost(_lost(1800.0))
    assert len(made) == 1 and TERMINAL_1800 in made[0].labels[-1]
    # a second ProviderLost updates the same status rather than opening another
    await ch.on_provider_lost(_lost(30.0))
    assert len(made) == 1 and "waiting up to 30s; Esc stops" in made[0].labels[-1]
    # the next heartbeat clears the slot and the label goes back to the thinking indicator
    await ch.on_heartbeat(_beat())
    assert ch._provider_lost_text == "" and "unreachable" not in made[0].labels[-1]


async def test_terminal_classic_mode_shows_the_line_over_an_open_tool_burst():
    """WR-05: a family burst (a run of reads) stays open across iterations and its counter is the one
    live display; a drop during the next call must freeze it so the drop line is shown, not hidden."""
    ch = _terminal()
    made: list[_FakeStatus] = []

    def status(label, **kw):
        made.append(_FakeStatus(label))
        return made[-1]

    ch._console.status = status  # type: ignore[assignment]
    await ch.send_tool_call("web_search", {"query": "q"})
    await ch.send_tool_result("web_search", "ok", is_error=False)
    assert ch._burst is not None and ch._thinking is None

    await ch.on_provider_lost(_lost(1800.0))

    assert ch._burst is None  # frozen into its scrollback line
    assert made[0].stopped  # the burst's own counter stopped
    assert ch._thinking is made[-1] and TERMINAL_1800 in made[-1].labels[-1]
    assert "web_search" in ch._console.file.getvalue()


async def test_terminal_subscribes_on_start_and_unsubscribes_on_stop(tmp_path):
    ch = _terminal()
    ch._box_active = True  # no rich Status; the slot is what a delivery changes
    ch._history_file = str(tmp_path / "history")
    await ch.start()
    try:
        await ch.bus.publish(_lost(1800.0))
        assert ch._provider_lost_text == TERMINAL_1800
    finally:
        await ch.stop()
    assert ch._provider_lost_text == ""  # stop() clears the line with the thinking state
    await ch.bus.publish(_lost(1800.0))
    assert ch._provider_lost_text == "", "a stopped channel hears nothing"


# ---------------------------------------------------------------- Discord


async def test_discord_delivers_the_base_text_and_unsubscribes_on_stop(monkeypatch):
    install_fake_discord(monkeypatch)
    bus = EventBus()
    ch = DispatchChannel(bus, {"adapter": DiscordAdapter(token="t"), "allow": {"42"}})
    sent: list[str] = []

    async def _send(content: str) -> None:
        sent.append(content)

    ch._send = _send  # type: ignore[assignment]
    await ch.start()
    try:
        await bus.publish(_lost(1800.0))
        await bus.publish(_lost(30.0, detail="", parent_id="child-1"))  # D-44: still shown
        assert sent == [BASE_1800, BASE_30]
        assert all("Esc" not in line for line in sent)
    finally:
        await ch.stop()
    await bus.publish(_lost(1800.0))
    assert sent == [BASE_1800, BASE_30], "a stopped channel hears nothing"


# ---------------------------------------------------------------- ACP


class _Conn:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def session_update(self, session_id: str, update: Any, **kw: Any) -> None:
        self.texts.append(update.content.text)


async def test_acp_delivers_the_base_text_and_unsubscribes_on_stop(tmp_path):
    bus = EventBus()
    ch = AcpChannel(config_dir=str(tmp_path / "config"))
    conn = _Conn()
    ch._conn = conn
    ch._session_id = "sess"
    ch.bus = bus
    await ch.start()
    try:
        await bus.publish(_lost(1800.0))
        await bus.publish(_lost(30.0, detail="", parent_id="parent-session"))
        # the base text, set off by blank lines from the agent-message text around it (IN-10)
        assert conn.texts == [f"\n\n{BASE_1800}\n\n", f"\n\n{BASE_30}\n\n"]
        assert all("Esc" not in t for t in conn.texts)
    finally:
        await ch.stop()
    await bus.publish(_lost(1800.0))
    assert len(conn.texts) == 2, "a stopped channel hears nothing"


async def test_acp_keeps_the_line_apart_from_a_streamed_answer(tmp_path):
    """IN-10: a mid-stream drop puts the line between the partial answer and the re-sent one; ACP shows
    all three as one agent message, so the line is its own paragraph rather than glued to either."""
    bus = EventBus()
    ch = AcpChannel(config_dir=str(tmp_path / "config"))
    conn = _Conn()
    ch._conn = conn
    ch._session_id = "sess"
    ch.bus = bus
    await ch.send_message("partial ans")
    await ch.on_provider_lost(_lost(1800.0))
    await ch.send_message("the whole answer")
    joined = "".join(conn.texts)
    assert f"partial ans\n\n{BASE_1800}\n\nthe whole answer" == joined


def test_the_plugin_api_version_did_not_move():
    from localharness.plugins import api

    assert api.PLUGIN_API_VERSION == "1"


def test_spec_11_lists_on_provider_lost_in_the_base_class_block():
    """IN-12: a plugin-channel author reads the ChannelAdapter listing, not only the subscription table."""
    from pathlib import Path

    spec = (Path(__file__).resolve().parents[2] / "docs" / "specs" / "11-channels.md").read_text(
        encoding="utf-8")
    listing = spec[spec.index("    async def on_heartbeat(self, event: Heartbeat)"):]
    listing = listing[:listing.index("```")]
    assert "async def on_provider_lost(self, event: ProviderLost) -> None:" in listing
    assert "detail" in listing and "parent_id" in listing and "subscribe" in listing
