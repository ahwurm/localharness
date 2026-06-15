"""Tests for ChannelAdapter ABC, channel errors, and TerminalChannel."""
from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any, AsyncIterator

import pytest

from localharness.channels.errors import (
    ChannelError,
    ChannelInputError,
    ChannelOutputError,
    ChannelStartError,
    NotInteractiveError,
)


# ---------------------------------------------------------------------------
# Task 1: ABC enforcement and error hierarchy
# ---------------------------------------------------------------------------


class TestChannelErrors:
    def test_channel_error_is_exception(self):
        assert issubclass(ChannelError, Exception)

    def test_start_error_is_channel_error(self):
        assert issubclass(ChannelStartError, ChannelError)

    def test_output_error_is_channel_error(self):
        assert issubclass(ChannelOutputError, ChannelError)

    def test_input_error_is_channel_error(self):
        assert issubclass(ChannelInputError, ChannelError)

    def test_not_interactive_error_is_channel_error(self):
        assert issubclass(NotInteractiveError, ChannelError)

    def test_output_error_stores_channel_id_and_underlying(self):
        cause = ValueError("disk full")
        err = ChannelOutputError("terminal", cause)
        assert err.channel_id == "terminal"
        assert err.underlying is cause

    def test_input_error_stores_channel_id_and_underlying(self):
        cause = OSError("read error")
        err = ChannelInputError("terminal", cause)
        assert err.channel_id == "terminal"
        assert err.underlying is cause

    def test_not_interactive_stores_channel_id(self):
        err = NotInteractiveError("file")
        assert err.channel_id == "file"


class TestChannelAdapterABC:
    def test_cannot_instantiate_abc_directly(self):
        from localharness.channels.base import ChannelAdapter
        from localharness.core.bus import EventBus

        with pytest.raises(TypeError):
            ChannelAdapter(EventBus(), {})  # type: ignore[abstract]

    def test_incomplete_subclass_raises_type_error(self):
        from localharness.channels.base import ChannelAdapter

        class Incomplete(ChannelAdapter):
            channel_id = "incomplete"
            # Missing all abstract methods

        with pytest.raises(TypeError):
            Incomplete(None, {})  # type: ignore

    def test_complete_subclass_can_be_instantiated(self):
        from localharness.channels.base import ChannelAdapter
        from localharness.core.bus import EventBus

        class Complete(ChannelAdapter):
            channel_id = "complete"

            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, content, agent_id=None, metadata=None): pass
            async def send_streaming(self, token_stream, agent_id=None) -> str: return ""
            async def send_tool_call(self, tool_name, arguments, agent_id=None): pass
            async def send_tool_result(self, tool_name, result, is_error, agent_id=None): pass
            async def send_error(self, error, detail=None, agent_id=None): pass
            async def read_input(self, prompt="> ") -> str: return ""

        ch = Complete(EventBus(), {})
        assert ch.bus is not None


# ---------------------------------------------------------------------------
# Task 2: TerminalChannel tests
# ---------------------------------------------------------------------------


def make_terminal_channel(out: StringIO | None = None, err_out: StringIO | None = None):
    """Helper: create TerminalChannel with captured consoles for testing."""
    from rich.console import Console
    from localharness.channels.terminal import TerminalChannel, TERMINAL_THEME
    from localharness.core.bus import EventBus

    bus = EventBus()
    ch = TerminalChannel(bus, {})

    if out is not None:
        from rich.theme import Theme
        ch._console = Console(file=out, force_terminal=True, width=120, theme=TERMINAL_THEME, highlight=False)
    if err_out is not None:
        from rich.theme import Theme
        ch._err_console = Console(file=err_out, stderr=False, force_terminal=True, width=120, theme=TERMINAL_THEME, highlight=False)
    return ch


class TestTerminalChannelBasics:
    def test_channel_id_is_terminal(self):
        from localharness.channels.terminal import TerminalChannel
        assert TerminalChannel.channel_id == "terminal"

    @pytest.mark.asyncio
    async def test_send_message_with_agent_id_prints_panel(self):
        out = StringIO()
        ch = make_terminal_channel(out=out)
        await ch.send_message("Hello world", agent_id="myagent")
        rendered = out.getvalue()
        assert "myagent" in rendered
        assert "Hello world" in rendered

    @pytest.mark.asyncio
    async def test_send_message_without_agent_id_prints_plain(self):
        out = StringIO()
        ch = make_terminal_channel(out=out)
        await ch.send_message("Just plain text")
        rendered = out.getvalue()
        assert "Just plain text" in rendered
        # No Panel border chars — no title visible
        assert "╭" not in rendered  # panel top-left corner

    @pytest.mark.asyncio
    async def test_send_tool_call_uses_diamond(self):
        out = StringIO()
        ch = make_terminal_channel(out=out)
        await ch.send_tool_call("file_read", {"path": "src/main.py"})
        rendered = out.getvalue()
        assert "\u25c6" in rendered  # diamond ◆
        assert "file_read" in rendered

    @pytest.mark.asyncio
    async def test_send_tool_call_includes_key_arg(self):
        out = StringIO()
        ch = make_terminal_channel(out=out)
        await ch.send_tool_call("bash", {"command": "pytest tests/"})
        rendered = out.getvalue()
        assert "pytest tests/" in rendered

    @pytest.mark.asyncio
    async def test_send_tool_result_success_uses_checkmark(self):
        out = StringIO()
        ch = make_terminal_channel(out=out)
        await ch.send_tool_result("file_read", "line1\nline2\nline3", is_error=False)
        rendered = out.getvalue()
        assert "\u2713" in rendered  # checkmark ✓
        assert "file_read" in rendered

    @pytest.mark.asyncio
    async def test_send_tool_result_error_uses_cross(self):
        out = StringIO()
        ch = make_terminal_channel(out=out)
        await ch.send_tool_result("bash", "TypeError: expected str", is_error=True)
        rendered = out.getvalue()
        assert "\u2717" in rendered  # cross ✗
        assert "bash" in rendered

    @pytest.mark.asyncio
    async def test_send_error_writes_to_err_console(self):
        err_out = StringIO()
        ch = make_terminal_channel(err_out=err_out)
        await ch.send_error("Something went wrong", detail="line1\nline2")
        rendered = err_out.getvalue()
        assert "Something went wrong" in rendered

    @pytest.mark.asyncio
    async def test_send_streaming_returns_full_text(self):
        out = StringIO()
        ch = make_terminal_channel(out=out)

        async def token_gen() -> AsyncIterator[str]:
            for t in ["Hello", " ", "world"]:
                yield t

        result = await ch.send_streaming(token_gen())
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_send_streaming_acquires_output_lock(self):
        """Verify lock is held: a concurrent send_message should queue behind streaming."""
        out = StringIO()
        ch = make_terminal_channel(out=out)
        lock_held_during_stream = False

        async def slow_stream() -> AsyncIterator[str]:
            nonlocal lock_held_during_stream
            # After first token, check if lock is locked
            yield "token"
            lock_held_during_stream = ch._output_lock.locked()

        await ch.send_streaming(slow_stream())
        assert lock_held_during_stream is True

    @pytest.mark.asyncio
    async def test_read_input_raises_channel_start_error_if_not_started(self):
        ch = make_terminal_channel()
        with pytest.raises(ChannelStartError):
            await ch.read_input()

    @pytest.mark.asyncio
    async def test_heartbeat_captures_context_pct_for_input_bubble(self):
        from localharness.core.events import Heartbeat
        ch = make_terminal_channel()
        assert ch._context_pct is None  # nothing shown before the first measurement
        await ch.on_heartbeat(Heartbeat(
            agent_id="a", session_id="s", iteration=1, context_utilization_pct=42.0,
        ))
        assert ch._context_pct == 42.0


class TestContextMeter:
    def test_levels_step_with_usage(self):
        from localharness.channels.terminal import _ctx_segments
        levels = {p: _ctx_segments(p)[0][0][0] for p in (20, 55, 72, 88)}
        assert levels == {
            20: "class:ctx-low", 55: "class:ctx-mid",
            72: "class:ctx-high", 88: "class:ctx-crit",
        }

    def test_crit_at_compaction_threshold(self):
        from localharness.channels.terminal import _ctx_segments
        assert _ctx_segments(80)[0][0][0] == "class:ctx-crit"  # summary compaction fires at 80%

    def test_bar_fills_and_width_matches_plaintext(self):
        from localharness.channels.terminal import _ctx_segments
        for pct in (0, 5, 50, 100):
            frags, width = _ctx_segments(pct)
            plain = "".join(text for _, text in frags)
            assert plain.count("█") == min(10, pct // 10)  # filled cells
            assert len(plain) == width  # width must match for border alignment

    def test_clamps_out_of_range(self):
        from localharness.channels.terminal import _ctx_segments
        assert "120%" not in "".join(t for _, t in _ctx_segments(120.0)[0])
        assert _ctx_segments(120.0)[0][0][1].count("█") == 10
        assert _ctx_segments(-5.0)[0][0][1] == ""  # no filled cells below zero


class TestFormatArgsCompact:
    def test_string_truncation(self):
        from localharness.channels.terminal import _format_args_compact
        long_str = "a" * 80
        result = _format_args_compact({"path": long_str})
        assert "..." in result or "\u2026" in result  # ellipsis

    def test_boolean_lowercase(self):
        from localharness.channels.terminal import _format_args_compact
        result = _format_args_compact({"flag": True, "other": False})
        assert "true" in result
        assert "false" in result

    def test_list_compact_when_long(self):
        from localharness.channels.terminal import _format_args_compact
        result = _format_args_compact({"items": [1, 2, 3, 4, 5]})
        assert "[5 items]" in result

    def test_dict_compact(self):
        from localharness.channels.terminal import _format_args_compact
        result = _format_args_compact({"opts": {"a": 1, "b": 2}})
        assert "{2 keys}" in result

    def test_none_becomes_null(self):
        from localharness.channels.terminal import _format_args_compact
        result = _format_args_compact({"val": None})
        assert "null" in result
