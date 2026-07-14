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


def make_terminal_channel(
    out: StringIO | None = None, err_out: StringIO | None = None, force_terminal: bool = True,
):
    """Helper: create TerminalChannel with captured consoles for testing.

    force_terminal=False gives a non-TTY console (is_terminal False) — the capture/pipe
    path, where live spinners must be skipped and only frozen lines printed."""
    from rich.console import Console
    from localharness.channels.terminal import TerminalChannel, TERMINAL_THEME
    from localharness.core.bus import EventBus

    bus = EventBus()
    ch = TerminalChannel(bus, {})

    if out is not None:
        ch._console = Console(file=out, force_terminal=force_terminal, width=120, theme=TERMINAL_THEME, highlight=False)
    if err_out is not None:
        ch._err_console = Console(file=err_out, stderr=False, force_terminal=force_terminal, width=120, theme=TERMINAL_THEME, highlight=False)
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


class TestThinkingIndicator:
    """REPL-02: a rich console.status spinner runs while the model generates and is torn
    down the instant any real output lands — never over the input bubble or a stream."""

    @staticmethod
    def _channel():
        # force_terminal (via make_terminal_channel) so rich Status renders headlessly
        return make_terminal_channel(out=StringIO())

    @staticmethod
    def _heartbeat():
        from localharness.core.events import Heartbeat
        return Heartbeat(
            agent_id="a", session_id="s", iteration=1,
            context_utilization_pct=10.0, last_tool=None,
        )

    @pytest.mark.asyncio
    async def test_heartbeat_starts_thinking_indicator(self):
        ch = self._channel()
        assert ch._state == "IDLE"
        await ch.on_heartbeat(self._heartbeat())
        assert ch._thinking is not None      # spinner live the moment generation begins
        assert ch._context_pct == 10.0       # existing utilization tracking preserved
        ch._stop_thinking()                  # tidy the daemon refresh thread

    @pytest.mark.asyncio
    async def test_send_message_stops_thinking_indicator(self):
        ch = self._channel()
        await ch.on_heartbeat(self._heartbeat())
        assert ch._thinking is not None
        await ch.send_message("hello")
        assert ch._thinking is None

    @pytest.mark.asyncio
    async def test_tool_output_stops_thinking_indicator(self):
        ch = self._channel()
        await ch.on_heartbeat(self._heartbeat())
        await ch.send_tool_call("glob", {"pattern": "*.py"})
        assert ch._thinking is None
        await ch.on_heartbeat(self._heartbeat())   # per-iteration rhythm: restarts
        assert ch._thinking is not None
        await ch.send_tool_result("glob", "ok", is_error=False)
        assert ch._thinking is None

    @pytest.mark.asyncio
    async def test_thinking_never_starts_outside_idle(self):
        for state in ("WAITING_INPUT", "STREAMING"):
            ch = self._channel()
            ch._state = state
            await ch.on_heartbeat(self._heartbeat())
            assert ch._thinking is None, f"indicator must not start while {state}"

    @pytest.mark.asyncio
    async def test_stop_cleans_up_thinking(self):
        ch = self._channel()
        await ch.on_heartbeat(self._heartbeat())
        assert ch._thinking is not None
        await ch.stop()
        assert ch._thinking is None          # stop() must not leak a running Status thread


class TestDreamingIndicator:
    """#20: a quiet '· dreaming…' console.status runs while a background memory
    consolidation/mining pass is in flight and is torn down the instant it ends, the
    user starts typing, or a turn begins. Extends the REPL-02 thinking machinery (same
    console.status, IDLE-only start, stop-first) — the dot can never draw over the input
    bubble because WAITING_INPUT is not IDLE. Terminal-only, driven by bus events."""

    @staticmethod
    def _channel():
        # force_terminal so rich Status renders headlessly (mirrors the thinking tests)
        return make_terminal_channel(out=StringIO())

    @staticmethod
    def _started():
        from localharness.core.events import ConsolidationStarted
        return ConsolidationStarted(agent_id="a")

    @staticmethod
    def _finished():
        from localharness.core.events import ConsolidationFinished
        return ConsolidationFinished(agent_id="a")

    def test_dreaming_label_is_exact(self):
        """The official label is exactly '· dreaming…' (middle-dot + ellipsis)."""
        from localharness.channels.terminal import _DREAMING_LABEL
        assert _DREAMING_LABEL == "· dreaming…"
        assert [hex(ord(c)) for c in _DREAMING_LABEL[:1]] == ["0xb7"]  # leading middle-dot

    @pytest.mark.asyncio
    async def test_pass_start_shows_dreaming_when_idle(self):
        ch = self._channel()
        assert ch._state == "IDLE"
        await ch.on_consolidation_started(self._started())
        assert ch._dreaming is not None      # the dot appears the moment the pass starts
        ch._stop_dreaming()                  # tidy the daemon refresh thread

    @pytest.mark.asyncio
    async def test_pass_end_clears_dreaming(self):
        ch = self._channel()
        await ch.on_consolidation_started(self._started())
        assert ch._dreaming is not None
        await ch.on_consolidation_finished(self._finished())
        assert ch._dreaming is None          # cleared instantly when the pass ends

    @pytest.mark.asyncio
    async def test_dreaming_never_starts_over_the_prompt_or_a_stream(self):
        # WAITING_INPUT (bubble up) and STREAMING (mid-generation): a pass-start must
        # never draw the dot — this is the "never disturb the input prompt" guarantee.
        for state in ("WAITING_INPUT", "STREAMING"):
            ch = self._channel()
            ch._state = state
            await ch.on_consolidation_started(self._started())
            assert ch._dreaming is None, f"dreaming must not start while {state}"

    @pytest.mark.asyncio
    async def test_output_path_stops_dreaming_first(self):
        # _stop_thinking is the stop-first primitive every output path calls BEFORE it
        # prints — it must tear the dreaming dot down too (no dot bleeding into output).
        ch = self._channel()
        await ch.on_consolidation_started(self._started())
        assert ch._dreaming is not None
        ch._stop_thinking()
        assert ch._dreaming is None

    @pytest.mark.asyncio
    async def test_send_message_clears_dreaming(self):
        ch = self._channel()
        await ch.on_consolidation_started(self._started())
        assert ch._dreaming is not None
        await ch.send_message("hello")       # real output landing clears the dot first
        assert ch._dreaming is None

    @pytest.mark.asyncio
    async def test_turn_begin_replaces_dreaming_with_thinking(self):
        from localharness.core.events import Heartbeat
        ch = self._channel()
        await ch.on_consolidation_started(self._started())
        assert ch._dreaming is not None
        await ch.on_heartbeat(Heartbeat(
            agent_id="a", session_id="s", iteration=1, context_utilization_pct=10.0,
        ))
        assert ch._dreaming is None          # a turn beginning clears the dreaming dot…
        assert ch._thinking is not None      # …and the thinking spinner takes the slot
        ch._stop_thinking()

    @pytest.mark.asyncio
    async def test_stop_cleans_up_dreaming(self):
        ch = self._channel()
        await ch.on_consolidation_started(self._started())
        assert ch._dreaming is not None
        await ch.stop()
        assert ch._dreaming is None          # stop() must not leak a running Status thread

    @pytest.mark.asyncio
    async def test_terminal_subscribes_via_bus_and_discord_ignores(self, tmp_path):
        # Terminal wires the dreaming dot through its bus subscription in start(); a
        # representative non-interactive channel (Discord) has no such surface at all —
        # the dot is REPL-terminal-only (the 33.3 thinking-spinner precedent).
        from localharness.channels.discord import DiscordChannel
        ch = self._channel()
        ch._history_file = str(tmp_path / ".repl_history")
        await ch.start()
        try:
            await ch.bus.publish(self._started())
            assert ch._dreaming is not None  # delivered via the real bus subscription
        finally:
            await ch.stop()
        assert not hasattr(DiscordChannel, "on_consolidation_started")
        assert not hasattr(DiscordChannel, "on_consolidation_finished")


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


class TestBurstConsolidation:
    """Consecutive same-family tool calls collapse into ONE live counter line — the
    localharness.dev demo line (`◆ web_search · web_fetch — the open web, a fresh
    window each · 30/30`) becomes real product output instead of a line per call
    (the run-run-run scrollback the site demos had to consolidate by hand; owner
    call 2026-07-02 "consolidate — keep true behavior"). Per-call truth stays on
    the bus ledger (bus-events.jsonl); the channel change is display-only."""

    WEB_FROZEN = "◆ web_search · web_fetch — the open web, a fresh window each · 3/3"

    @staticmethod
    def _pipe_channel():
        """Non-TTY console: no live frames in the capture, only frozen lines —
        assertions can count occurrences exactly."""
        out = StringIO()
        return make_terminal_channel(out=out, force_terminal=False), out

    @pytest.mark.asyncio
    async def test_web_burst_prints_nothing_until_closed(self):
        ch, out = self._pipe_channel()
        await ch.send_tool_call("web_search", {"query": "DGX Spark specs"})
        await ch.send_tool_result("web_search", "r1\nr2", is_error=False)
        await ch.send_tool_call("web_fetch", {"url": "https://nvidia.com"})
        await ch.send_tool_result("web_fetch", "page", is_error=False)
        assert out.getvalue() == ""  # burst open: no per-call spam, no partial line

    @pytest.mark.asyncio
    async def test_web_burst_freezes_to_demo_line_on_next_tool(self):
        ch, out = self._pipe_channel()
        for tool, arg in (
            ("web_search", {"query": "a"}),
            ("web_fetch", {"url": "b"}),
            ("web_search", {"query": "c"}),  # repeat tool: first-use order kept, no dup
        ):
            await ch.send_tool_call(tool, arg)
            await ch.send_tool_result(tool, "ok", is_error=False)
        await ch.send_tool_call("memory_search", {"query": "spark"})
        rendered = out.getvalue()
        assert rendered.count(self.WEB_FROZEN) == 1
        assert "✓ web results — UNTRUSTED, treated as data only" in rendered
        assert "✓ web_search" not in rendered  # per-call results absorbed
        assert rendered.index(self.WEB_FROZEN) < rendered.index("memory_search")

    @pytest.mark.asyncio
    async def test_errors_absorbed_and_annotated(self):
        ch, out = self._pipe_channel()
        for i in range(3):
            await ch.send_tool_call("web_fetch", {"url": f"u{i}"})
            await ch.send_tool_result(
                "web_fetch", "429 Too Many Requests" if i == 1 else "ok", is_error=(i == 1),
            )
        await ch.send_message("done")
        rendered = out.getvalue()
        assert "◆ web_fetch — the open web, a fresh window each · 3/3 · 1 error" in rendered
        assert "✗" not in rendered      # no red per-call line mid-burst
        assert "429" not in rendered    # detail lives in the bus ledger, not scrollback

    @pytest.mark.asyncio
    async def test_section_reads_burst_has_own_label_and_no_untrusted_note(self):
        ch, out = self._pipe_channel()
        for h in ("5613cfd00330", "695bb3032f1f"):
            await ch.send_tool_call("tool_result_get", {"handle": h})
            await ch.send_tool_result("tool_result_get", "section text", is_error=False)
        await ch.send_message("combined")
        rendered = out.getvalue()
        assert "◆ tool_result_get — section reads, a fresh window each · 2/2" in rendered
        assert "UNTRUSTED" not in rendered

    @pytest.mark.asyncio
    async def test_interposed_call_splits_bursts(self):
        """The demo sequence: researcher's web burst → ◆ agent search-verifier → the
        verifier's own burst. Each burst freezes with its own honest count."""
        ch, out = self._pipe_channel()
        for i in range(2):
            await ch.send_tool_call("web_search", {"query": f"q{i}"})
            await ch.send_tool_result("web_search", "ok", is_error=False)
        await ch.send_tool_call("agent", {"agent_name": "search-verifier"})
        await ch.send_tool_call("web_fetch", {"url": "docs.nvidia.com"})
        await ch.send_tool_result("web_fetch", "ok", is_error=False)
        await ch.send_message("verdict")
        rendered = out.getvalue()
        assert "◆ web_search — the open web, a fresh window each · 2/2" in rendered
        assert "◆ web_fetch — the open web, a fresh window each · 1/1" in rendered
        assert "◆ agent search-verifier" in rendered

    @pytest.mark.asyncio
    async def test_burst_closes_before_streaming(self):
        """send_streaming opens a rich Live — an open burst spinner must freeze first
        (rich allows one live display per console)."""
        out = StringIO()
        ch = make_terminal_channel(out=out)  # force_terminal: exercises the real Status path

        async def tokens() -> AsyncIterator[str]:
            yield "answer"

        await ch.send_tool_call("web_search", {"query": "q"})
        await ch.send_tool_result("web_search", "ok", is_error=False)
        result = await ch.send_streaming(tokens())
        assert result == "answer"
        rendered = out.getvalue()
        assert "· 1/1" in rendered
        assert ch._burst is None

    @pytest.mark.asyncio
    async def test_burst_flushes_on_stop(self):
        ch, out = self._pipe_channel()
        await ch.send_tool_call("web_search", {"query": "q"})
        await ch.stop()
        assert "◆ web_search — the open web, a fresh window each · 0/1" in out.getvalue()
        assert ch._burst is None

    @pytest.mark.asyncio
    async def test_thinking_indicator_suppressed_during_burst(self):
        """The burst spinner IS the live indicator — a Heartbeat mid-burst must not
        start a second rich live display."""
        from localharness.core.events import Heartbeat
        ch = make_terminal_channel(out=StringIO())
        hb = Heartbeat(agent_id="a", session_id="s", iteration=1, context_utilization_pct=10.0)
        await ch.send_tool_call("web_search", {"query": "q"})
        await ch.on_heartbeat(hb)
        assert ch._thinking is None
        assert ch._context_pct == 10.0  # meter tracking still updates
        await ch.send_message("bye")    # closes burst, stops spinner thread
        assert ch._burst is None

    @pytest.mark.asyncio
    async def test_live_counter_ticks_calls_and_done(self):
        ch, _ = self._pipe_channel()
        await ch.send_tool_call("web_search", {"query": "a"})
        await ch.send_tool_result("web_search", "ok", is_error=False)
        await ch.send_tool_call("web_fetch", {"url": "b"})
        b = ch._burst
        assert b is not None and (b.done, b.calls) == (1, 2)
        assert "· 1/2" in ch._burst_text(b, final=False)
        await ch.stop()

    @pytest.mark.asyncio
    async def test_orphan_family_result_prints_legacy_line(self):
        """A family result with no open burst (shouldn't happen in the sequential
        loop) must never be dropped — falls back to the per-call line."""
        ch, out = self._pipe_channel()
        await ch.send_tool_result("web_fetch", "late result", is_error=False)
        assert "✓ web_fetch" in out.getvalue()

    @pytest.mark.asyncio
    async def test_non_family_tools_keep_per_call_lines(self):
        ch, out = self._pipe_channel()
        await ch.send_tool_call("glob", {"pattern": "*.py"})
        await ch.send_tool_result("glob", "a.py", is_error=False)
        rendered = out.getvalue()
        assert "◆ glob *.py" in rendered
        assert "✓ glob (1 lines)" in rendered


class TestMarkupSafetyAndCompactCalls:
    """FIX A/B/C — regression for the dogfood report where a `write` tool call dumped a
    237-line file body into the chat view AND markup-injection silently ate the code:
    lowercase-leading `[..]` spans are parsed as Rich style tags, so `[j]`/`[i]` vanished
    and `projs = [calc(p) for p in M_DATA]` rendered as `projs =`.

    A: every dynamic value interpolated into a markup-enabled print is escaped, so user/
       model content can never be interpreted as a style tag (our own `[tool.call]` etc.
       stay literal).
    B: content-bearing tool calls collapse to one summary line — write/edit show
       `<tool> <path> (<n> lines)`, every other tool caps its arg preview to one line.
    C: the result label never claims a line count it can't back up."""

    @staticmethod
    def _pipe():
        # non-TTY: plain text, no ANSI, no live frames — exact substring assertions
        out = StringIO()
        return make_terminal_channel(out=out, force_terminal=False), out

    @pytest.mark.asyncio
    async def test_write_call_collapses_to_one_summary_line_body_absent(self):
        # content BEFORE path — the arg ordering that made _get_key_arg dump the body
        body = "\n".join(
            ["import data", "projs = [calc(p) for p in M_DATA]"]
            + [f"row[{i}] = x[j] + y[i]  # line {i}" for i in range(198)]
        )
        assert body.count("\n") == 199  # exactly 200 lines
        ch, out = self._pipe()
        await ch.send_tool_call("write", {"content": body, "path": "docs/sim.py"})
        rendered = out.getvalue()
        assert rendered.strip().count("\n") == 0                # ONE line, not 200
        assert "◆ write docs/sim.py (200 lines)" in rendered    # path + honest size hint
        assert "calc(p) for p in M_DATA" not in rendered        # body never dumped
        assert "# line 150" not in rendered

    @pytest.mark.asyncio
    async def test_dynamic_brackets_render_verbatim_not_swallowed(self):
        # the exact corruption from the report: lowercase-leading [..] are parsed as
        # style tags and eaten unless escaped. A non-write tool's arg IS rendered.
        cmd = "projs = [calc(p) for p in M_DATA]; a = x[j]; b = y[i]"
        ch, out = self._pipe()
        await ch.send_tool_call("bash", {"command": cmd})
        rendered = out.getvalue()
        assert "[calc(p) for p in M_DATA]" in rendered   # survived verbatim
        assert "x[j]" in rendered
        assert "y[i]" in rendered

    @pytest.mark.asyncio
    async def test_long_non_write_arg_truncated_to_one_line(self):
        cmd = "echo " + "A" * 200
        ch, out = self._pipe()
        await ch.send_tool_call("bash", {"command": cmd})
        rendered = out.getvalue()
        assert rendered.strip().count("\n") == 0     # one line
        assert "…" in rendered                  # ellipsis marks the truncation
        assert "A" * 200 not in rendered             # full arg body not shown
        assert "echo AAAA" in rendered               # prefix preserved

    @pytest.mark.asyncio
    async def test_write_result_label_omits_misleading_count(self):
        ch, out = self._pipe()
        # write returns a status line ("Written N bytes"), NOT the file body — old code
        # printed "(1 lines)", implying one line was written.
        await ch.send_tool_result(
            "write", "Written 5678 bytes to /abs/docs/sim.py", is_error=False,
        )
        rendered = out.getvalue()
        assert "✓ write" in rendered      # ✓ write
        assert "lines)" not in rendered        # never a fabricated line count

    @pytest.mark.asyncio
    async def test_model_text_markup_prints_literally(self):
        ch, out = self._pipe()
        await ch.send_message("Result: [bold]not-a-style[/bold] and score[42]")
        rendered = out.getvalue()
        assert "[bold]not-a-style[/bold]" in rendered   # printed, not applied as a style
        assert "score[42]" in rendered


# ---------------------------------------------------------------------------
# #73: agent delegation-outcome receipts. Every `agent` completion prints a
# truthful, harness-rendered line whose status comes from the TOOL RESULT
# (is_error) — so the model can't pass its own prose off as the subagent's
# ("the joke-writer gave me that one!") when the delegation actually failed.
# Driven through the real on_action/on_observation dispatch seam (base.py).
# ---------------------------------------------------------------------------


def _agent_action(delegate: str, task: str = "Write three puns about databases."):
    from localharness.core.events import Action
    return Action(
        agent_id="orchestrator", session_id="s1", action_type="tool_call",
        tool_call_id="tc1", tool_name="agent",
        tool_params={"agent_id": delegate, "task": task},
    )


def _agent_observation(output: str, error: str | None):
    from localharness.core.events import Observation
    return Observation(
        agent_id="orchestrator", session_id="s1", observation_type="tool_result",
        tool_call_id="tc1", tool_name="agent", output=output, error=error,
    )


# loop.py sets BOTH Observation fields to '[tool error] <msg>' on a tool failure.
_FAIL_MSG = "[tool error] Agent 'joke-writer' failed: boom"


class TestAgentDelegationReceipt:
    def _pipe(self):
        out = StringIO()
        return make_terminal_channel(out=out, force_terminal=False), out

    @pytest.mark.asyncio
    async def test_success_receipt_names_delegate_and_says_completed(self):
        ch, out = self._pipe()
        await ch.on_action(_agent_action("joke-writer"))
        await ch.on_observation(_agent_observation(output="Three puns: ...", error=None))
        rendered = out.getvalue()
        assert "◆ agent joke-writer — completed" in rendered  # ◆ … — completed
        assert "FAILED" not in rendered

    @pytest.mark.asyncio
    async def test_failure_receipt_says_failed_with_first_line_detail(self):
        ch, out = self._pipe()
        await ch.on_action(_agent_action("joke-writer"))
        await ch.on_observation(_agent_observation(output=_FAIL_MSG, error=_FAIL_MSG))
        rendered = out.getvalue()
        assert "◆ agent joke-writer — FAILED:" in rendered
        assert "Agent 'joke-writer' failed: boom" in rendered  # honest detail from the result
        assert "[tool error]" not in rendered  # loop-internal prefix stripped from the receipt
        assert "completed" not in rendered

    @pytest.mark.asyncio
    async def test_failure_is_error_styled_success_is_not(self):
        # TTY mode: a dropped delegation must be visually unmissable — the failure line
        # carries the error style (bold red, \x1b[1;31m); success does not.
        so, fo = StringIO(), StringIO()
        cs = make_terminal_channel(out=so, force_terminal=True)
        cf = make_terminal_channel(out=fo, force_terminal=True)
        await cs.on_action(_agent_action("joke-writer"))
        await cs.on_observation(_agent_observation(output="done", error=None))
        await cf.on_action(_agent_action("joke-writer"))
        await cf.on_observation(_agent_observation(output=_FAIL_MSG, error=_FAIL_MSG))
        assert "\x1b[1;31m" in fo.getvalue()      # bold red on failure
        assert "\x1b[1;31m" not in so.getvalue()  # success is not error-styled

    @pytest.mark.asyncio
    async def test_status_comes_from_result_flag_not_prose(self):
        # The honesty spine (#73): status is a function of is_error, never of the text.
        # A SUCCESS whose text is full of doom still reads 'completed'.
        ch, out = self._pipe()
        await ch.on_action(_agent_action("joke-writer"))
        await ch.on_observation(_agent_observation(
            output="could not find one, this failed to amuse anybody", error=None))
        rendered = out.getvalue()
        assert "— completed" in rendered
        assert "FAILED" not in rendered
