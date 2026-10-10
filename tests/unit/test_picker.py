"""The one-line arrow-key picker (cli/picker.py, spec 16 section 7.1, D-04).

Real keystrokes through the real key bindings (create_pipe_input + DummyOutput), the idiom the
input-box suite uses. The terminal channel's `run_picker` tests live at the bottom of this file.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localharness.cli import picker as picker_mod
from localharness.cli.picker import PICKER_STYLE, build_picker, pick

LEVELS = ("off", "low", "high", "max")


async def _keys(keys: str, options=LEVELS, current="high"):
    with create_pipe_input() as inp:
        inp.send_text(keys)
        app = build_picker(options, current, input=inp, output=DummyOutput())
        return await asyncio.wait_for(app.run_async(), timeout=10.0)


@pytest.mark.parametrize(
    "keys,expected",
    [
        ("\r", "high"),                    # Enter takes the highlighted (starting) option
        ("\x1b[C\r", "max"),               # Right
        ("\x1b[C\x1b[C\r", "off"),         # Right past the last wraps to the first
        ("\x1b[D\r", "low"),               # Left
        ("\x1b[D\x1b[D\x1b[D\r", "max"),   # Left past the first wraps to the last
        ("\t\r", "max"),                   # Tab moves forward
        ("\x1b[Z\r", "low"),               # Shift-Tab moves back
        ("\x1b[C\x1b[D\r", "high"),        # forward then back is a no-op
    ],
)
async def test_keys_move_the_highlight_and_enter_returns_it(keys, expected):
    assert await _keys(keys) == expected


async def test_escape_returns_none():
    assert await _keys("\x1b") is None


async def test_ctrl_c_returns_none():
    assert await _keys("\x03") is None


async def test_escape_after_moving_still_returns_none():
    assert await _keys("\x1b[C\x1b") is None


@pytest.mark.parametrize("current", ["nope", None, ""])
async def test_an_unknown_current_starts_on_the_first_option(current):
    assert await _keys("\r", current=current) == "off"


async def test_current_in_the_options_starts_there():
    assert await _keys("\r", current="low") == "low"


def test_empty_options_raise_before_any_application_is_built(monkeypatch):
    built = []
    monkeypatch.setattr(picker_mod, "Application", lambda *a, **k: built.append(1))
    with pytest.raises(ValueError, match="picker needs at least one option"):
        build_picker([], "x")
    assert built == []


async def test_pick_is_the_async_wrapper():
    with create_pipe_input() as inp:
        inp.send_text("\x1b[C\r")
        got = await asyncio.wait_for(
            pick(LEVELS, "high", input=inp, output=DummyOutput()), timeout=10.0
        )
    assert got == "max"


def test_the_highlighted_option_is_selected_and_the_rest_are_hints():
    app = build_picker(LEVELS, "high", output=DummyOutput())
    frags = app._lh_picker_fragments()
    assert [(cls, text.strip()) for cls, text in frags] == [
        ("class:hint", "off"),
        ("class:hint", "low"),
        ("class:selected", "high"),
        ("class:hint", "max"),
    ]
    assert app._lh_picker_index() == 2


async def test_the_highlight_follows_the_keys_in_the_rendered_fragments():
    with create_pipe_input() as inp:
        inp.send_text("\x1b[C\r")
        app = build_picker(LEVELS, "high", input=inp, output=DummyOutput())
        await asyncio.wait_for(app.run_async(), timeout=10.0)
    selected = [t.strip() for cls, t in app._lh_picker_fragments() if cls == "class:selected"]
    assert selected == ["max"]


async def test_non_ascii_options_render_and_come_back_unchanged():
    options = ("off", "máx ✓", "高")
    app = build_picker(options, "máx ✓", output=DummyOutput())
    assert "máx ✓" in "".join(t for _, t in app._lh_picker_fragments())
    assert await _keys("\r", options=options, current="máx ✓") == "máx ✓"
    assert await _keys("\x1b[C\r", options=options, current="máx ✓") == "高"


def test_the_picker_style_has_a_selected_rule():
    assert any(cls == "selected" for cls, _ in PICKER_STYLE.style_rules)


def test_the_widget_imports_nothing_from_the_repl_or_a_channel():
    tree = ast.parse(Path(picker_mod.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert all(not m.startswith("localharness.channels") for m in imported), imported
    assert "localharness.cli.repl" not in imported, imported
    assert any(m.startswith("prompt_toolkit") for m in imported)


# ------------------------------------------------------------ TerminalChannel.run_picker (D-04)

from io import StringIO  # noqa: E402

from prompt_toolkit.application import create_app_session  # noqa: E402
from prompt_toolkit.history import InMemoryHistory  # noqa: E402

from localharness.channels import terminal as terminal_mod  # noqa: E402
from localharness.channels.terminal import (  # noqa: E402
    ERASE_APPLICATIONS_WHEN_DONE,
    INPUT_STYLE,
    TERMINAL_THEME,
    TerminalChannel,
)
from localharness.core.bus import EventBus  # noqa: E402


def _channel() -> TerminalChannel:
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(
        file=StringIO(), force_terminal=True, width=120, theme=TERMINAL_THEME, highlight=False
    )
    return ch


@pytest.fixture
def can_draw(monkeypatch):
    monkeypatch.setattr(terminal_mod, "cannot_ask_reason", lambda: None)


def _fake_box(ch: TerminalChannel, calls: list[str]) -> None:
    ch._box_active = True
    ch._box_restart_args = ("queue", "interrupt")

    async def _stop():
        calls.append("stop")
        ch._box_active = False

    async def _start(*args):
        calls.append(f"start{args}")
        ch._box_active = True

    ch.stop_input_box = _stop
    ch.start_input_box = _start


async def test_run_picker_suspends_the_box_then_restarts_it(monkeypatch, can_draw):
    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)

    async def _pick(options, current, **kw):
        calls.append(f"pick{tuple(options)}{current}")
        return "off"

    monkeypatch.setattr("localharness.cli.picker.pick", _pick)
    assert await ch.run_picker(("off", "low"), "low") == "off"
    assert calls == ["stop", "pick('off', 'low')low", "start('queue', 'interrupt')"]


async def test_run_picker_restarts_the_box_when_the_picker_raises(monkeypatch, can_draw):
    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)

    async def _pick(options, current, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("localharness.cli.picker.pick", _pick)
    with pytest.raises(RuntimeError, match="boom"):
        await ch.run_picker(("off", "low"), "low")
    assert calls == ["stop", "start('queue', 'interrupt')"]


async def test_run_picker_never_overlaps_the_box_application(monkeypatch, can_draw):
    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)
    seen: list[bool] = []

    async def _pick(options, current, **kw):
        seen.append(ch._box_active)
        return None

    monkeypatch.setattr("localharness.cli.picker.pick", _pick)
    assert await ch.run_picker(("off",), None) is None
    assert seen == [False]


async def test_run_picker_leaves_an_inactive_box_alone(monkeypatch, can_draw):
    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)
    ch._box_active = False

    async def _pick(options, current, **kw):
        return "low"

    monkeypatch.setattr("localharness.cli.picker.pick", _pick)
    assert await ch.run_picker(("off", "low"), "off") == "low"
    assert calls == []


async def test_run_picker_is_a_no_op_without_a_terminal(monkeypatch):
    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)
    monkeypatch.setattr(terminal_mod, "cannot_ask_reason", lambda: "no tty")

    async def _pick(*a, **kw):
        calls.append("pick")
        return "off"

    monkeypatch.setattr("localharness.cli.picker.pick", _pick)
    assert await ch.run_picker(("off", "low"), "low") is None
    assert calls == []


async def test_run_picker_draws_on_the_tty_when_stdout_is_redirected(monkeypatch, can_draw):
    ch = _channel()
    monkeypatch.setattr(ch, "can_run_input_box", lambda: False)
    got: dict = {}

    async def _pick(options, current, **kw):
        got.update(kw)
        return "low"

    class _Out:
        pass

    sentinel = _Out()
    monkeypatch.setattr(
        "prompt_toolkit.output.defaults.create_output", lambda **kw: sentinel
    )
    monkeypatch.setattr("localharness.cli.picker.pick", _pick)
    assert await ch.run_picker(("off", "low"), "off") == "low"
    assert got["output"] is sentinel


async def test_run_picker_passes_a_style_that_keeps_the_selected_rule(monkeypatch, can_draw):
    ch = _channel()
    got: dict = {}

    async def _pick(options, current, **kw):
        got.update(kw)
        return None

    monkeypatch.setattr("localharness.cli.picker.pick", _pick)
    await ch.run_picker(("off", "low"), "off")
    assert got["erase_when_done"] is ERASE_APPLICATIONS_WHEN_DONE
    rules = got["style"].style_rules
    assert any(cls == "selected" for cls, _ in rules)
    # INPUT_STYLE goes last, so the terminal's own hint/frame colours win over the picker's blanks.
    hint_rules = [fmt for cls, fmt in rules if cls == "hint"]
    assert hint_rules[-1] == dict(INPUT_STYLE.style_rules)["hint"]
    frame_rules = [fmt for cls, fmt in rules if cls == "frame"]
    assert frame_rules[-1] == dict(INPUT_STYLE.style_rules)["frame"]


async def test_run_picker_with_a_real_box_takes_real_keystrokes(can_draw):
    ch = _channel()
    ch._history = InMemoryHistory()
    queue: asyncio.Queue = asyncio.Queue()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        await ch.start_input_box(queue, on_interrupt=lambda: None)
        try:
            inp.send_text("\x1b[C\r")
            got = await asyncio.wait_for(ch.run_picker(("off", "low", "high", "max"), "high"), 10.0)
            assert got == "max"
            assert ch._box_active is True, "the input box was not restarted"
        finally:
            await ch.stop_input_box()
