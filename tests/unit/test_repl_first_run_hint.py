"""#49: the first-run '/help' hint must reach the user in an interactive terminal.

Root cause: the hint is emitted as ordinary scrollback by the startup banner
(ui.startup_banner), printed just above the prompt_toolkit input box. That box owns/repaints
the bottom region of a real TTY, so the hint is not reliably retained interactively — while
in piped mode (no interactive box) the banner text is captured fine. The robust fix renders
the hint inside the input bubble itself (the `_build_input_app(hint=...)` slot, previously
always ""), so it is drawn WITH the first prompt and cannot be clobbered. In piped mode the
banner keeps carrying the hint (unchanged).
"""
from __future__ import annotations

from io import StringIO

import pytest


# --- ui.startup_banner: the hint is now gated by an explicit show_hint (tty relocates it) ---

def _render_banner(is_returning: bool, show_hint: bool) -> str:
    from rich.console import Console
    from localharness.cli.ui import startup_banner
    out = StringIO()
    Console(file=out, force_terminal=False, width=120).print(
        startup_banner("qwen3-coder", is_returning, show_hint=show_hint)
    )
    return out.getvalue()


def test_banner_shows_hint_when_show_hint_true_first_run():
    # Piped path (show_hint defaults True): behavior unchanged — the hint is in the banner.
    assert "Describe a task" in _render_banner(is_returning=False, show_hint=True)


def test_banner_omits_hint_when_show_hint_false():
    # TTY path: the hint moves to the input bubble, so the banner must NOT duplicate it.
    assert "Describe a task" not in _render_banner(is_returning=False, show_hint=False)


def test_banner_never_shows_first_run_hint_when_returning():
    assert "Describe a task" not in _render_banner(is_returning=True, show_hint=True)


# --- start_cmd: the exact first-prompt hint text, incl. the returning-session reminder ---

def test_first_prompt_hint_text_first_run_and_returning():
    from localharness.cli.start_cmd import _first_prompt_hint
    assert _first_prompt_hint(is_returning=False) == "Describe a task, or /help for commands."
    # Returning-session polish: a short reminder still reinforces /help every session.
    assert _first_prompt_hint(is_returning=True) == "/help for commands."


# --- terminal.read_input: the first bubble carries the hint, later prompts do not ---

@pytest.mark.asyncio
async def test_read_input_renders_first_prompt_hint_in_the_bubble_once(monkeypatch, tmp_path):
    from localharness.channels import terminal as term_mod
    from localharness.channels.terminal import TerminalChannel, TERMINAL_THEME
    from localharness.core.bus import EventBus
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {}, history_file=str(tmp_path / ".hist"))
    ch._console = Console(file=StringIO(), force_terminal=True, width=120, theme=TERMINAL_THEME)
    await ch.start()

    # start_cmd sets this for interactive sessions (full hint first-run / short if returning).
    ch.first_prompt_hint = "Describe a task, or /help for commands."

    seen_hints: list[str] = []

    class _FakeApp:
        async def run_async(self):
            return "typed line"

    def _fake_build(history, prompt, hint, context_pct=None, model_names_fn=None):
        seen_hints.append(hint)
        return _FakeApp()

    monkeypatch.setattr(term_mod, "_build_input_app", _fake_build)

    first = await ch.read_input()
    second = await ch.read_input()
    await ch.stop()

    assert first == "typed line" and second == "typed line"
    assert seen_hints[0] == "Describe a task, or /help for commands."  # first prompt carries it
    assert seen_hints[1] == ""  # subsequent prompts don't repeat it


def test_terminal_channel_exposes_first_prompt_hint_default_empty():
    from localharness.channels.terminal import TerminalChannel
    from localharness.core.bus import EventBus
    ch = TerminalChannel(EventBus(), {})
    assert ch.first_prompt_hint == ""  # nothing shown until start_cmd opts in
