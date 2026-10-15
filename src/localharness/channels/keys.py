"""The terminal box's chords, in code (spec 21 §7).

`config/chords.py` spells the four names a person may configure; this module realises them for
prompt_toolkit: the `KEYS` table the box binds from, the one borrowed key Shift+Enter rides on, the
guarded install of its two wire sequences, and the renderers the box legend and `/help` use. It
imports the vocabulary from `config/chords.py` and prompt_toolkit, nothing from the REPL or the
channel.

It reads the environment it is handed and never queries or reconfigures the terminal: no request
goes out on the wire to name the terminal, and no keyboard mode is pushed. Input devices are not
read either; spec 21 §12 (DEC21) says why.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from prompt_toolkit.input import vt100_parser
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.keys import Keys

from localharness.config.chords import LABELS

if TYPE_CHECKING:
    from localharness.config.models import TerminalKeysConfig

logger = logging.getLogger(__name__)

__all__ = [
    "KEYS",
    "SHIFT_ENTER_KEY",
    "SHIFT_ENTER_SEQUENCES",
    "detect_terminal",
    "help_block",
    "install_wire_sequences",
    "nudge_legend",
]

# prompt_toolkit offers no way to add a key, and no table entry produces Shift+Enter (its xterm
# form is remapped to plain Enter as a placeholder). The borrowed member is the one nothing can
# produce: ControlF3's xterm sequence collides with the cursor-position report, so the table
# comments it out and no terminal can deliver it. Binding Shift+Enter to it, and pointing the two
# wire forms at it, cannot shadow a key a person has. This is the only place the member is named.
SHIFT_ENTER_KEY = Keys.ControlF3

SHIFT_ENTER_SEQUENCES: tuple[str, str] = (
    "\x1b[13;2u",  # the kitty keyboard protocol form
    "\x1b[27;2;13~",  # the xterm modifyOtherKeys form
)
"""The two wire encodings of Shift+Enter that `install_wire_sequences` maps to `SHIFT_ENTER_KEY`."""

KEYS: dict[str, tuple[Keys | str, ...]] = {
    "alt-enter": ("escape", "enter"),
    "shift-enter": (SHIFT_ENTER_KEY,),
    "ctrl-g": ("c-g",),
    "ctrl-o": ("c-o",),
}
"""Each chord name as the prompt_toolkit key tuple a binding takes. The one place the escape/enter
pair for Alt+Enter is spelled under `channels/`."""


def install_wire_sequences() -> bool:
    """Teach prompt_toolkit's input parser the two Shift+Enter encodings; True when installed.

    Writes exactly the two constants in `SHIFT_ENTER_SEQUENCES` into `ANSI_SEQUENCES` (replacing the
    xterm entry's placeholder remap to Enter; the two Ctrl+Enter entries are left as Enter) and
    clears the parser's prefix cache, which memoises answers from the old table. It is idempotent.

    The table is process-wide, so later prompts in the process (the picker included) see the install
    too; the app builders call this only when `shift-enter` is in one of the two lists. The cache is
    private to prompt_toolkit: when a release renames it nothing is installed, one info line says so
    and the result is False, while `\\` then Enter and the other three chords keep working.
    """
    cache = getattr(vt100_parser, "_IS_PREFIX_OF_LONGER_MATCH_CACHE", None)
    if cache is None:
        logger.info(
            "shift+enter is unavailable: this prompt_toolkit has no prefix cache to reset "
            "(backslash then enter, alt-enter, ctrl-g and ctrl-o still work)"
        )
        return False
    for sequence in SHIFT_ENTER_SEQUENCES:
        ANSI_SEQUENCES[sequence] = SHIFT_ENTER_KEY
    cache.clear()
    return True


def nudge_legend(names: Sequence[str]) -> str:
    """The working-state legend's nudge clause (spec 21 §6.6); the caller appends ` · ctrl+c stop`.

    The spellings are `!text` and the configured chords' labels, in the configured order:
    `!text or alt+enter nudge`, `!text, shift+enter or ctrl+o nudge`, `!text nudge` with none.
    """
    labels = [LABELS[name] for name in names]
    if not labels:
        return "!text nudge"
    if len(labels) == 1:
        return f"!text or {labels[0]} nudge"
    return f"!text, {', '.join(labels[:-1])} or {labels[-1]} nudge"


# Each terminal's display name and its one line that makes Shift+Enter send `ESC [ 13 ; 2 u` (spec 21
# §8, without the markdown). The `\x1b` and `\u001b` in these are literal backslash text the person
# pastes into the terminal's own config, never a control character: they are printed by `/help`.
_SETUP: dict[str, str] = {
    "kitty": r"kitty.conf: map shift+enter send_text all \x1b[13;2u",
    "Ghostty": r"config: keybind = shift+enter=text:\x1b[13;2u",
    "WezTerm": (
        r'keys: { key = "Enter", mods = "SHIFT", '
        r'action = wezterm.action.SendString("\x1b[13;2u") }'
    ),
    "Alacritty": r'[[keyboard.bindings]] key = "Enter", mods = "Shift", chars = "\u001b[13;2u"',
    "foot": r"[text-bindings]: \x1b[13;2u=Shift+Return",
    "iTerm2": "Profiles → Keys → add Shift+Enter → Send Escape Sequence → [13;2u",
    "VS Code": (
        r'keybindings.json: { "key": "shift+enter", '
        r'"command": "workbench.action.terminal.sendSequence", '
        r'"args": { "text": "\u001b[13;2u" }, "when": "terminalFocus" }'
    ),
    "Windows Terminal": (
        r'actions: { "command": { "action": "sendInput", "input": "\u001b[13;2u" }, '
        r'"keys": "shift+enter" }'
    ),
}


def detect_terminal(env: Mapping[str, str]) -> tuple[str, str] | None:
    """The terminal's name and its one setup line, from the environment only (spec 21 §7).

    Nothing is queried on the wire. Under tmux the answer is None first, because the outer
    terminal's variables are inherited into it and its line would be wrong there. A variable
    present with an empty value counts as unset. Anything unnamed (GNOME Terminal, Konsole, xterm,
    a bare `TERM` over ssh) is None; the worst a wrong value can do is pick a different help line.
    """

    def has(name: str) -> bool:
        return bool(env.get(name))

    term = env.get("TERM") or ""
    program = env.get("TERM_PROGRAM") or ""
    if has("TMUX") or program == "tmux":
        return None
    if has("KITTY_WINDOW_ID") or term == "xterm-kitty":
        name = "kitty"
    elif program == "ghostty" or term == "xterm-ghostty":
        name = "Ghostty"
    elif program == "WezTerm":
        name = "WezTerm"
    elif term == "alacritty" or has("ALACRITTY_WINDOW_ID"):
        name = "Alacritty"
    elif term in ("foot", "foot-extra"):
        name = "foot"
    elif program == "iTerm.app":
        name = "iTerm2"
    elif program == "vscode":
        name = "VS Code"
    elif has("WT_SESSION"):
        name = "Windows Terminal"
    else:
        return None
    return name, _SETUP[name]


_EVERY_TERMINAL = ("ctrl-o", "ctrl-g", "alt-enter")
"""Chords every terminal sends: one of them as a newline chord is a working line break."""

_ROUTE_CHORDS = ("ctrl-o", "ctrl-g")
"""The two chords the route may name, in the order it tries them; Alt+Enter is never the route."""

_MIN_LABEL_WIDTH = 19
"""The width of the spec's default rows: `\\ then enter` and `!text, alt+enter` padded to it."""


def help_block(
    keys: TerminalKeysConfig, *, terminal: tuple[str, str] | None, shift_enter_seen: bool
) -> str:
    """The `Keys:` block `/help` prints while the persistent box is live (spec 21 §6.7).

    Two rows, three lines, no trailing newline. Row 1 is the line break, in one of three forms:
    `shift-enter` is a newline chord and has arrived this session (`line break — shift+enter too`);
    it is a newline chord, has not arrived and the terminal is named (the honest sentence and that
    terminal's setup line); or otherwise `line break` plus, when it would help, one route that works
    on every terminal. The route names the first of ctrl-o, ctrl-g that neither list uses; it is
    omitted when both are taken, and when a chord every terminal sends is already a newline chord,
    because the person then has a working line break and the route would only replace it. Never a
    README pointer, never Alt+Enter as the route. Row 2 is the nudge.
    """
    newline = list(keys.newline)
    nudge = list(keys.nudge)
    left1 = "\\ then enter" + "".join(f", {LABELS[n]}" for n in newline if n != "shift-enter")
    left2 = "!text" + "".join(f", {LABELS[n]}" for n in nudge)

    if "shift-enter" in newline and shift_enter_seen:
        text1 = "line break — shift+enter too"
    elif "shift-enter" in newline and terminal is not None:
        name, setup = terminal
        text1 = f"line break. shift+enter submits until {name} is set up — {setup}"
    else:
        text1 = "line break"
        if not set(newline) & set(_EVERY_TERMINAL):
            route = next((n for n in _ROUTE_CHORDS if n not in nudge and n not in newline), None)
            if route is not None:
                text1 += (
                    f" (or give {LABELS[route]} to it: "
                    f"localharness components set terminal.keys.newline {route})"
                )
    text2 = "send the line into the running turn (a nudge)"

    width = max(_MIN_LABEL_WIDTH, len(left1) + 3, len(left2) + 3)
    return f"Keys:\n  {left1:<{width}}{text1}\n  {left2:<{width}}{text2}"
