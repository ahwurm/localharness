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

from prompt_toolkit.input import vt100_parser
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.keys import Keys

logger = logging.getLogger(__name__)

__all__ = [
    "KEYS",
    "SHIFT_ENTER_KEY",
    "SHIFT_ENTER_SEQUENCES",
    "install_wire_sequences",
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
