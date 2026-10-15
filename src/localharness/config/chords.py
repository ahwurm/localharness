"""The terminal box's chord vocabulary, spelled once (spec 21 §5, §7).

Four names a person may put in `terminal.keys.nudge` and `terminal.keys.newline`. It is pure: no
prompt_toolkit, no I/O, and nothing imported from the package, because `config/models.py` imports
`ChordName` from here and an import back would cycle. The wire sequences and key bindings that
realise each name live in `channels/keys.py`.
"""
from __future__ import annotations

from typing import Literal, get_args

__all__ = ["CHORD_NAMES", "LABELS", "ChordName"]

ChordName = Literal["alt-enter", "shift-enter", "ctrl-g", "ctrl-o"]

CHORD_NAMES: tuple[str, ...] = get_args(ChordName)

LABELS: dict[str, str] = {
    "alt-enter": "alt+enter",
    "shift-enter": "shift+enter",
    "ctrl-g": "ctrl+g",
    "ctrl-o": "ctrl+o",
}
"""How each chord is written in the legend, `/help` and the README."""
