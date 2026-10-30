"""The one immutable tuple of text limits every surface shares."""
from typing import NamedTuple


class _Caps(NamedTuple):
    LINE: int
    FINDING: int
    BLOCK: int
    DOC: int
    TOP: int
    QUESTIONS: int


Caps = _Caps(LINE=80, FINDING=240, BLOCK=4_000, DOC=8_000, TOP=8, QUESTIONS=6)
