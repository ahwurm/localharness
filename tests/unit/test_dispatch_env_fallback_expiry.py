"""The Discord env fallback was deleted in 0.17.1 (DISP-02), as 0.16.0 announced for 0.17.0. This
test fails if any of its markers comes back under src/: the pre-settings variables are not read.
The scanner is checked against a planted marker, so the guard is never vacuous. This file lives
under tests/, outside the scanned root.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import localharness

SRC = Path(localharness.__file__).resolve().parent
MARKERS = ("LOCALHARNESS_DISCORD_", "DISCORD_BOT_TOKEN", "def env_fallback")
"""Claude Code's `.env` file source was removed ahead of 0.17.0 (it belonged to another program),
so it is no marker: the refusal line that names the file is a message, not a fallback."""


def residue(src_root: Path) -> list[str]:
    """`<relative path>: <marker>` for every fallback marker left in a .py under src_root."""
    return [f"{p.relative_to(src_root).as_posix()}: {m}"
            for p in sorted(src_root.rglob("*.py")) for m in MARKERS if m in p.read_text()]


@pytest.mark.parametrize("marker", MARKERS)
def test_scanner_catches_each_marker(tmp_path, marker):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "clean.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "dirty.py").write_text(f"y = {marker!r}\n")
    assert residue(tmp_path) == [f"pkg/dirty.py: {marker}"]


def test_no_fallback_marker_under_src():
    assert residue(SRC) == [], "the Discord env fallback was deleted in 0.17.1 — do not bring it back"
