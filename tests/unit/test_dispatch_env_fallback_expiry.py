"""At 0.17.0 this test fails until the Discord env fallback is deleted (DISP-02). Do not loosen it; delete the fallback.

Below 0.17.0 it proves the opposite — the fallback IS present and the scanner catches every marker —
so it is never vacuous. The version is `localharness.__version__` (the source of truth), never dist
metadata (stale on editable installs). This file lives under tests/, outside the scanned root.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from packaging.version import Version

import localharness
import localharness.dispatch.config as dispatch_config

SRC = Path(localharness.__file__).resolve().parent
MARKERS = ("LOCALHARNESS_DISCORD_", "DISCORD_BOT_TOKEN", "def env_fallback")
"""Claude Code's `.env` file source was removed ahead of 0.17.0 (it belonged to another program),
so it is no marker: the refusal line that names the file is a message, not a fallback."""
EXPIRY = Version("0.17.0")


def expired(version: str) -> bool:
    return Version(version) >= EXPIRY


def residue(src_root: Path) -> list[str]:
    """`<relative path>: <marker>` for every fallback marker left in a .py under src_root."""
    return [f"{p.relative_to(src_root).as_posix()}: {m}"
            for p in sorted(src_root.rglob("*.py")) for m in MARKERS if m in p.read_text()]


def test_expired_boundary():
    assert expired("0.17.0") and expired("0.18.1")
    assert not expired("0.16.9") and not expired("0.9.30")


@pytest.mark.parametrize("marker", MARKERS)
def test_scanner_catches_each_marker(tmp_path, marker):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "clean.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "dirty.py").write_text(f"y = {marker!r}\n")
    assert residue(tmp_path) == [f"pkg/dirty.py: {marker}"]


@pytest.mark.skipif(expired(localharness.__version__), reason="0.17.0+: the fallback must be gone")
def test_fallback_present_before_expiry():
    found = residue(SRC)
    assert "dispatch/config.py: def env_fallback" in found, found
    # Every marker must match live code today, so a marker that has drifted (a typo, a renamed
    # path) cannot leave a legacy source the 0.17.0 scan would no longer see.
    assert {m for m in MARKERS if any(f.endswith(f": {m}") for f in found)} == set(MARKERS), found
    assert callable(getattr(dispatch_config, "env_fallback", None))


def test_no_fallback_from_0_17_0():
    if not expired(localharness.__version__):
        assert residue(SRC), "the fallback must be live below 0.17.0"
        return
    assert residue(SRC) == [], "0.17.0: delete the Discord env fallback (DISP-02) — do not loosen this test"
