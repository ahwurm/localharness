"""INIT-13 (Nyquist audit): the receipt row is printed after an interactive (terminal) `init` too,
not only by a scripted run; and no settings question is asked."""
from __future__ import annotations

from tests.unit.test_init_core_setup import (  # noqa: F401  (autouse fixture and helpers)
    _detect,
    _exited,
    _flat,
    _hermetic,
    _init,
    _silent,
    _terminal,
)


def test_a_terminal_init_prints_the_receipt_row_after_configured(tmp_path, monkeypatch):
    _detect(monkeypatch)
    _terminal(monkeypatch)
    _silent(monkeypatch)

    result = _init(tmp_path)

    _exited(result, 0)
    flat = _flat(result)
    row = ("The agent can edit files, run commands and remember across sessions — "
           "org.permissions.mode: read-only or memory.enabled: false turns either off.")
    assert row in flat, flat
    later = "Later: just run localharness start — it brings back your last choice."
    assert flat.index("LocalHarness configured at") < flat.index(row) < flat.index(later)
    assert "/model" not in flat[flat.index(later):].split("★")[0]
    assert "Run 'localharness start' to begin." not in flat
    assert "Keep the usual settings" not in flat
