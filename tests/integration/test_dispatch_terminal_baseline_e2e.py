"""The terminal session a default user sees, pinned with and without the dispatch extra (49 Wave 0).

Captured from the pre-move tree. One real offline start per case through the memory baseline's
`_start` (REAL `_start_async`, memory on, one turn), no Discord env and HOME a tmp dir:
- `absent`: the venv as it is (discord.py not installed);
- `present`: `extra_installed` patched True, as if `localharness[dispatch]` were installed.

Pinned per case in `tests/fixtures/dispatch_plugin/terminal_banner.txt`: the `Plugins:` line text
and the startup-warnings list. And in BOTH cases the first system prompt and the root tool names
must still equal the Phase 47 memory goldens (read, never written here).

The ONE allowed delta after the move: `DISPATCH_APPEND["present"]` is `", dispatch"` (49-06).
Anything else that moves is a finding, never a regenerate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.dispatch_support import isolate_discord_env
from tests.integration.test_memory_compat_baseline_e2e import (
    GOLDEN_PROMPT,
    GOLDEN_TOOLS,
    REGEN,
    _seed_store,
    _start,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "dispatch_plugin" / "terminal_banner.txt"
CASES = ("absent", "present")
DISPATCH_APPEND = {"absent": "", "present": ", dispatch"}  # dispatch (49) is bundled and on by default
AUTORESEARCH_APPEND = ", autoresearch"  # autoresearch (50) is bundled and on by default, no extra


def _plugins_text(printed: list[str]) -> str:
    lines = [p.split("Plugins: ")[1].split("[/]")[0] for p in printed if "Plugins:" in p]
    assert len(lines) == 1, printed
    return lines[0]


def _startup_warnings(printed: list[str]) -> list[str]:
    """The summary line's warnings group (start_cmd's `; ` joiner), [] when there is none."""
    line = next(p for p in printed if "startup)" in p)
    open_at = line.find("\\[")
    return [] if open_at == -1 else line[open_at + 2:].rsplit("]", 1)[0].split("; ")


def _golden_lines(case: str, line: str) -> dict[str, str]:
    """Each case owns its line; REGEN rewrites only this case's line, file kept in CASES order."""
    rows = {}
    if GOLDEN.exists():
        rows = dict(r.split("\t", 1) for r in GOLDEN.read_text(encoding="utf-8").splitlines() if r)
    if REGEN:
        rows[case] = line
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text("".join(f"{c}\t{rows[c]}\n" for c in CASES if c in rows), encoding="utf-8")
    return rows


@pytest.mark.parametrize("extra", CASES)
async def test_terminal_banner_prompt_and_tools(extra, tmp_path, monkeypatch):
    isolate_discord_env(monkeypatch, tmp_path)
    if extra == "present":
        from localharness.plugins import resolve
        monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: True)

    await _seed_store(tmp_path)
    out = await _start(tmp_path, monkeypatch, memory={"consolidation": {"enabled": False}})
    printed = out["printed"]
    assert not any("Dispatch mode" in p for p in printed), "a terminal start announced dispatch"

    plugins, warnings = _plugins_text(printed), _startup_warnings(printed)
    line = f"Plugins: {plugins}\twarnings={json.dumps(warnings, ensure_ascii=False)}"
    rows = _golden_lines(extra, line)
    want_plugins, want_warnings = rows[extra].split("\t")
    assert f"Plugins: {plugins}" == want_plugins + DISPATCH_APPEND[extra] + AUTORESEARCH_APPEND, "the Plugins: line drifted"
    assert f"warnings={json.dumps(warnings, ensure_ascii=False)}" == want_warnings, "startup warnings drifted"

    first = out["calls"][0]
    prompt = first["system"]["content"].replace(str(tmp_path), "<TMP>")
    assert prompt == GOLDEN_PROMPT.read_text(encoding="utf-8"), "first system prompt drifted (47 golden)"
    assert "\n".join(first["tools"]) + "\n" == GOLDEN_TOOLS.read_text(encoding="utf-8"), \
        "root tool names drifted (47 golden)"
