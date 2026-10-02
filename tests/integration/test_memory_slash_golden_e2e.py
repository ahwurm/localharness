"""`/memory` pinned byte-for-byte from the unconverted tree (Phase 48 Wave 1, captured at 7757e09).

These goldens MUST hold unedited after `/memory` moves into the memory plugin (48-03). A diff is a
finding to explain, never a golden to regenerate.

Drive: the REAL `_start_async` (only external boundaries stubbed, as in the 47 baselines) over a
seeded store; `OrchestratorREPL.run` is replaced by a driver that swaps `self._channel` for a
recorder and calls `self._handle_slash(cmd)` for a fixed command list. No turn runs, so nothing
reaches a model. The test touches only surfaces that survive the move: `_handle_slash` and the
channel's `send_message` / `send_renderable`.

Promote: this start has no `.localharness/` workspace, so `promote <id>` previews and
`promote <id> confirm` returns today's "needs a project layer" text — both pinned here. The
confirm/revert rules against a real global store are pinned by the unedited
tests/unit/test_memory_promote.py.

Normalisation (one function, `_norm`): `str(tmp_path)` -> `<TMP>`; `YYYY-MM-DD HH:MM` -> `<STAMP>`.
Ages stay literal (facts are seeded seconds before, so they read `0m`).
"""
from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console

from tests.integration.test_memory_compat_baseline_e2e import (
    AGENT,
    _FrozenNow,
    _golden,
    _no_session_start_pass,
    _write_root_agent,
)
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "memory_surfaces"
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")


class _Recorder:
    def __init__(self, orig, out: list):
        self._orig, self._out = orig, out

    async def send_message(self, text, metadata=None):
        self._out.append(("msg", (metadata or {}).get("style"), text))

    async def send_renderable(self, r):
        con = Console(record=True, width=100, color_system=None, force_terminal=False)
        con.print(r)
        self._out.append(("renderable", None, con.export_text()))

    def __getattr__(self, name):
        return getattr(self._orig, name)


async def _seed(tmp_path: Path) -> dict[str, int]:
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(agent_id=AGENT, division_id="default", org_id="default",
                        base_dir=str(tmp_path))
    await store.open()
    try:
        await store.store_fact("user-editor", "The user edits in Neovim with a tiling WM.",
                               tags=["prefs"], source="baseline")
        await store.store_fact("project-lang", "The project is Python 3.12 on uv.", source="baseline")
        await store.store_fact("deploy-target", "staging box", source="baseline")
        await store.store_fact("deploy-target", "prod box", source="baseline")
        return {k: (await store.get_fact(k)).id for k in ("user-editor", "project-lang", "deploy-target")}
    finally:
        await store.close()


def _norm(text: str, tmp_path: Path) -> str:
    return _STAMP.sub("<STAMP>", text.replace(str(tmp_path), "<TMP>"))


async def _drive(tmp_path: Path, monkeypatch, cmds: list[str], *, memory_off: bool) -> str:
    _stub_start_boundaries(tmp_path, monkeypatch)
    _offline_provider(tmp_path)
    if memory_off:
        with (tmp_path / "config.yaml").open("a", encoding="utf-8") as f:
            f.write("memory:\n  enabled: false\n")
    _no_session_start_pass(monkeypatch)
    monkeypatch.setattr("localharness.agent.loop.datetime", _FrozenNow)
    monkeypatch.chdir(tmp_path)
    _write_root_agent(tmp_path, {"consolidation": {"enabled": False}})
    _capture_start_console(monkeypatch)
    out: list[tuple] = []

    async def drive(self):
        self._channel = _Recorder(self._channel, out)
        for c in cmds:
            out.append(("cmd", None, c))
            out.append(("handled", None, str(await self._handle_slash(c))))

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", drive)
    from localharness.cli.start_cmd import _start_async
    await _start_async(None, False, False, str(tmp_path))
    assert out, "the driver never ran"
    return _norm("\n---\n".join(f"[{k}|{s}] {t}" for k, s, t in out), tmp_path) + "\n"


async def test_slash_memory_matches_the_golden_with_memory_on(tmp_path, monkeypatch):
    ids = await _seed(tmp_path)
    e, l, d = ids["user-editor"], ids["project-lang"], ids["deploy-target"]
    cmds = ["/memory", f"/memory show {e}", f"/memory show {d}", "/memory show banana",
            "/memory show 9999", "/memory search neovim", "/memory search zzzznomatch",
            f"/memory forget {l}", f"/memory forget {l} confirm", f"/memory forget {l} confirm",
            f"/memory promote {e}", f"/memory promote {e} confirm", "/memory bogus",
            f"/memory SHOW {e}"]
    text = await _drive(tmp_path, monkeypatch, cmds, memory_off=False)
    assert "/memory failed" not in text and "prod box" in text, text
    assert text == _golden(FIXTURES / "slash_memory_on.txt", text), "/memory drifted from the golden"


async def test_slash_memory_matches_the_golden_with_memory_off(tmp_path, monkeypatch):
    await _seed(tmp_path)
    text = await _drive(tmp_path, monkeypatch, ["/memory", "/memory show 1"], memory_off=True)
    assert text == _golden(FIXTURES / "slash_memory_off.txt", text), "/memory (off) drifted from the golden"
