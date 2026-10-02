"""MEMP-01 / MEMP-02 / MEMP-04 on the REAL start, with memory as the bundled plugin (Phase 47).

Every drive is 47-01's baseline drive (`test_memory_compat_baseline_e2e._start`, imported): the real
`_start_async`, the external boundaries stubbed, the provider on the loopback discard port, the
model a recorder. On top, `start_plugins` is wrapped by a recorder that calls through and keeps the
LifecycleResult and the session's registry, so each assertion reads what the lifecycle really did.

- MEMP-02: `bash_exec` and the three memory tools are on the root together, declared as the plugin
  declares them, and survive the capability floor (the model was offered them).
- MEMP-01: each soft piece (session row, accumulator, scheduler construction) fails ALONE — its
  warning, tools and injection intact; a store that will not open fails the WHOLE plugin — named in
  `result.failed`, tools unregistered, slot empty, guardrails still in the prompt.
- MEMP-04: a pass the running plugin's scheduler launches emits ConsolidationStarted/Finished on the
  session bus, where the terminal channel's own subscriptions receive them.
"""
from __future__ import annotations

import asyncio
import functools
from pathlib import Path

import pytest

import tests.integration.test_memory_compat_baseline_e2e as baseline
from tests.integration.test_all_plugins_off_e2e import _record
from tests.integration.test_memory_compat_baseline_e2e import (
    MEMORY_TOOLS, _no_session_start_pass, _seed_store, _start, _warnings,
)

ROOT = Path(__file__).resolve().parents[2]
GUARDRAILS_SENTINEL = "BASELINE-GUARDRAILS-SENTINEL"
# The 17 memory test files 47-CONTEXT names (15 + the two slot files).
MEMORY_TEST_FILES = (
    "test_global_memory_untouched.py", "test_markdown_memory.py", "test_memory_activation.py",
    "test_memory_cli.py", "test_memory_cmd.py", "test_memory_consolidation.py",
    "test_memory_global_safety_dir.py", "test_memory_index_origin_labels.py",
    "test_memory_index_tools.py", "test_memory_promote.py", "test_memory_salience_archive.py",
    "test_memory_store_exec_deny.py", "test_memory_store.py", "test_memory_window_store.py",
    "test_repl_memory_cmd.py", "test_memory_slot.py", "test_memory_slot_seat.py",
)


async def _run(tmp_path, monkeypatch, *, memory=None, extra_config: str = "", **kw) -> dict:
    lifecycles: list = []
    _record(monkeypatch, "localharness.plugins.lifecycle.start_plugins", lifecycles, is_async=True)
    if extra_config:  # appended after the drive writes config.yaml, via the boundary helper
        real = baseline._stub_start_boundaries

        def stub(tmp, mp, **k):
            real(tmp, mp, **k)
            with (tmp / "config.yaml").open("a", encoding="utf-8") as f:
                f.write(extra_config)
        monkeypatch.setattr(baseline, "_stub_start_boundaries", stub)
    out = await _start(tmp_path, monkeypatch, memory=memory, **kw)
    assert len(lifecycles) == 1, "start_plugins did not run exactly once"
    kwargs, result = lifecycles[0]
    return {**out, "result": result, "registry": kwargs["registry"]}


def _prompt(out: dict) -> str:
    return out["calls"][0]["system"]["content"]


def _tools(out: dict) -> set[str]:
    return set(out["calls"][0]["tools"])


# --------------------------------------------------------------------------- MEMP-02


async def test_root_boots_with_bash_exec_and_the_memory_tools_together(tmp_path, monkeypatch):
    out = await _run(tmp_path, monkeypatch, memory={"consolidation": {"enabled": False}})
    reg = out["registry"]
    assert {"bash_exec"} | MEMORY_TOOLS <= set(reg._tools["global"]), sorted(reg._tools["global"])
    expected_origin = {"memory_search": "untrusted", "memory_get": "untrusted", "remember": "trusted"}
    for name, origin in expected_origin.items():
        s = reg.schema_of(name)
        assert (s.source_plugin, s.ingest, s.host, s.result_origin) == ("memory", "none", "safe", origin), (
            name, s.source_plugin, s.ingest, s.host, s.result_origin)
    assert reg.schema_of("bash_exec").source_plugin is None
    # The capability floor kept all four: the model was offered them on the first request.
    assert {"bash_exec"} | MEMORY_TOOLS <= _tools(out), sorted(_tools(out))
    assert out["result"].slot.occupant_name == "memory"


# --------------------------------------------------------------------------- MEMP-01


def _break(monkeypatch, target: str, *, sync: bool = False) -> None:
    import importlib
    mod_path, cls_name, meth = target.rsplit(".", 2)
    cls = getattr(importlib.import_module(mod_path), cls_name)

    if sync:
        def boom(self, *a, **k):
            raise RuntimeError("boom")
    else:
        async def boom(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(cls, meth, boom)


async def test_broken_scheduler_construction_alone_leaves_tools_and_injection(tmp_path, monkeypatch):
    await _seed_store(tmp_path / "on")
    _no_session_start_pass(monkeypatch)
    _break(monkeypatch, "localharness.memory.consolidation.ConsolidationScheduler.__init__", sync=True)
    out = await _run(tmp_path / "on", monkeypatch, memory={"consolidation": {"enabled": True}})
    assert any(w.startswith("memory consolidation:") for w in _warnings(out["printed"])), out["printed"]
    assert "memory" in out["result"].loaded_names and out["result"].slot.occupied
    assert MEMORY_TOOLS <= set(out["registry"]._tools["global"])
    assert "## Agent Memory\n" in _prompt(out)

    # The twin with memory off by the canonical key: no section, no tools — the section above was memory's.
    (tmp_path / "off").mkdir()
    off = await _run(tmp_path / "off", monkeypatch, memory={"consolidation": {"enabled": True}},
                     extra_config="memory:\n  enabled: false\n")
    assert "## Agent Memory" not in _prompt(off)
    assert not (MEMORY_TOOLS & set(off["registry"]._tools["global"]))
    assert not (MEMORY_TOOLS & _tools(off)) and not off["result"].slot.occupied


async def test_broken_store_open_empties_the_slot(tmp_path, monkeypatch):
    from localharness.cli import start_cmd
    real_start = start_cmd._start_async
    # The verbose banner too: `_start` passes verbose=False; force it on, nothing else changed.
    monkeypatch.setattr(start_cmd, "_start_async",
                        lambda a, _v, d, c, **k: real_start(a, True, d, c, **k))
    _break(monkeypatch, "localharness.memory.sqlite.MemoryStore.open")
    out = await _run(tmp_path, monkeypatch, memory={"consolidation": {"enabled": False}})
    result = out["result"]
    assert "memory" in result.failed, result.failed
    assert "memory" not in result.loaded_names
    assert not (MEMORY_TOOLS & set(out["registry"]._tools["global"])), "a failed plugin kept its tools"
    assert not (MEMORY_TOOLS & _tools(out))
    assert result.slot.occupied is False
    prompt = _prompt(out)
    assert prompt.count(GUARDRAILS_SENTINEL) == 1, "guardrails lost with memory's failure"
    assert "## Agent Memory" not in prompt
    assert any(w.startswith("plugin memory:") for w in _warnings(out["printed"])), out["printed"]
    assert any("Memory: in-memory (no persistence)" in p for p in out["printed"]), out["printed"]


@pytest.mark.parametrize(("prefix", "target"), [
    ("session-start:", "localharness.memory.sqlite.MemoryStore.create_session"),
    ("session-accumulator:", "localharness.cli.session_accumulator.SessionAccumulator.open"),
], ids=["test_broken_session_row_alone", "test_broken_accumulator_alone"])
async def test_one_soft_piece_alone(tmp_path, monkeypatch, prefix, target):
    await _seed_store(tmp_path)
    _no_session_start_pass(monkeypatch)
    _break(monkeypatch, target)
    out = await _run(tmp_path, monkeypatch, memory={"consolidation": {"enabled": True}})
    warnings = _warnings(out["printed"])
    assert any(w.startswith(prefix) for w in warnings), warnings
    assert not any(w.startswith("plugin memory:") for w in warnings), warnings
    assert "memory" in out["result"].loaded_names and out["result"].slot.occupied
    assert MEMORY_TOOLS <= set(out["registry"]._tools["global"]) and MEMORY_TOOLS <= _tools(out)
    assert "## Agent Memory\n" in _prompt(out)


# --------------------------------------------------------------------------- MEMP-04 composed


async def test_a_plugin_run_consolidation_pass_reaches_the_terminal_subscribers(tmp_path, monkeypatch):
    from localharness.channels.terminal import TerminalChannel
    from localharness.core.events import ConsolidationFinished, ConsolidationStarted

    _no_session_start_pass(monkeypatch)
    terminal_saw: list[str] = []

    async def on_started(self, event):
        terminal_saw.append(type(event).__name__)

    async def on_finished(self, event):
        terminal_saw.append(type(event).__name__)
    # Recorders in the terminal channel's own handler slots: start() subscribes THESE, by name.
    monkeypatch.setattr(TerminalChannel, "on_consolidation_started", on_started)
    monkeypatch.setattr(TerminalChannel, "on_consolidation_finished", on_finished)

    lifecycles: list = []
    _record(monkeypatch, "localharness.plugins.lifecycle.start_plugins", lifecycles, is_async=True)
    seen: dict = {"bus": []}

    async def session(self):
        assert isinstance(self._channel, TerminalChannel), type(self._channel)
        self._channel._history_file = str(tmp_path / "history")
        await self._channel.start()
        try:
            for cls in (ConsolidationStarted, ConsolidationFinished):
                self._bus.subscribe(cls, lambda e: seen["bus"].append(type(e).__name__))
            (rp,) = [r for r in lifecycles[0][1].running if r.name == "memory"]
            seen["same_bus"] = rp.ctx.bus is self._bus
            sched = rp.plugin._sched
            assert sched is not None, "consolidation is on but the plugin built no scheduler"
            sched.launch()
            await asyncio.wait_for(sched._run_task, 10)
        finally:
            await self._channel.stop()

    real = baseline._stub_start_boundaries
    monkeypatch.setattr(baseline, "_stub_start_boundaries",
                        functools.partial(real, repl_run=session))
    await _start(tmp_path, monkeypatch, memory={"consolidation": {"enabled": True}}, turn=False)

    assert seen["same_bus"] is True, "the plugin's bus is not the session bus"
    assert seen["bus"] == ["ConsolidationStarted", "ConsolidationFinished"], seen["bus"]
    assert terminal_saw == ["ConsolidationStarted", "ConsolidationFinished"], terminal_saw


def test_the_memory_test_files_are_all_present():
    missing = [f for f in MEMORY_TEST_FILES if not (ROOT / "tests" / "unit" / f).is_file()]
    assert missing == [], missing
