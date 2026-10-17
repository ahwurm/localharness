"""Shared recorders for the launch tests (D-30): the Popen recorder, the strategy-chain recorder and an
open-event recorder.

An importable helper module, like tests/dispatch_support.py: its name does not start with `test_`, so
pytest collects nothing from it. A test module imports what it needs by name:

    from tests.unit.launch_support import PopenRecorder, popen, record_chain  # noqa: F401

- `PopenRecorder` / the `popen` fixture: record what `start_server` hands `subprocess.Popen`; start nothing.
- `record_chain(monkeypatch)`: record the keywords one strategy activation passes down the
  serve_command -> start_server -> wait_ready chain.
- `record_opens()`: record every file the code under test opens, to prove a file is stat-ed but never
  opened.
- `stub_liveness(monkeypatch, alive)` / the `harness_launched` fixture (Phase 5, D-62): answer every
  strategy's `liveness` without touching docker, a pidfile or an HTTP endpoint. The REPL seeds its
  accelerator lock through that liveness, so a test that builds a gpu-primary `OrchestratorREPL`
  stubs it; `harness_launched` is the "the harness launched this primary" answer
  (`Liveness(True, "pid 4321")`).
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from localharness.provider import lifecycle, server


class PopenRecorder:
    """Records what start_server hands subprocess.Popen; starts nothing."""

    calls: list[dict] = []

    def __init__(self, cmd, **kwargs):
        type(self).calls.append({"cmd": cmd, **kwargs})
        self.pid = 4242


_Popen = PopenRecorder  # the old private name, so a later diff stays readable


@pytest.fixture
def popen(monkeypatch):
    PopenRecorder.calls = []
    monkeypatch.setattr(server.subprocess, "Popen", PopenRecorder)
    return PopenRecorder.calls


def record_chain(monkeypatch) -> dict:
    calls: dict = {}

    def serve_command(spec, **kw):
        calls["serve_command"] = kw
        return ["CMD"]

    def start_server(config_dir, cmd, **kw):
        # `dropped_out` is the collector activate hands every launch (D-25), not part of the launch
        # the tests pin: keep it out of the recorded keywords.
        calls["start_server"] = {k: v for k, v in kw.items() if k != "dropped_out"}
        return 7

    async def wait_ready(base_url, **kw):
        calls["wait_ready"] = {k: v for k, v in kw.items() if k == "api_key"}
        return ["m"]

    monkeypatch.setattr(server, "serve_command", serve_command)
    monkeypatch.setattr(server, "start_server", start_server)
    monkeypatch.setattr(server, "wait_ready", wait_ready)
    return calls


def stub_liveness(monkeypatch, alive) -> None:
    """Replace `liveness` on the four strategy classes. `alive` is a bool, or a zero-argument
    callable read at each call (a test that counts the asks, or that follows another seam)."""
    def _liveness(self, spec, config_dir, **kw):
        up = bool(alive() if callable(alive) else alive)
        return lifecycle.Liveness(up, "pid 4321" if up else None)

    for cls in (lifecycle.SpawnedProcessStrategy, lifecycle.ManagedVllmStrategy,
                lifecycle.DaemonStrategy, lifecycle.LmsStrategy):
        monkeypatch.setattr(cls, "liveness", _liveness)


@pytest.fixture
def harness_launched(monkeypatch) -> None:
    """The harness launched the primary: every strategy's liveness is `Liveness(True, "pid 4321")`."""
    stub_liveness(monkeypatch, True)


# The audit hook is installed once per process: an audit hook cannot be removed. Between
# `record_opens()` blocks `_ACTIVE` is None and the hook does nothing at all.
_HOOK_INSTALLED = False
_ACTIVE: list | None = None


def _audit_hook(event: str, args: tuple) -> None:
    if event == "open" and _ACTIVE is not None:
        _ACTIVE.append(str(args[0]))


@contextmanager
def record_opens() -> Iterator[list[str]]:
    """Yield the path of every `open` audit event raised while the block is active.

    It proves a file is stat-ed but never opened (D-06, D-22, DOCTOR-03): `os.stat` and `Path.exists`
    raise no `open` event, `open()` and `Path.read_text` do. Opens by anything else in the process during
    the block are recorded too, so assert on the one path under test, not on the whole list.
    """
    global _HOOK_INSTALLED, _ACTIVE
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True
    opened: list[str] = []
    _ACTIVE = opened
    try:
        yield opened
    finally:
        _ACTIVE = None


def offer_primary_as(monkeypatch, name: str) -> None:
    """The list also offers the primary's record under ``name``: a model the managed restart relaunches it on."""
    import dataclasses

    from localharness.cli import model_ops

    real = model_ops.plan_entries

    def plan(harness, **kw):
        entries = real(harness, **kw)
        record = dataclasses.replace(model_ops.serving_records(harness)[0], name=name, served=name)
        entry = model_ops.PlanEntry(name, "cold", record.spec.runtime, record=record)
        group = model_ops.EntryGroup("this box can launch", (entry,), True)
        return dataclasses.replace(entries, groups=(*entries.groups, group))

    monkeypatch.setattr(model_ops, "plan_entries", plan)
