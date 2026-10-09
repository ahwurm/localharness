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
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from localharness.provider import server


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
        calls["start_server"] = kw
        return 7

    async def wait_ready(base_url, **kw):
        calls["wait_ready"] = {k: v for k, v in kw.items() if k == "api_key"}
        return ["m"]

    monkeypatch.setattr(server, "serve_command", serve_command)
    monkeypatch.setattr(server, "start_server", start_server)
    monkeypatch.setattr(server, "wait_ready", wait_ready)
    return calls


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
