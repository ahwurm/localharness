"""Shared helpers for the project-instructions tests (phase 11).

A plain module, imported as ``from tests.unit.instructions_support import ...``. Later plans
import from it and never edit it. Every store path resolves through the autouse
``LOCALHARNESS_HOME`` fixture in ``tests/conftest.py`` with ``LOCALHARNESS_DIR`` removed, so
no test here touches the real ``~/.localharness``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from localharness.config import session_presence
from localharness.config.project_records import PROJECT_RECORDS_FILE


@pytest.fixture
def instructions_home(monkeypatch) -> Path:
    """The hermetic global config dir. ``LOCALHARNESS_DIR`` outranks ``LOCALHARNESS_HOME`` in the
    resolution chain, so it is cleared: a developer who exports it must not point these tests at
    a real config dir."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    return Path(os.environ["LOCALHARNESS_HOME"])


def make_root(tmp_path: Path, files: dict[str, str | bytes]) -> Path:
    """Write ``files`` (relative path -> text or raw bytes) under ``tmp_path / "proj"``."""
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return root


def store_file() -> Path:
    """The store file under the hermetic global dir."""
    return Path(os.environ["LOCALHARNESS_HOME"]) / PROJECT_RECORDS_FILE


def load_store() -> dict:
    """The store as plain data, or ``{}`` when it does not exist."""
    path = store_file()
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def register_session(
    session_id: str, *, pid: int | None = None, workspace: Path | None = None
) -> int:
    """Write a presence row for ``session_id`` and return its pid (``None`` -> this process).

    The registry keeps one row per pid, so two live sessions need two pids."""
    mine = os.getpid() if pid is None else pid
    session_presence.register(
        None, agent="t", channel="terminal", session_id=session_id,
        workspace=workspace if workspace is not None else Path.cwd(), pid=mine,
    )
    return mine


def dead_pid() -> int:
    """The pid of a process that has already exited and been waited on."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@contextmanager
def live_child() -> Iterator[int]:
    """Yield the pid of a live child process; terminated and waited on at exit."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield proc.pid
    finally:
        proc.terminate()
        proc.wait()
