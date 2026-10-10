"""Shared fixtures and helpers for the machine-trust unit tests (Phase 4).

`g` is conftest's hermetic `LOCALHARNESS_HOME`; `write_yaml` writes a settings file there;
`answers` / `never_asked` stand in for `rich.prompt.Confirm.ask`; `names` / `by_name` read a
`trust.machine_snapshot` result. `test_machine_fingerprint.py` keeps its own private copies of the
first four on purpose: this module is for the files that came after it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def g() -> Path:
    return Path(os.environ["LOCALHARNESS_HOME"])


def write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def answers(monkeypatch, *replies) -> list[str]:
    """Answer each `Confirm.ask` in turn (an exception is raised, not returned); the returned list
    collects the prompts that were put."""
    asked: list[str] = []
    queue = list(replies)

    def _ask(prompt, *_a, **_kw):
        asked.append(str(prompt))
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


def never_asked(monkeypatch) -> None:
    def _boom(*_a, **_kw):
        raise AssertionError("asked where the rule forbids asking")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def names(snapshot: list[dict]) -> list[tuple[str, str, str]]:
    """The `(file, kind, name)` triples of a snapshot, in its own order."""
    return [(e["file"], e["kind"], e["name"]) for e in snapshot]


def by_name(snapshot: list[dict], name: str) -> dict:
    """The one entry called ``name`` (an AssertionError when there are none or several)."""
    found = [e for e in snapshot if e["name"] == name]
    assert len(found) == 1, f"{name!r}: {len(found)} entries in {names(snapshot)}"
    return found[0]
