"""A fake `journalctl` / `systemctl` for the units tests: argv in, canned stdout out.

`fake_systemd` installs a `FakeSystemd` as `units._run`, so no test spawns a process or reads the
journal of the machine it runs on. Every call is recorded (argv, timeout, cap) for the exact-argv
assertions; an argv with no canned answer reads as "not checked" (None), like a missing program.
`fake_manager` does the same for `units._run_manager`, the autostart unit's runner: its answers carry
an exit code and stderr (`ManagerResult`), and an argv with no answer is the runner's None.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pytest

from localharness.provider import units

Matcher = Callable[[tuple[str, ...]], bool]


class FakeSystemd:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.limits: list[tuple[float, int]] = []
        self._answers: list[tuple[Matcher, str | None]] = []

    def answer(self, when: Iterable[str] | Matcher, stdout: str | None) -> None:
        """Canned stdout for an argv equal to `when` (a sequence), or for any argv `when(argv)`
        accepts (a callable). Later answers win over earlier ones; None is "not checked"."""
        if callable(when):
            matcher: Matcher = when
        else:
            wanted = tuple(when)
            matcher = lambda argv: argv == wanted  # noqa: E731
        self._answers.insert(0, (matcher, stdout))

    def answer_prefix(self, *prefix: str, stdout: str | None) -> None:
        """Canned stdout for any argv that starts with `prefix`."""
        self.answer(lambda argv: argv[:len(prefix)] == prefix, stdout)

    def journal(self, entries: Iterable[Mapping[str, Any] | str], *, unit: str | None = None) -> None:
        """Answer the journal query (`journalctl ...`) with one JSON line per entry (a str entry is
        written verbatim, for the non-JSON line cases)."""
        text = "".join((e if isinstance(e, str) else json.dumps(e)) + "\n" for e in entries)
        self.answer(lambda argv: argv[0] == "journalctl" and (
            unit is None or f"_SYSTEMD_USER_UNIT={unit}" in argv), text)

    def __call__(self, argv: list[str], *, timeout: float, cap: int, **kw: Any) -> str | None:
        key = tuple(argv)
        self.calls.append(key)
        self.limits.append((timeout, cap))
        for matcher, stdout in self._answers:
            if matcher(key):
                if stdout is not None and len(stdout.encode()) > cap:
                    return None
                return stdout
        return None

    def calls_of(self, program: str) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c and c[0] == program]


@pytest.fixture
def fake_systemd(monkeypatch) -> FakeSystemd:
    fake = FakeSystemd()
    monkeypatch.setattr(units, "_run", fake)
    return fake


class FakeManager:
    """`units._run_manager` for a test: argv in, a `ManagerResult` (or None) out. Every call is recorded
    (argv, timeout, cap); an argv nobody answered is None ("the manager could not be asked")."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.limits: list[tuple[float, int]] = []
        self._answers: list[tuple[Matcher, units.ManagerResult | None]] = []

    def answer(self, when: Iterable[str] | Matcher, out: str | None = "", code: int = 0,
               err: str = "") -> None:
        """The answer for an argv equal to `when` (a sequence), or for any argv `when(argv)` accepts
        (a callable). Later answers win over earlier ones; `out=None` is the runner's None."""
        if callable(when):
            matcher: Matcher = when
        else:
            wanted = tuple(when)
            matcher = lambda argv: argv == wanted  # noqa: E731
        result = None if out is None else units.ManagerResult(code, out, err)
        self._answers.insert(0, (matcher, result))

    def answer_prefix(self, *prefix: str, out: str | None = "", code: int = 0, err: str = "") -> None:
        """The answer for any argv that starts with `prefix`."""
        self.answer(lambda argv: argv[:len(prefix)] == prefix, out=out, code=code, err=err)

    def __call__(self, argv: list[str], *, timeout: float, cap: int, **kw: Any) -> units.ManagerResult | None:
        key = tuple(argv)
        self.calls.append(key)
        self.limits.append((timeout, cap))
        for matcher, result in self._answers:
            if matcher(key):
                if result is not None and len(result.out.encode()) > cap:
                    return None
                return result
        return None

    def calls_of(self, program: str) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c and c[0] == program]


@pytest.fixture
def fake_manager(monkeypatch) -> FakeManager:
    fake = FakeManager()
    monkeypatch.setattr(units, "_run_manager", fake)
    return fake
