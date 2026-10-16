"""The seam every test of a caller of the interview drives: `scripted` answers `interview.ask` in order."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from localharness.cli import interview
from localharness.cli.interview import Answer, Option

CANCEL = object()
UNREACHED = object()


@dataclass(frozen=True)
class Asked:
    question: str
    options: tuple[Option, ...]
    default: Any
    cancel: str
    multi: bool


@dataclass
class ScriptedInterview:
    answers: list
    asked: list[Asked] = field(default_factory=list)

    def ask(self, question: str, options: Sequence[Option], *, default, cancel: str, multi: bool = False,
            input=None, output=None) -> Answer:
        self.asked.append(Asked(question, tuple(options), default, cancel, multi))
        if not self.answers:
            raise AssertionError("asked")
        answer = self.answers.pop(0)
        if answer is CANCEL:
            return Answer((cancel,), True, True)
        if answer is UNREACHED:
            return Answer((cancel,), True, False)
        if isinstance(answer, (set, frozenset)):
            return Answer(tuple(o.key for o in options if o.key in answer), False, True)
        return Answer((answer,), False, True)


def scripted(monkeypatch, *answers) -> ScriptedInterview:
    script = ScriptedInterview(list(answers))
    monkeypatch.setattr(interview, "ask", script.ask)
    return script
