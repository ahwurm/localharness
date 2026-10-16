"""The one question app every first-run and model-library question goes through."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output.defaults import create_output

from localharness.cli.picker import PICKER_STYLE
from localharness.cli.workspace import local_operator, visible_text

HINT_SINGLE = "↑↓ move · Enter confirm"
HINT_MULTI = "↑↓ move · Space select · Enter confirm"
INTERVIEW_EMPTY_ERROR = "interview needs at least one option"
MIN_ROWS = 3
MIN_COLUMNS = 20
CHROME_ROWS = 3


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    hint: str = ""


@dataclass(frozen=True)
class Answer:
    """`reached` is False when nobody could be asked; `cancelled` alone cannot tell that from a person's Esc."""
    keys: tuple[str, ...]
    cancelled: bool
    reached: bool = True


def build_interview(
    question: str,
    options: Sequence[Option],
    *,
    default: int | frozenset[str] | set[str],
    cancel: str,
    multi: bool = False,
    input: Any = None,
    output: Any = None,
) -> Application:
    opts = list(options)
    shown_question = visible_text(question)
    rows = (output if output is not None else create_output()).get_size().rows
    chosen = {i for i, o in enumerate(opts) if multi and o.key in default}
    state: dict[str, Any] = {"index": 0 if multi else default, "summary": None, "digits": "", "toggled": None}

    def _row(i: int) -> list[tuple[str, str]]:
        o = opts[i]
        mark = ("[x] " if i in chosen else "[ ] ") if multi else "  "
        text = f"{mark}{i + 1}. {visible_text(o.label)}"
        hint = f" — {visible_text(o.hint)}" if o.hint else ""
        if i == state["index"]:
            return [("class:selected", text + hint)]
        return [("", text), ("class:hint", hint)] if hint else [("", text)]

    def _parts() -> tuple[list, list, list]:
        if state["summary"] is not None:
            return [("", state["summary"])], [], []
        lines: list[tuple[str, str]] = []
        for i in range(len(opts)):
            lines += ([("", "\n")] if i else []) + _row(i)
        return [("", shown_question)], lines, [("class:hint", HINT_MULTI if multi else HINT_SINGLE)]

    def _fragments() -> list[tuple[str, str]]:
        q, lines, hint = _parts()
        return q + ([("", "\n")] + lines + [("", "\n")] + hint if lines else [])

    def _finish(event, keys: tuple[str, ...], summary: str, cancelled: bool) -> None:
        state["summary"] = f"{shown_question} {summary}"
        event.app.exit(result=Answer(keys, cancelled))

    def _move(step: int) -> None:
        state.update(index=(state["index"] + step) % len(opts), digits="", toggled=None)

    def _flip(i: int) -> None:
        chosen.symmetric_difference_update({i})

    kb = KeyBindings()

    @kb.add("down")
    def _down(event) -> None:
        _move(1)

    @kb.add("up")
    def _up(event) -> None:
        _move(-1)

    @kb.add("enter")
    def _choose(event) -> None:
        if multi:
            picked = [o for i, o in enumerate(opts) if i in chosen]
            _finish(event, tuple(o.key for o in picked), ", ".join(visible_text(o.label) for o in picked) or "none", False)
        else:
            o = opts[state["index"]]
            _finish(event, (o.key,), visible_text(o.label), False)

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _leave(event) -> None:
        _finish(event, (cancel,), "(cancelled)", True)

    if multi:
        @kb.add("space")
        def _toggle(event) -> None:
            _flip(state["index"])
            state.update(digits="", toggled=None)

    def _number(digit: str) -> None:
        pair = state["digits"] + digit
        if len(state["digits"]) == 1 and 1 <= int(pair) <= len(opts):
            if state["toggled"] is not None:
                _flip(state["toggled"])
            target, state["digits"], state["toggled"] = int(pair) - 1, "", None
        elif 1 <= int(digit) <= len(opts):
            target, state["digits"] = int(digit) - 1, digit
        else:
            state.update(digits="", toggled=None)
            return
        state["index"] = target
        if multi:
            _flip(target)
            state["toggled"] = target if state["digits"] else None

    for d in "0123456789":
        kb.add(d)(lambda event, d=d: _number(d))

    live = Condition(lambda: state["summary"] is None)
    options_window = Window(
        FormattedTextControl(lambda: _parts()[1], show_cursor=False,
                             get_cursor_position=lambda: Point(0, state["index"])),
        height=min(len(opts), max(1, rows - CHROME_ROWS)), wrap_lines=True)
    body = HSplit([
        Window(FormattedTextControl(lambda: _parts()[0]), wrap_lines=True),
        ConditionalContainer(options_window, filter=live),
        ConditionalContainer(Window(FormattedTextControl(lambda: _parts()[2]), wrap_lines=True), filter=live),
    ])
    app = Application(
        layout=Layout(body),
        key_bindings=kb,
        style=PICKER_STYLE,
        mouse_support=False,
        erase_when_done=False,
        **({"input": input} if input is not None else {}),
        **({"output": output} if output is not None else {}),
    )
    app._lh_interview_fragments = _fragments  # type: ignore[attr-defined]
    app._lh_interview_state = lambda: {**state, "chosen": frozenset(opts[i].key for i in chosen)}  # type: ignore[attr-defined]
    return app


def _validate(options: Sequence[Option], default: Any, cancel: str, multi: bool) -> None:
    if not options:
        raise ValueError(INTERVIEW_EMPTY_ERROR)
    keys = {o.key for o in options}
    if len(keys) != len(options):
        raise ValueError("interview option keys must be unique")
    if not cancel:
        raise ValueError("interview needs a cancel key")
    if multi and not (isinstance(default, (set, frozenset)) and default <= keys):
        raise ValueError("interview default must be a set of option keys when multi")
    if not multi and not (isinstance(default, int) and 0 <= default < len(options)):
        raise ValueError("interview default is not one of the options")


def ask(
    question: str,
    options: Sequence[Option],
    *,
    default: int | frozenset[str] | set[str],
    cancel: str,
    multi: bool = False,
    input: Any = None,
    output: Any = None,
) -> Answer:
    """Ask one question; `reached=False` when no person could be asked, with nothing read."""
    _validate(options, default, cancel, multi)
    unreached = Answer((cancel,), True, reached=False)
    if not local_operator(False) or os.environ.get("TERM") == "dumb":
        return unreached
    try:
        output = output if output is not None else create_output()
        size = output.get_size()
        if size.rows < MIN_ROWS or size.columns < MIN_COLUMNS:
            return unreached
        app = build_interview(question, options, default=default, cancel=cancel, multi=multi,
                              input=input, output=output)
        return app.run(in_thread=True)
    except Exception:
        return unreached
