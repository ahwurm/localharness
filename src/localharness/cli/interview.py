"""The one question app every first-run and model-library question goes through."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from localharness.cli.picker import PICKER_STYLE
from localharness.cli.workspace import local_operator, visible_text

HINT_SINGLE = "↑↓ move · Enter confirm"
HINT_MULTI = "↑↓ move · Space select · Enter confirm"
INTERVIEW_EMPTY_ERROR = "interview needs at least one option"


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
    default: int,
    cancel: str,
    multi: bool = False,
    input: Any = None,
    output: Any = None,
) -> Application:
    if multi:
        raise NotImplementedError("multi")
    opts = list(options)
    state: dict[str, Any] = {"index": default, "summary": None}

    def _row(i: int) -> list[tuple[str, str]]:
        o = opts[i]
        text = f"  {i + 1}. {visible_text(o.label)}"
        hint = f" — {visible_text(o.hint)}" if o.hint else ""
        if i == state["index"]:
            return [("class:selected", text + hint)]
        return [("", text), ("class:hint", hint)] if hint else [("", text)]

    def _parts() -> tuple[list, list, list]:
        if state["summary"] is not None:
            return [("", state["summary"])], [], []
        rows: list[tuple[str, str]] = []
        for i in range(len(opts)):
            rows += ([("", "\n")] if i else []) + _row(i)
        return [("", visible_text(question))], rows, [("class:hint", HINT_SINGLE)]

    def _fragments() -> list[tuple[str, str]]:
        q, rows, hint = _parts()
        return q + ([("", "\n")] + rows + [("", "\n")] + hint if rows else [])

    def _finish(event, keys: tuple[str, ...], summary: str, cancelled: bool) -> None:
        state["summary"] = summary
        event.app.exit(result=Answer(keys, cancelled))

    kb = KeyBindings()

    @kb.add("down")
    def _down(event) -> None:
        state["index"] = (state["index"] + 1) % len(opts)

    @kb.add("up")
    def _up(event) -> None:
        state["index"] = (state["index"] - 1) % len(opts)

    @kb.add("enter")
    def _choose(event) -> None:
        o = opts[state["index"]]
        _finish(event, (o.key,), f"{visible_text(question)} {visible_text(o.label)}", False)

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _leave(event) -> None:
        _finish(event, (cancel,), f"{visible_text(question)} (cancelled)", True)

    live = Condition(lambda: state["summary"] is None)
    body = HSplit([
        Window(FormattedTextControl(lambda: _parts()[0]), wrap_lines=True),
        ConditionalContainer(
            Window(FormattedTextControl(lambda: _parts()[1], show_cursor=False), wrap_lines=True), filter=live),
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
    app._lh_interview_state = lambda: dict(state)  # type: ignore[attr-defined]
    return app


def ask(
    question: str,
    options: Sequence[Option],
    *,
    default: int,
    cancel: str,
    multi: bool = False,
    input: Any = None,
    output: Any = None,
) -> Answer:
    """Ask one question; `reached=False` when no person could be asked, with nothing read."""
    if not options:
        raise ValueError(INTERVIEW_EMPTY_ERROR)
    if len({o.key for o in options}) != len(options):
        raise ValueError("interview option keys must be unique")
    if not cancel:
        raise ValueError("interview needs a cancel key")
    if not multi and not 0 <= default < len(options):
        raise ValueError("interview default is not one of the options")
    if not local_operator(False):
        return Answer((cancel,), True, reached=False)
    app = build_interview(question, options, default=default, cancel=cancel, multi=multi,
                          input=input, output=output)
    try:
        return app.run(in_thread=True)
    except Exception:
        return Answer((cancel,), True, reached=False)
