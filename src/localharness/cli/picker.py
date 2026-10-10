"""The one-line arrow-key picker (spec 16 section 7.1).

The same keys as the input box: Left/Right and Tab/Shift-Tab move the highlight (wrapping), Enter
returns it, Esc and Ctrl-C return ``None``. ``/effort`` uses it now; ``/model`` and the preset
chooser reuse it later. This module imports prompt_toolkit and nothing from the REPL or from any
channel: the terminal channel hands it a style (merged with its own) and its erase-on-exit flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

PICKER_EMPTY_ERROR = "picker needs at least one option"
# `selected` is the only rule the picker owns: the terminal channel merges this style with its own
# INPUT_STYLE (which has `hint` and `frame` but no `selected`), so the highlight survives there.
PICKER_STYLE = Style.from_dict({"selected": "reverse bold", "hint": "", "frame": ""})


def build_picker(
    options: Sequence[str],
    current: str | None,
    *,
    input: Any = None,
    output: Any = None,
    style: Any = None,
    erase_when_done: bool = True,
) -> Application:
    """A one-line chooser over ``options`` with the highlight starting on ``current``.

    ``current`` not among the options starts on the first one. Raises ``ValueError`` on no options
    before any Application exists.
    """
    opts = list(options)
    if not opts:
        raise ValueError(PICKER_EMPTY_ERROR)
    state = [opts.index(current) if current in opts else 0]

    def _fragments() -> list[tuple[str, str]]:
        return [
            ("class:selected" if i == state[0] else "class:hint", f" {o} ")
            for i, o in enumerate(opts)
        ]

    kb = KeyBindings()

    def _move(step: int) -> None:
        state[0] = (state[0] + step) % len(opts)

    @kb.add("right")
    @kb.add("tab")
    def _forward(event) -> None:
        _move(1)

    @kb.add("left")
    @kb.add("s-tab")
    def _back(event) -> None:
        _move(-1)

    @kb.add("enter")
    def _choose(event) -> None:
        event.app.exit(result=opts[state[0]])

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _leave(event) -> None:
        event.app.exit(result=None)

    body = Window(FormattedTextControl(_fragments), height=1, style="class:frame")
    app = Application(
        layout=Layout(body),
        key_bindings=kb,
        style=style or PICKER_STYLE,
        mouse_support=False,
        erase_when_done=erase_when_done,
        **({"input": input} if input is not None else {}),
        **({"output": output} if output is not None else {}),
    )
    # Test seams: the rendered fragments and the highlighted index, without a terminal.
    app._lh_picker_fragments = _fragments  # type: ignore[attr-defined]
    app._lh_picker_index = lambda: state[0]  # type: ignore[attr-defined]
    return app


async def pick(options: Sequence[str], current: str | None, **kw: Any) -> str | None:
    """Run the picker; the chosen option, or ``None`` when the person left it."""
    return await build_picker(options, current, **kw).run_async()
