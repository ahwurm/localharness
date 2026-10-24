"""The Plugins flow: /plugins on a terminal, drawn inside one suspension of the input box."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

import click

from localharness.channels.base import sanitize_for_display
from localharness.cli.components_cmd import redact_text
from localharness.cli.interview import Option, Unreachable, confirm_line, n_of, prompt_line
from localharness.cli.plugins_cmd import PluginSession, Refusal, ReloadPlan, Touch, refused
from localharness.core.interrupt import sigint_cancels

PLUGINS_QUESTION = "Plugins — Esc or Ctrl-C closes"
CLOSE = "+close"
CLOSE_LABELS = MappingProxyType({
    "plain": "Close",
    "restarts": "Close — restarts to apply ({plugins})",
    "blocked": "Close — restart deferred: {asks} to answer after closing ({where}), then open and close Plugins",
})
_CANT_DRAW = ("use /plugins enable <name>", "the screen could not be drawn", "Quit and start localharness again to apply")
UNREACHABLE = MappingProxyType({why: _CANT_DRAW for why in Unreachable} | {Unreachable.TOO_SMALL: (
    "enlarge the window, then open /plugins again", "the window got too small",
    "Enlarge the window, then open /plugins and close it")})
UNAVAILABLE = "Plugins can't be drawn here — {fix}"

_STOP = object()
_STOPS = (click.exceptions.Abort, click.exceptions.Exit, EOFError, KeyboardInterrupt, SystemExit)


@dataclass
class Flow:
    channel: Any
    session: PluginSession
    should_reload: Callable[[Mapping[str, Touch]], ReloadPlan]
    console: Any
    close: Callable[[], None]
    prompt: Callable
    confirm: Callable
    plan: ReloadPlan


@dataclass(frozen=True)
class _Escape:
    error: str


class _Abandon(Exception):
    def __init__(self, *, why: Unreachable | None = None, error: str = "") -> None:
        super().__init__(error)
        self.why, self.error = why, error


def unavailable(why: Unreachable) -> str:
    return UNAVAILABLE.format(fix=UNREACHABLE[why][0])


def at_close(plan: ReloadPlan) -> str:
    return "when you close Plugins" if plan.reload else "when you next close Plugins with no asks pending"


def _first_line(exc: Exception) -> str:
    return redact_text(sanitize_for_display(f"{type(exc).__name__}: {exc}".splitlines()[0]))


async def _contained(flow: Flow, node: Callable, names: tuple[str, ...], progress: Any) -> Any:
    try:
        return await node()
    except _STOPS:
        return _STOP
    except Exception as exc:  # noqa: BLE001 — a node never ends the session
        if isinstance(exc, Refusal) and names and progress is not None:
            return refused(flow.session, names[0], str(exc), "done" if progress.phase == "question_installed" else "none")
        return _Escape(_first_line(exc))


async def guarded(flow: Flow, node: Callable, *, names: tuple[str, ...] = (), progress: Any = None) -> Any:
    """The node's value, `_STOP` for a user stop, or `_Abandon` for an error the node let out."""
    task = asyncio.ensure_future(_contained(flow, node, names, progress))
    async with sigint_cancels(task) as cancelled:
        try:
            got = await task
        except asyncio.CancelledError:
            if not cancelled():
                task.cancel()
                raise
            return _STOP
    if isinstance(got, _Escape):
        raise _Abandon(error=got.error)
    return got


async def _ask_root(flow: Flow) -> Any:
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    session = flow.session
    entries = resolve(ConfigLoader(config_dir=session.cfg_path, local_config_dir=session.workspace)).plan.entries
    flow.plan = plan = flow.should_reload(session.touched)
    close = (CLOSE_LABELS["restarts"].format(plugins=n_of(len(plan.names), "plugin")) if plan.reload
             else CLOSE_LABELS["plain"])
    options = [Option(str(i), entry.name, entry.state) for i, entry in enumerate(entries)] + [Option(CLOSE, close)]
    return await flow.channel.run_interview(PLUGINS_QUESTION, options, default=0, cancel=CLOSE)


async def _root(flow: Flow) -> None:
    while True:
        answer = await guarded(flow, partial(_ask_root, flow))
        if answer is _STOP:
            return
        if not answer.reached:
            raise _Abandon(why=answer.why)
        if answer.cancelled or answer.keys == (CLOSE,):
            return


def _finish(flow: Flow, *, closing: bool) -> bool:
    plan = flow.should_reload(flow.session.touched)
    if closing and plan.reload:
        flow.close()
        return True
    return False


async def run(channel: Any, session: PluginSession, should_reload: Callable[[Mapping[str, Touch]], ReloadPlan]) -> bool:
    """Plugins for one /plugins; True when closing it should restart the session."""
    async with channel.suspended_box() as (_output, console, close):
        flow = Flow(channel, session, should_reload, console, close, partial(prompt_line, console=console),
                    partial(confirm_line, console=console), should_reload(session.touched))
        try:
            await _root(flow)
        except Exception:  # noqa: BLE001 — Plugins closes; the session goes on
            return _finish(flow, closing=False)
        return _finish(flow, closing=True)
