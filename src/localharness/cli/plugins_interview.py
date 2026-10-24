"""The Plugins flow: /plugins on a terminal, drawn inside one suspension of the input box."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

import click
from rich.text import Text

from localharness.channels.base import sanitize_for_display
from localharness.cli import plugins_cmd
from localharness.cli.components_cmd import redact_text
from localharness.cli.interview import Option, Unreachable, confirm_line, n_of, prompt_line
from localharness.cli.plugins_cmd import (
    ACTION_INSTALL, ACTION_RERUN, ACTION_SHOW_INSTALL, PluginSession, Refusal, ReloadPlan, Touch, refused,
)
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
ASKS_PENDING = "{asks} parked — answer in /pending after closing; the keys that answer asks are off here"
ROW_STATES = MappingProxyType({
    "probe_failed_stops": "on — check failed, stops {at_close}; Info shows why",
    "probe_failed": "on — check failed; Info shows why",
    "unconfigured": "on — not set up{stops}; {rerun}",
    "starts": "on — starts {at_close}",
    "stops": "off {at_close}",
    "rebinds": "on — changed, applies {at_close}",
    "load_failed": "on — did not load (Info shows why)",
    "not_running": "on — not running; {rerun} starts it",
    "needs_extra_argv": "needs {extra} — {install}",
    "needs_extra_manual": "needs {extra} — {show_install}",
    "needs_extra_third_party": "needs {extra} — install it by hand, then {rerun}",
    "off_needs_extra": "off (needs {extra})",
})

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


def _say(console: Any, *lines: str) -> None:
    for line in lines:
        console.print(Text(sanitize_for_display(line)), soft_wrap=True)


def _where(plan: ReloadPlan) -> str:
    return " and ".join(w for w, n in (("in /pending", plan.parked), ("shown next", plan.waiting)) if n)


def _close_label(plan: ReloadPlan) -> str:
    if plan.reload:
        return CLOSE_LABELS["restarts"].format(plugins=n_of(len(plan.names), "plugin"))
    if plan.names:
        return CLOSE_LABELS["blocked"].format(asks=n_of(plan.parked + plan.waiting, "ask"), where=_where(plan))
    return CLOSE_LABELS["plain"]


def _text(key: str, plan: ReloadPlan, **fill: str) -> str:
    return ROW_STATES[key].format(
        at_close=at_close(plan), rerun=ACTION_RERUN, install=ACTION_INSTALL, show_install=ACTION_SHOW_INSTALL, **fill)


def _row_state(entry: Any, session: PluginSession, plan: ReloadPlan) -> str:
    name, touch = entry.name, session.touched.get(entry.name)
    probe = touch.probe[0] if touch is not None and touch.probe is not None else None
    running, queued = name in session.running, name in plan.names
    if entry.state == "on":
        if probe == "failed":
            return _text("probe_failed_stops" if running and queued else "probe_failed", plan)
        if probe == "unconfigured" or (probe is None and name in session.unconfigured):
            return _text("unconfigured", plan, stops=f", stops {at_close(plan)}" if running and queued else "")
        if queued:
            return _text("rebinds" if running else "starts", plan)
        if probe is None and name in session.failed:
            return _text("load_failed", plan)
        return "on" if running else _text("not_running", plan)
    if entry.state in ("off", "available") and running and queued:
        return _text("stops", plan)
    if entry.state == "needs-extra" and entry.manifest is not None:
        route = plugins_cmd.install_route_for(entry)
        key = ("needs_extra_third_party" if route is None
               else "needs_extra_argv" if route.argv is not None else "needs_extra_manual")
        return _text(key, plan, extra=f"localharness[{entry.manifest.requires_extra}]")
    if entry.state == "off" and (extra := plugins_cmd.extra_for(entry)):
        return _text("off_needs_extra", plan, extra=f"localharness[{extra}]")
    return entry.state


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
    options = [Option(str(i), entry.name, _row_state(entry, session, plan)) for i, entry in enumerate(entries)]
    options.append(Option(CLOSE, _close_label(plan)))
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
        if flow.plan.parked:
            _say(console, ASKS_PENDING.format(asks=n_of(flow.plan.parked, "ask")))
        try:
            await _root(flow)
        except Exception:  # noqa: BLE001 — Plugins closes; the session goes on
            return _finish(flow, closing=False)
        return _finish(flow, closing=True)
