"""The Plugins flow: /plugins on a terminal, drawn inside one suspension of the input box."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
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
    ACTION_DISABLE, ACTION_ENABLE, ACTION_INFO, ACTION_INSTALL, ACTION_RERUN, ACTION_SHOW_INSTALL, ENABLE_MANUAL,
    OVERRIDE_LINE, OVERRIDE_SAVED, ROUTE_BACK, Kind, PassProgress, PassResult, PluginSession, Refusal, ReloadPlan, Touch,
    disable_pass, info_renderable, manual_block, override_line, prior_touch, refused, touch_after_stop, touch_from,
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

NOT_APPLIED = "{changes} saved, not applied: {why}. {fix}"
NOT_APPLIED_CASES = MappingProxyType({
    "asks": ("a restart would drop {asks}", "Answer the asks {where}, then open /plugins and close it"),
    "closed_early": ("Plugins closed on an error: {error}", "Quit and start localharness again to apply"),
})
CLOSED_ON_ERROR = "Plugins closed on an error: {error}"

TURN_SEVERAL_ON = "Turn several on…"
RESULT = "{name}: {outcome}"
RESULT_OUTCOMES: dict[Kind, str] = {
    "enabled": "enabled{starts}",
    "enabled_unconfigured": "{lead}not set up — {detail}; {rerun} to finish{then}",
    "check_failed": "enabled, but its check failed: {detail}{starts}",
    "check_failed_noload": "enabled, but not loading — {detail}; {rerun} to retry{stops}",
    "check_skipped": "enabled — {detail}{starts}",
    "enabled_not_loading": "enabled — not loading: {detail}{stops}",
    "rerun_ok": "setup re-run — check passed{rebound}",
    "rerun_failed": "setup re-run — check failed: {detail}{rebound}",
    "rerun_blocked": "setup not re-run — {detail}",
    "disabled": "off{later}",
    "saved": "{key} saved — applies {applies}",
    "saved_overridden": "{key} saved on this machine, but this project's {file} still wins here",
    "override": OVERRIDE_LINE,
    "declined": "stays off",
    "refused": "settings not changed — {detail}{installed}",
}
INSTALLED = " (package installed — {again} to finish)"
STOP_LINE = "{names}: stopped — {tail}"
AUDIT_MISSING = "; the audit entry may be missing"
STOP_TAILS = MappingProxyType({
    "consent": "nothing changed",
    "install": "install interrupted — {again} to retry",
    "install_batch": f"install interrupted — pick them again under {TURN_SEVERAL_ON}",
    "question": "nothing changed",
    "question_installed": "package installed, not set up — {again} to finish",
    "download": "enabled, but its download did not finish — {rerun}{retry}",
    "check": "enabled, but not checked — {rerun}{retry}",
    "edit": "nothing changed",
    "edit_written": "{key} saved — applies {applies}" + AUDIT_MISSING,
    "disable": "nothing changed",
})
STOPPING = "{names}: stopping — pressing Ctrl-C two more times quits localharness"

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


def _applies(loads: bool, plan: ReloadPlan) -> str:
    return at_close(plan) if loads else "when it starts"


def _clean(text: str) -> str:
    return redact_text(sanitize_for_display(text))


def result_line(name: str, result: PassResult, session: PluginSession, plan: ReloadPlan, progress: PassProgress,
                *, key: str = "") -> str | None:
    """What a returned pass says happened, or None when its own block was the result."""
    if result.kind not in RESULT_OUTCOMES:
        return None
    if result.kind == "override":
        return override_line(name, result.override, OVERRIDE_SAVED if progress.wrote else "")
    running, named, ac = name in session.running, name in plan.names, at_close(plan)
    detail = _clean(result.detail or result.step.failed_check or result.step.skipped_check)
    if result.kind == "enabled_unconfigured":
        detail = detail.removeprefix("unconfigured — ") or "setup questions skipped"
    stops = f"; stops {ac}" if running else ""
    return RESULT_OUTCOMES[result.kind].format(
        starts=f"; applies {ac}" if running and named else "; already running" if running else f"; starts {ac}",
        stops=stops, detail=detail, rerun=ACTION_RERUN, key=key,
        lead="" if running else "enabled, ", then=stops or ", then it starts",
        rebound=f"; applies {ac}" if running and named else f"; starts {ac}" if not running and result.loads else "",
        later=f" {ac}" if running else "", applies=_applies(result.loads, plan),
        file=result.override[1] if result.override else "",
        installed=INSTALLED.format(again=ACTION_RERUN if progress.written else ACTION_ENABLE)
        if result.kind == "refused" and result.install == "done" else "")


def stop_tail(names: tuple[str, ...], progress: PassProgress, session: PluginSession, plan: ReloadPlan,
              *, in_batch: bool = False, key: str = "", file: Any = None) -> str:
    """The state a stop leaves behind, one sentence chosen by the phase it landed in."""
    from localharness.plugins import lifecycle

    phase = progress.phase
    if phase == "edit" and progress.wrote and progress.overridden:
        return RESULT_OUTCOMES["saved_overridden"].format(key=key, file=file) + AUDIT_MISSING
    tail_key = "install_batch" if phase == "install" and in_batch else "edit_written" if phase == "edit" and progress.wrote else phase
    touch = session.touched.get(names[0]) or prior_touch(session, names[0])
    again = {"install": ACTION_INSTALL, "question_installed": ACTION_RERUN}.get(phase, ACTION_ENABLE)
    return STOP_TAILS[tail_key].format(
        again=again if progress.written else ACTION_ENABLE, rerun=ACTION_RERUN, retry=" in a minute" if lifecycle.downloading(names[0]) else "", key=key,
        applies=_applies(touch.loads, plan))


def stopping_line(console: Any, names: tuple[str, ...], progress: PassProgress) -> None:
    if progress.phase == "install":
        _say(console, STOPPING.format(names=", ".join(names)))


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


async def guarded(flow: Flow, node: Callable, *, names: tuple[str, ...] = (), progress: Any = None,
                  in_batch: bool = False, key: str = "", file: Any = None) -> Any:
    """The node's value, or `_STOP` for a user stop; an error the node let out raises `_Abandon`."""
    seam = progress is not None and bool(names)
    priors = {n: flow.session.touched.get(n) or prior_touch(flow.session, n) for n in names} if seam else {}
    task = asyncio.ensure_future(_contained(flow, node, names, progress))
    on_stopping = partial(stopping_line, flow.console, names, progress) if seam else None
    async with sigint_cancels(task, on_stopping=on_stopping) as cancelled:
        try:
            got = await task
        except asyncio.CancelledError:
            if not cancelled():
                task.cancel()
                raise
            got = _STOP
    if got is _STOP and seam:
        _stopped(flow, names, progress, priors, in_batch=in_batch, key=key, file=file)
    if isinstance(got, _Escape):
        raise _Abandon(error=got.error)
    return got


def _stopped(flow: Flow, names: tuple[str, ...], progress: PassProgress, priors: Mapping[str, Touch], **tail: Any) -> None:
    """Record what the stopped pass wrote, recompute the plan from it, then say what state the stop left."""
    for name in names:
        if (touch := touch_after_stop(flow.session, name, progress, priors[name])) is not None:
            flow.session.touched[name] = touch
    flow.plan = flow.should_reload(flow.session.touched)
    _say(flow.console, STOP_LINE.format(names=", ".join(names), tail=stop_tail(names, progress, flow.session, flow.plan, **tail)))


async def _act(flow: Flow, entry: Any, node: Callable, progress: PassProgress, *, in_batch: bool = False, key: str = "",
               file: Any = None) -> Any:
    """Run a seam node, store the touch its result leaves, recompute the plan and print the result line."""
    name = entry.name
    prior = flow.session.touched.get(name) or prior_touch(flow.session, name)
    result = await guarded(flow, node, names=(name,), progress=progress, in_batch=in_batch, key=key, file=file)
    if result is _STOP:
        return result
    if (touch := touch_from(flow.session, name, result, prior)) is not None:
        flow.session.touched[name] = touch
    flow.plan = flow.should_reload(flow.session.touched)
    if (outcome := result_line(name, result, flow.session, flow.plan, progress, key=key)) is not None:
        _say(flow.console, RESULT.format(name=name, outcome=outcome))
    return result


async def _ask_root(flow: Flow) -> Any:
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    session = flow.session
    entries = resolve(ConfigLoader(config_dir=session.cfg_path, local_config_dir=session.workspace)).plan.entries
    flow.plan = plan = flow.should_reload(session.touched)
    options = [Option(str(i), entry.name, _row_state(entry, session, plan)) for i, entry in enumerate(entries)]
    options.append(Option(CLOSE, _close_label(plan)))
    answer = await flow.channel.run_interview(PLUGINS_QUESTION, options, default=0, cancel=CLOSE)
    return answer, [entry.name for entry in entries]


async def _root(flow: Flow) -> None:
    while True:
        got = await guarded(flow, partial(_ask_root, flow))
        if got is _STOP:
            return
        answer, names = got
        if not answer.reached:
            raise _Abandon(why=answer.why)
        if answer.cancelled or answer.keys == (CLOSE,):
            return
        await _menu(flow, names[int(answer.keys[0])])


def _no_report(kind: str, payload: Any) -> None:
    return None


async def _show_install(flow: Flow, entry: Any, route: Any) -> None:
    _say(flow.console, ENABLE_MANUAL.format(name=entry.name), *manual_block([entry.name], route.manual, ROUTE_BACK))


async def _disable(flow: Flow, entry: Any, route: Any) -> None:
    progress = PassProgress(phase="disable")

    async def node() -> PassResult:
        return await disable_pass(flow.session, entry.name, progress=progress, report=_no_report)

    await _act(flow, entry, node, progress)


async def _info(flow: Flow, entry: Any, route: Any) -> bool:
    flow.console.print(info_renderable(flow.session, entry), soft_wrap=True)
    return True


def _manual_route(entry: Any, enabled: bool | None, route: Any) -> bool:
    return entry.state == "needs-extra" and route is not None and route.manual is not None


ACTIONS: tuple[tuple[str, str, Callable[[Any, bool | None, Any], bool], Callable[[Flow, Any, Any], Awaitable[bool | None]]], ...] = (
    ("show_install", ACTION_SHOW_INSTALL, _manual_route, _show_install),
    ("disable", ACTION_DISABLE, lambda entry, enabled, route: enabled is True, _disable),
    ("info", ACTION_INFO, lambda entry, enabled, route: True, _info),
)


def actions_for(entry: Any, session: PluginSession, route: Any) -> list[tuple[Option, Callable]]:
    """The menu rows a plugin's state allows, each with its handler; a handler answering True keeps the menu open."""
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    enabled = resolve(ConfigLoader(config_dir=session.cfg_path, local_config_dir=session.workspace)).enabled.get(entry.name)
    return [(Option(key, label), handler) for key, label, available, handler in ACTIONS if available(entry, enabled, route)]


async def _ask_menu(flow: Flow, name: str) -> Any:
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    session = flow.session
    entry = resolve(ConfigLoader(config_dir=session.cfg_path, local_config_dir=session.workspace)).plan.entry(name)
    if entry is None:
        return None
    flow.plan = flow.should_reload(session.touched)
    route = plugins_cmd.install_route_for(entry)
    actions = actions_for(entry, session, route)
    question = f"{sanitize_for_display(name)} — {_row_state(entry, session, flow.plan)} (Esc = back)"
    answer = await flow.channel.run_interview(question, [option for option, _ in actions], default=0, cancel="back")
    return answer, entry, route, {option.key: handler for option, handler in actions}


async def _menu(flow: Flow, name: str) -> None:
    while True:
        got = await guarded(flow, partial(_ask_menu, flow, name))
        if got is _STOP or got is None:
            return
        answer, entry, route, handlers = got
        if not answer.reached:
            raise _Abandon(why=answer.why)
        if answer.cancelled or not await handlers[answer.keys[0]](flow, entry, route):
            return


def _not_applied(plan: ReloadPlan, why: Unreachable | None, error: str) -> str:
    """The one line for a change that was saved but cannot be applied: an unreached node, then asks, then an error."""
    if why is not None:
        reason, fix = UNREACHABLE[why][1:]
    elif plan.parked or plan.waiting:
        reason, fix = (t.format(asks=n_of(plan.parked + plan.waiting, "ask"), where=_where(plan))
                       for t in NOT_APPLIED_CASES["asks"])
    else:
        reason, fix = (t.format(error=error) for t in NOT_APPLIED_CASES["closed_early"])
    return NOT_APPLIED.format(changes=n_of(len(plan.names), "plugin change"), why=reason, fix=fix)


def _finish(flow: Flow, *, closing: bool, why: Unreachable | None = None, error: str = "") -> bool:
    plan = flow.should_reload(flow.session.touched)
    if closing and plan.reload:
        flow.close()
        return True
    if plan.names:
        _say(flow.console, _not_applied(plan, why, error))
    elif why is not None:
        _say(flow.console, unavailable(why))
    elif error:
        _say(flow.console, CLOSED_ON_ERROR.format(error=error))
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
        except _Abandon as gone:
            return _finish(flow, closing=False, why=gone.why, error=gone.error)
        except Exception as exc:  # noqa: BLE001 — Plugins closes; the session goes on
            return _finish(flow, closing=False, error=_first_line(exc))
        return _finish(flow, closing=True)
