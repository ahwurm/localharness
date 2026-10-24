"""The Plugins flow: /plugins on a terminal, drawn inside one suspension of the input box."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, get_args, get_origin

import click
from rich.text import Text

from localharness.channels.base import sanitize_for_display
from localharness.cli import plugins_cmd
from localharness.cli.components_cmd import redact_text, shown, userinfo_present, workspace_file_of
from localharness.cli.interview import Option, Unreachable, confirm_line, n_of, prompt_line
from localharness.cli.plugins_cmd import (
    ACTION_CONFIGURE, ACTION_DISABLE, ACTION_ENABLE, ACTION_INFO, ACTION_INSTALL, ACTION_RERUN, ACTION_SHOW_INSTALL, CTRLC, CTRLC_DOES,
    ENABLE_MANUAL, OVERRIDE_LINE, OVERRIDE_SAVED, ROUTE_BACK, Kind, PassProgress, PassResult, PluginSession, Refusal, ReloadPlan, Touch,
    disable_pass, enable_pass, info_renderable, install_pass, manual_block, override_line, prior_touch, refused, touch_after_stop,
    touch_from,
)
from localharness.config.redact import is_secret
from localharness.core.interrupt import sigint_cancels
from localharness.plugins.api import SetupField
from localharness.registry.paths import unwrap_optional

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

SEVERAL = "+several"
TURN_SEVERAL_ON = "Turn several on…"
TURN_SEVERAL_ON_HINT = "turn on more than one at once"
TURN_SEVERAL_ON_WHICH = "Turn on which? (Esc = back)"
BATCH_PROGRESS = "{i} of {m}: {name}"
BATCH_STOPPED = "not started: {names}{more} — enable them from the list"
BATCH_NOTHING_LEFT = "nothing left to start"
BATCH_MORE = " and {k} more"
BATCH_SUMMARY = "{done} of {m} enabled{still}{restart}"
BATCH_STILL = ", install still needed: {k}"
BATCH_RESTART = "; restarts {at_close}"
BATCH_ENABLED = frozenset({
    "enabled", "enabled_unconfigured", "check_failed", "check_failed_noload", "check_skipped", "enabled_manual",
    "enabled_not_loading"})
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
RERUN_HINT = "download still stopping"
RELOAD_WAITS = (
    "{name}: download still stopping — the restart waits for it.",
    "Ctrl-C quits localharness now instead (a stuck download may need a second Ctrl-C); saved changes are kept.")
CONFIGURE_QUESTION = "{name} — settings (Esc = back)"
BOOL_QUESTION = "{key}: on or off?"
LITERAL_QUESTION = "{key}: which?"
SCALAR_PROMPT = "{key} [{shown}]"
READ_ONLY = "read-only here — edit {path} in {file}"
DONE = "Done"
NEXT_STEPS_KINDS = frozenset({
    "enabled", "enabled_unconfigured", "check_failed", "check_failed_noload", "check_skipped", "enabled_not_loading",
    "rerun_ok", "rerun_failed", "install_skipped", "enabled_manual"})
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
    except _Abandon:
        raise
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
        if seam:
            _record(flow, names, progress, priors)
        raise _Abandon(error=got.error)
    return got


def _record(flow: Flow, names: tuple[str, ...], progress: PassProgress, priors: Mapping[str, Touch]) -> None:
    """Store what the ended pass had written and recompute the plan from it."""
    for name in names:
        if (touch := touch_after_stop(flow.session, name, progress, priors[name])) is not None:
            flow.session.touched[name] = touch
    flow.plan = flow.should_reload(flow.session.touched)


def _stopped(flow: Flow, names: tuple[str, ...], progress: PassProgress, priors: Mapping[str, Touch], **tail: Any) -> None:
    """Record what the stopped pass wrote, then say what state the stop left."""
    _record(flow, names, progress, priors)
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
    if result.kind == "enabled_manual":
        _manual(flow.console, name, plugins_cmd.install_route_for(entry))
    if result.kind in NEXT_STEPS_KINDS:
        steps = result.next_steps.splitlines()
        _say(flow.console, *(line for paragraph in result.help for line in ("", paragraph)),
             *(("", *steps) if result.help and steps else steps))
    return result


def _loaded(session: PluginSession) -> tuple[Any, Any]:
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    loader = ConfigLoader(config_dir=session.cfg_path, local_config_dir=session.workspace)
    return loader, resolve(loader)


def _resolved(session: PluginSession) -> tuple[Any, tuple[Any, ...]]:
    loader, resolution = _loaded(session)
    return loader, tuple(resolution.plan.entries)


def _switchable_on(entry: Any, loader: Any) -> bool:
    route = plugins_cmd.install_route_for(entry)
    installable = route is not None and route.argv is not None
    return (entry.state in ("off", "available", "needs-extra") and (installable or route is None and entry.state != "needs-extra")
            and plugins_cmd.override_of(loader, entry, True, False) is None)


async def _ask_root(flow: Flow) -> Any:
    session = flow.session
    loader, entries = _resolved(session)
    flow.plan = plan = flow.should_reload(session.touched)
    options = [Option(str(i), entry.name, _row_state(entry, session, plan)) for i, entry in enumerate(entries)]
    several = sum(_switchable_on(entry, loader) for entry in entries) >= 2
    if several:
        options.insert(0, Option(SEVERAL, TURN_SEVERAL_ON, TURN_SEVERAL_ON_HINT))
    options.append(Option(CLOSE, _close_label(plan)))
    answer = await flow.channel.run_interview(PLUGINS_QUESTION, options, default=int(several), cancel=CLOSE)
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
        if answer.keys == (SEVERAL,):
            await _several(flow)
        else:
            index = int(answer.keys[0])
            await _menu(flow, index, names[index])


def _no_report(kind: str, payload: Any) -> None:
    return None


def _manual(console: Any, name: str, route: Any) -> None:
    _say(console, ENABLE_MANUAL.format(name=name), *manual_block([name], route.manual, ROUTE_BACK))


async def _show_install(flow: Flow, entry: Any, route: Any) -> None:
    _manual(flow.console, entry.name, route)


def _pass_handler(mode: Literal["enable", "rerun"]) -> Callable[[Flow, Any, Any], Awaitable[None]]:
    async def handler(flow: Flow, entry: Any, route: Any) -> None:
        progress = PassProgress()

        async def node() -> PassResult:
            return await enable_pass(flow.session, entry.name, mode=mode, progress=progress, prompt=flow.prompt,
                                     confirm=flow.confirm, report=_no_report, install="own", console=flow.console)

        await _act(flow, entry, node, progress)

    return handler


async def _disable(flow: Flow, entry: Any, route: Any) -> None:
    progress = PassProgress(phase="disable")

    async def node() -> PassResult:
        return await disable_pass(flow.session, entry.name, progress=progress, report=_no_report)

    await _act(flow, entry, node, progress)


async def _info(flow: Flow, entry: Any, route: Any) -> bool:
    flow.console.print(info_renderable(flow.session, entry), soft_wrap=True)
    return True


def _key(row: Any) -> str:
    return row.path.removeprefix(f"{row.plugin}.")


def _value(row: Any) -> str:
    return _clean(shown(row.current_value, row.annotation))


async def _read_only(flow: Flow, row: Any, loader: Any) -> None:
    _say(flow.console, READ_ONLY.format(path=row.path, file=loader.user_overlay_path))


async def _secret(flow: Flow, row: Any, loader: Any) -> str | None:
    key, value, secret = _key(row), row.current_value, is_secret(row.annotation)
    pairs = await plugins_cmd.ask_fields(
        (SetupField(key=key, prompt=key, secret=secret),), lambda k: str(value), lambda k: bool(value),
        lambda k: "none" if unwrap_optional(row.annotation) is not row.annotation else "", prompt=flow.prompt)
    return pairs[0].removeprefix(f"{key}=") if pairs else None


async def _pick(flow: Flow, question: str, choices: tuple[str, ...], current: str) -> str | None:
    answer = await flow.channel.run_interview(
        question, [Option(c, c) for c in choices], default=choices.index(current) if current in choices else 0, cancel="back")
    if not answer.reached:
        raise _Abandon(why=answer.why)
    return None if answer.cancelled else answer.keys[0]


async def _bool(flow: Flow, row: Any, loader: Any) -> str | None:
    return await _pick(flow, BOOL_QUESTION.format(key=_key(row)), ("on", "off"), "on" if row.current_value else "off")


async def _literal(flow: Flow, row: Any, loader: Any) -> str | None:
    choices = tuple(map(str, get_args(unwrap_optional(row.annotation))))
    return await _pick(flow, LITERAL_QUESTION.format(key=_key(row)), choices, str(row.current_value))


async def _scalar(flow: Flow, row: Any, loader: Any) -> str | None:
    key = _key(row)
    text = SCALAR_PROMPT.format(key=key, shown=_value(row)) + ": "
    if (raw := await flow.prompt(text, default="", secret=key.endswith("url"))) is None:
        raise click.Abort()
    return raw or None


EDITORS: tuple[tuple[Callable[[Any], bool], Callable[[Flow, Any, Any], Awaitable[str | None]]], ...] = (
    (lambda row: get_origin(unwrap_optional(row.annotation)) not in (None, Literal), _read_only),
    (lambda row: is_secret(row.annotation) or userinfo_present(row.current_value), _secret),
    (lambda row: unwrap_optional(row.annotation) is bool, _bool),
    (lambda row: get_origin(unwrap_optional(row.annotation)) is Literal, _literal),
    (lambda row: unwrap_optional(row.annotation) in (str, int, float), _scalar),
    (lambda row: True, _read_only),
)


async def _edit(flow: Flow, entry: Any, loader: Any, row: Any) -> None:
    editor = next(handler for matches, handler in EDITORS if matches(row))
    key, file, progress = _key(row), workspace_file_of(loader, row.winning_layer), PassProgress(phase="edit")

    async def write() -> PassResult:
        return await plugins_cmd.configure_one(flow.session, row, raw, progress=progress, report=_no_report)

    while True:
        raw = await guarded(flow, partial(editor, flow, row, loader), names=(entry.name,), progress=progress)
        if raw is _STOP or raw is None:
            return
        result = await _act(flow, entry, write, progress, key=key, file=file)
        if result is _STOP or result.kind != "refused":
            return


async def _ask_settings(flow: Flow, name: str) -> Any:
    loader, resolution = _loaded(flow.session)
    entry = resolution.plan.entry(name)
    rows = plugins_cmd.settings_rows(resolution, loader, entry) if entry is not None else ()
    if not rows:
        return None
    options = [Option(str(i), _key(row), f"{_value(row)} · {row.winning_layer}") for i, row in enumerate(rows)]
    answer = await flow.channel.run_interview(
        CONFIGURE_QUESTION.format(name=sanitize_for_display(name)), [*options, Option("done", DONE)], default=0, cancel="done")
    return answer, loader, rows


async def _configure(flow: Flow, entry: Any, route: Any) -> bool:
    while True:
        got = await guarded(flow, partial(_ask_settings, flow, entry.name))
        if got is _STOP or got is None:
            return True
        answer, loader, rows = got
        if not answer.reached:
            raise _Abandon(why=answer.why)
        if answer.cancelled or answer.keys == ("done",):
            return True
        await _edit(flow, entry, loader, rows[int(answer.keys[0])])


def _argv_route(entry: Any, enabled: bool | None, route: Any, session: PluginSession, settings: bool) -> bool:
    return entry.state == "needs-extra" and route is not None and route.argv is not None


def _manual_route(entry: Any, enabled: bool | None, route: Any, session: PluginSession, settings: bool) -> bool:
    return entry.state == "needs-extra" and route is not None and route.manual is not None


def _rerun_offered(entry: Any, enabled: bool | None, route: Any, session: PluginSession, settings: bool) -> bool:
    touch, m = session.touched.get(entry.name), entry.manifest
    return entry.state == "on" and (
        bool(m is not None and (m.setup or m.setup_action)) or entry.name not in session.running
        or (touch is not None and touch.probe is not None and touch.probe[0] != "on"))


ACTIONS: tuple[tuple[str, str, Callable[[Any, bool | None, Any, PluginSession, bool], bool],
                     Callable[[Flow, Any, Any], Awaitable[bool | None]]], ...] = (
    ("enable", ACTION_ENABLE, lambda entry, enabled, route, session, settings: entry.state in ("off", "available"),
     _pass_handler("enable")),
    ("install", ACTION_INSTALL, _argv_route, _pass_handler("enable")),
    ("show_install", ACTION_SHOW_INSTALL, _manual_route, _show_install),
    ("configure", ACTION_CONFIGURE, lambda entry, enabled, route, session, settings: settings, _configure),
    ("rerun", ACTION_RERUN, _rerun_offered, _pass_handler("rerun")),
    ("disable", ACTION_DISABLE, lambda entry, enabled, route, session, settings: enabled is True, _disable),
    ("info", ACTION_INFO, lambda entry, enabled, route, session, settings: True, _info),
)


def actions_for(entry: Any, session: PluginSession, route: Any) -> list[tuple[Option, Callable]]:
    """The menu rows a plugin's state allows, each with its handler; a handler answering True keeps the menu open."""
    from localharness.plugins import lifecycle

    loader, resolution = _loaded(session)
    enabled = resolution.enabled.get(entry.name)
    settings = entry.state == "on" and bool(plugins_cmd.settings_rows(resolution, loader, entry))
    hint = RERUN_HINT if lifecycle.downloading(entry.name) else ""
    return [(Option(key, label, hint if key == "rerun" else ""), handler)
            for key, label, available, handler in ACTIONS if available(entry, enabled, route, session, settings)]


async def _ask_menu(flow: Flow, index: int, name: str) -> Any:
    session = flow.session
    entries = _resolved(session)[1]
    entry = entries[index] if index < len(entries) and entries[index].name == name else None
    if entry is None:
        return None
    flow.plan = flow.should_reload(session.touched)
    route = plugins_cmd.install_route_for(entry)
    actions = actions_for(entry, session, route)
    question = f"{sanitize_for_display(name)} — {_row_state(entry, session, flow.plan)} (Esc = back)"
    answer = await flow.channel.run_interview(question, [option for option, _ in actions], default=0, cancel="back")
    return answer, entry, route, {option.key: handler for option, handler in actions}


async def _menu(flow: Flow, index: int, name: str) -> None:
    while True:
        got = await guarded(flow, partial(_ask_menu, flow, index, name))
        if got is _STOP or got is None:
            return
        answer, entry, route, handlers = got
        if not answer.reached:
            raise _Abandon(why=answer.why)
        if answer.cancelled or not await handlers[answer.keys[0]](flow, entry, route):
            return


def _not_started(names: list[str]) -> str:
    if not names:
        return BATCH_NOTHING_LEFT
    more = BATCH_MORE.format(k=len(names) - 3) if len(names) > 3 else ""
    return BATCH_STOPPED.format(names=", ".join(names[:3]), more=more)


def _summary(flow: Flow, picks: list[Any], results: list[tuple[str, PassResult | None]]) -> str:
    kinds = [result.kind for _, result in results if result is not None]
    still = kinds.count("install_skipped")
    return BATCH_SUMMARY.format(
        done=sum(kind in BATCH_ENABLED for kind in kinds), m=len(picks),
        still=BATCH_STILL.format(k=still) if still else "",
        restart=BATCH_RESTART.format(at_close=at_close(flow.plan)) if flow.plan.names else "")


async def _enable_pick(flow: Flow, entry: Any, progress: PassProgress, install: str) -> PassResult:
    return await enable_pass(flow.session, entry.name, mode="enable", progress=progress, prompt=flow.prompt,
                             confirm=flow.confirm, report=_no_report, install=install, in_batch=True, console=flow.console)


async def _install_batch(flow: Flow, picks: list[Any], needs: list[Any]) -> Any:
    progress = PassProgress()

    async def node() -> frozenset[str]:
        return await install_pass(flow.session, {plugins_cmd.extra_for(entry) for entry in needs}, [e.name for e in needs],
                                  progress=progress, confirm=flow.confirm, console=flow.console, in_batch=True)

    return await guarded(flow, node, names=tuple(e.name for e in picks), progress=progress, in_batch=True)


async def _batch(flow: Flow, picks: list[Any]) -> None:
    _say(flow.console, CTRLC.format(does=CTRLC_DOES["batch"]))
    needs = [entry for entry in picks if plugins_cmd.extra_for(entry)]
    covered = await _install_batch(flow, picks, needs) if needs else frozenset()
    results: list[tuple[str, PassResult | None]] = []
    if covered is _STOP:
        _say(flow.console, _not_started([entry.name for entry in picks]))
    else:
        for i, entry in enumerate(picks, 1):
            _say(flow.console, BATCH_PROGRESS.format(i=i, m=len(picks), name=entry.name))
            install = "done" if entry.name in covered else "skipped" if entry in needs else "none"
            progress = PassProgress()
            result = await _act(flow, entry, partial(_enable_pick, flow, entry, progress, install), progress, in_batch=True)
            results.append((entry.name, None if result is _STOP else result))
            if result is _STOP:
                _say(flow.console, _not_started([e.name for e in picks[i:]]))
                break
    _say(flow.console, _summary(flow, picks, results))


async def _several(flow: Flow) -> None:
    loader, entries = _resolved(flow.session)
    pool = [entry for entry in entries if _switchable_on(entry, loader)]
    options = [Option(str(i), entry.name, _row_state(entry, flow.session, flow.plan)) for i, entry in enumerate(pool)]
    got = await guarded(flow, lambda: flow.channel.run_interview(
        TURN_SEVERAL_ON_WHICH, options, default=frozenset(), cancel="back", multi=True))
    if got is _STOP:
        return
    if not got.reached:
        raise _Abandon(why=got.why)
    if picks := [entry for option, entry in zip(options, pool) if option.key in got.keys and not got.cancelled]:
        await _batch(flow, picks)


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
    from localharness.plugins import lifecycle

    plan = flow.should_reload(flow.session.touched)
    if closing and plan.reload:
        flow.close()
        for name in sorted(flow.session.touched):
            if lifecycle.downloading(name):
                _say(flow.console, *(line.format(name=name) for line in RELOAD_WAITS))
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
