"""The plugin lifecycle (PAPI-02, PAPI-03, PAPI-11): the fixed order, stage-wise, contained.

start_plugins takes the plugins a Resolution put ON and runs them through four stages — create the
plugin, configure, tools, start — each stage over the whole of the plan's dependency order before the
next stage begins; stop_plugins stops them in reverse start order. A plugin that ships with
LocalHarness and one you installed go through this same path. The one difference is SAFE-06's gate
clamp, applied when a plugin you installed registers its tools (plugins/trust.py).

Containment: every call into plugin code runs inside `except (Exception, SystemExit)`. A plugin that
raises at any stage (sys.exit() included), or hands back something core cannot use, is failed for
the session: it is named in a warning ("plugin <name>: <reason>"), the tools it registered are
unregistered, the hook objects it put on ctx.hooks are unregistered, and every plugin that
`requires` it is failed too. The harness never exits because of a plugin; KeyboardInterrupt and
cancellation still propagate.

Around the stages, core also decides: each plugin's context, every path in it computed by core
(the artifact root included); which artifact roots are served (only the one core computed); who sits
in the memory slot; the plugins' slash rows, each importing its target only when the command first
runs; and, for doctor, every plugin's row without a session.

Named residuals — what containment does NOT cover:
- Hooks. A pluggy hook implementation a failed plugin registered on ctx.hooks is unregistered with
  it. A raw ToolRegistry hook (ctx.tools.register_pre_hook / register_post_hook) is not, and stays
  registered for the session; a hook registered without `name=` is reported by object id and module
  (tools/hooks.py). Hooks belong on ctx.hooks, registered under the plugin's name.
- A tool a plugin registers itself (ctx.tools.register) instead of returning it from tools() carries
  no source_plugin, escapes SAFE-06's clamp and is not unregistered if the plugin fails. Core judges
  what flows through tools(); plugin code is trusted code (SAFE-05).
- A stage that never returns holds up the session's start: containment covers exceptions, not hangs.
- A plugin whose start() raised is not stopped: whatever it acquired before raising stays acquired.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from localharness.agent.verdict import NAME_CLASSIFIED_TOOLS
from localharness.cli.slash_commands import SlashCommand
from localharness.core.artifacts import artifact_root
from localharness.core.bus import EventBus
from localharness.plugins.api import (
    Check, MemorySlotPlugin, Plugin, PluginContext, PluginPaths, SessionInfo,
)
from localharness.plugins.discovery import import_target
from localharness.plugins.slot import MemorySlot
from localharness.plugins.trust import third_party_overrides
from localharness.tools.base import ToolProtocol, ToolSchema
from localharness.tools.hooks import HookSystem
from localharness.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from localharness.plugins.resolve import Resolution
    from localharness.provider.client import LLMClient

log = logging.getLogger(__name__)


@dataclass
class RunningPlugin:
    """A plugin that got through start() this session: its instance, its context, whether it ships
    with LocalHarness, and the names of the tools it registered."""

    name: str
    plugin: Plugin
    ctx: PluginContext
    bundled: bool
    tool_names: tuple[str, ...] = ()


@dataclass
class LifecycleResult:
    """What start_plugins did. `running` is in start order; `failed` maps a plugin to its reason and
    `unconfigured` to the setting it is missing; `warnings` are the lines to show; `artifact_roots`
    the roots core accepted; `slot` the memory slot (empty unless the plan's occupant is running);
    `slash_rows` the running plugins' rows for the one slash table."""

    running: list[RunningPlugin] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    unconfigured: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    artifact_roots: dict[str, Path] = field(default_factory=dict)
    slot: MemorySlot = field(default_factory=MemorySlot)
    slash_rows: tuple[SlashCommand, ...] = ()

    @property
    def loaded_names(self) -> list[str]:
        return [r.name for r in self.running]


class _Refused(Exception):
    """Core refuses what a plugin handed it; the message is the whole reason."""


def _what(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


async def _call(fn: Callable[..., Any], *args: Any) -> Any:
    """Call a plugin method, awaiting it if it is async (a plain def is taken as written)."""
    value = fn(*args)
    return await value if inspect.isawaitable(value) else value


def _availability(got: Any) -> tuple[str, str]:
    """configure()'s answer as (state, detail): ("ready", ""), ("unconfigured", "<setting>"), or
    ("failed", why) for anything else — a configure() that forgot to return says so."""
    if isinstance(got, str) and got == "ready":
        return "ready", ""
    if isinstance(got, tuple) and len(got) == 2 and got[0] == "unconfigured" and isinstance(got[1], str):
        return "unconfigured", got[1]
    return "failed", f'configure() returned {got!r:.80}, not "ready" or ("unconfigured", "<setting>")'


def _checked(tools: Any, registry: ToolRegistry) -> list[tuple[ToolProtocol, ToolSchema]]:
    """What tools() returned, each tool with its schema read once — or _Refused, and then none of
    them is registered: not a list of tools, one name used twice, a name the permission gate
    classifies by name (NAME_CLASSIFIED_TOOLS), or the name of a tool already registered."""
    if not isinstance(tools, (list, tuple)):
        raise _Refused(f"tools() returned {type(tools).__name__}, not a list of tools")
    pairs = []
    for tool in tools:
        schema = tool.info() if isinstance(tool, ToolProtocol) else None
        if not isinstance(schema, ToolSchema):
            raise _Refused(f"tools() returned {tool!r:.80}, which is not a tool (a tool has info() "
                           "returning a ToolSchema, and an async run())")
        pairs.append((tool, schema))
    names = [schema.name for _, schema in pairs]
    for name in names:
        if names.count(name) > 1:
            raise _Refused(f"it contributes two tools named {name!r}")
        if name in NAME_CLASSIFIED_TOOLS:
            raise _Refused(f"its tool {name!r} has a name the permission gate classifies by name, "
                           "not by what the tool declares — rename it")
        if registry.schema_of(name) is not None:
            raise _Refused(f"its tool {name!r} has the name of a tool already registered")
    return pairs


def plugin_context(resolution: Resolution, name: str, *, bus: EventBus, registry: ToolRegistry,
                   hooks: HookSystem | None, llm: LLMClient | None,
                   paths: PluginPaths, session: SessionInfo | None = None) -> PluginContext:
    """`name`'s context (PAPI-03): the session's bus, tool registry, hooks and LLM client, its own
    validated settings, and the session's `paths` with the artifact root core computes —
    `<state_dir>/artifacts/<name>/` — iff the plugin wants artifacts (PAPI-10), else None."""
    settings = resolution.settings[name]
    wants = resolution.classes[name].wants_artifacts
    idle = None
    if llm is not None:  # lazy: provider/__init__ eagerly imports the httpx client
        from localharness.provider.idle_llm import LLMTextAdapter
        idle = LLMTextAdapter(llm)
    return PluginContext(
        bus=bus, tools=registry, hooks=hooks, config=settings.config,
        agent_config=settings.agent_config,
        paths=replace(paths, artifact_dir=artifact_root(paths.state_dir, name) if wants else None),
        llm=llm, idle_llm=idle, session=session)


async def start_plugins(resolution: Resolution, *, bus: EventBus, registry: ToolRegistry,
                        hooks: HookSystem | None, llm: LLMClient | None,
                        paths: PluginPaths, session: SessionInfo | None = None) -> LifecycleResult:
    """Run the ON plugins through create → configure → tools → start, each stage over the plan's
    order, and return what happened; never raises because of a plugin. `paths` is the SESSION's
    (artifact_dir None) — each plugin's is derived from it. Lines a plugin leaves on an optional
    `startup_warnings` list attribute during start() are added to the result's warnings verbatim.
    An interrupt or cancellation stops every plugin already running, then propagates."""
    result = LifecycleResult()
    live: dict[str, RunningPlugin] = {}     # created, and neither failed nor unconfigured (yet)
    hooked: dict[str, set[object]] = {}     # the hook objects each plugin put on ctx.hooks

    def hook_objects() -> set[object]:
        return set(hooks.pm.get_plugins()) if hooks is not None else set()

    async def disable(name: str) -> None:
        """Take out everything the plugin registered: its tools and its hook objects."""
        rp = live.pop(name, None)
        for tool in rp.tool_names if rp is not None else ():
            await registry.unregister(tool, scope="global")
        for obj in hooked.pop(name, ()):
            if hooks is not None and hooks.pm.is_registered(obj):
                hooks.pm.unregister(obj)

    async def fail(name: str, reason: str, exc: BaseException | None = None) -> None:
        result.failed[name] = reason
        result.warnings.append(f"plugin {name}: {reason}")
        log.warning("plugin %s: %s — disabled for this session", name, reason, exc_info=exc)
        await disable(name)

    async def contained(name: str, stage: str, fn: Callable[..., Any], *args: Any) -> tuple[bool, Any]:
        """(True, what fn returned), or (False, None) once the plugin is failed. Every hook object
        registered during the call is attributed to `name`, whichever way it ends."""
        before = hook_objects()
        try:
            return True, await _call(fn, *args)
        except _Refused as exc:
            reason, error = str(exc), None
        except (Exception, SystemExit) as exc:  # noqa: BLE001 — PAPI-11: never fatal
            reason, error = f"{stage}() raised {_what(exc)}", exc
        finally:
            hooked.setdefault(name, set()).update(hook_objects() - before)
        await fail(name, reason, error)
        return False, None

    async def create(name: str) -> None:
        ok, plugin = await contained(name, "__init__", resolution.classes[name])
        if ok:
            ctx = plugin_context(resolution, name, bus=bus, registry=registry, hooks=hooks,
                                 llm=llm, paths=paths, session=session)
            live[name] = RunningPlugin(name, plugin, ctx, resolution.plan.entry(name).bundled)

    async def configure(name: str) -> None:
        rp = live[name]
        ok, got = await contained(name, "configure", rp.plugin.configure, rp.ctx)
        if not ok:
            return
        state, detail = _availability(got)
        if state == "failed":
            await fail(name, detail)
        elif state == "unconfigured":
            result.unconfigured[name] = detail
            result.warnings.append(f"plugin {name}: unconfigured — set {detail}")
            await disable(name)

    async def contribute(name: str) -> None:
        rp = live[name]

        async def register_all() -> None:
            for tool, schema in _checked(await _call(rp.plugin.tools, rp.ctx), registry):
                overrides, note = ({}, None) if rp.bundled else third_party_overrides(schema)
                if note:
                    result.warnings.append(f"plugin {name}: {note}")
                await registry.register(tool, scope="global", source_plugin=name, overrides=overrides)
                rp.tool_names += (schema.name,)

        await contained(name, "tools", register_all)

    async def start(name: str) -> None:
        rp = live[name]
        ok, _ = await contained(name, "start", rp.plugin.start, rp.ctx)
        lines = getattr(rp.plugin, "startup_warnings", ())  # read once, whichever way start() ended
        if isinstance(lines, (list, tuple)):
            result.warnings.extend(line for line in lines if isinstance(line, str))
        if ok:
            result.running.append(rp)

    def unmet(name: str) -> str | None:
        dep = next((d for d in resolution.plan.entry(name).manifest.requires
                    if d in result.failed or d in result.unconfigured), None)
        if dep is None:
            return None
        return f"requires {dep}, which {'failed' if dep in result.failed else 'is unconfigured'} this session"

    try:
        for stage in (create, configure, contribute, start):
            for name in resolution.plan.order:
                if name in result.failed or name in result.unconfigured:
                    continue
                if (why := unmet(name)) is not None:
                    await fail(name, why)
                else:
                    await stage(name)

        for rp in result.running:  # PAPI-10: only the root core computed is ever served
            expected = rp.ctx.paths.artifact_dir
            if expected is None:
                continue
            try:
                returned = rp.plugin.artifact_root(rp.ctx)
                if returned is not None and Path(returned).resolve() == expected.resolve():
                    result.artifact_roots[rp.name] = expected
                    continue
                why = f"it returned {returned}, the harness computes {expected}"
            except (Exception, SystemExit) as exc:  # noqa: BLE001 — PAPI-11: never fatal
                why = f"its artifact_root() raised {_what(exc)}"
            result.warnings.append(f"plugin {rp.name}: artifact serving disabled this session — {why}")

        occupant = next((rp for rp in result.running if rp.name == resolution.plan.memory_occupant), None)
        if occupant is not None and isinstance(occupant.plugin, MemorySlotPlugin):
            result.slot = MemorySlot(occupant.plugin, occupant.ctx, occupant.name)

        result.slash_rows = tuple(
            SlashCommand(desc.name, desc.help, _slash_handler(desc.target, rp), takes_args=True,
                         plugin=rp.name)
            for rp in result.running for desc in resolution.plan.entry(rp.name).manifest.slash)
    except BaseException:
        # An interrupt or cancellation (which contained() deliberately does not catch) must not
        # leave started plugins open — their stop() closes what they hold (an aiosqlite thread
        # would otherwise hang interpreter exit, #43). The plugin whose start() was interrupted
        # is not in `running`; it cleans up its own half-open state.
        await stop_plugins(result)
        raise
    return result


def _slash_handler(target: str, rp: RunningPlugin) -> Callable[[str], Awaitable[Any]]:
    """A slash row's handler: imports `target` when the command first runs. A target naming a method
    of the running plugin's own class ("pkg.mod:Class.method") is called on THAT instance with
    (ctx, args) — the only way a handler reaches its plugin's live state; any other target is
    imported and awaited with (ctx, args) as before. Additive (no PLUGIN_API_VERSION bump). The REPL
    contains and names its failures (cli/repl.py _run_plugin_slash)."""
    async def handler(args: str) -> Any:
        module, _, attr = target.partition(":")
        owner, dot, method = attr.rpartition(".")
        if dot:
            cls = import_target(f"{module}:{owner}")
            if isinstance(cls, type) and isinstance(rp.plugin, cls):
                return await getattr(rp.plugin, method)(rp.ctx, args)
        return await import_target(target)(rp.ctx, args)
    return handler


async def stop_plugins(result: LifecycleResult) -> None:
    """Stop the running plugins in reverse start order. A stop() that raises is logged under the
    plugin's name and the next plugin is still stopped."""
    for rp in reversed(result.running):
        try:
            await _call(rp.plugin.stop, rp.ctx)
        except (Exception, SystemExit):  # noqa: BLE001 — PAPI-11: never fatal
            log.warning("plugin %s: stop() raised — the other plugins are still stopped", rp.name,
                        exc_info=True)


@dataclass(frozen=True)
class DoctorRow:
    """One plugin in doctor (PAPI-08). `state` is a plan state, or "unconfigured"; `detail` is the
    plan's display text for a plugin that is not on, "unconfigured — set <key>", "failed — <why>" for
    an on plugin that failed to come up, or "" beside an on plugin's `checks`. `bundled` is the plan
    entry's: a bundled plugin waiting on its install extra is information, not a warning."""

    name: str
    state: str
    detail: str
    checks: tuple[Check, ...] = ()
    bundled: bool = False


async def doctor_rows(resolution: Resolution, *, paths: PluginPaths) -> list[DoctorRow]:
    """Every plugin's row, in display order, WITHOUT a session. A plugin that is not on shows its
    plan state (off and available name the enable command). An ON plugin is created and configured
    against a throwaway context — its own settings and paths, a fresh EventBus, ToolRegistry and
    HookSystem, no LLM client — and its doctor() checks are returned; each call is contained, and a
    doctor() that raises is one failing check naming the exception."""
    return [await _doctor_row(resolution, entry.name, paths) if entry.state == "on"
            else DoctorRow(entry.name, entry.state, entry.display, bundled=entry.bundled)
            for entry in resolution.plan.entries]


async def _doctor_row(resolution: Resolution, name: str, paths: PluginPaths) -> DoctorRow:
    ctx = plugin_context(resolution, name, bus=EventBus(), registry=ToolRegistry(),
                         hooks=HookSystem(), llm=None, paths=paths)
    entry = resolution.plan.entry(name)
    bundled = entry.bundled if entry is not None else False
    stage = "__init__"
    try:
        plugin = resolution.classes[name]()
        stage = "configure"
        state, detail = _availability(await _call(plugin.configure, ctx))
    except (Exception, SystemExit) as exc:  # noqa: BLE001 — PAPI-11: never fatal
        return DoctorRow(name, "failed", f"failed — {stage}() raised {_what(exc)}", bundled=bundled)
    if state == "unconfigured":
        return DoctorRow(name, state, f"unconfigured — set {detail}", bundled=bundled)
    if state == "failed":
        return DoctorRow(name, state, f"failed — {detail}", bundled=bundled)
    try:
        checks = await _call(plugin.doctor, ctx)
        if not (isinstance(checks, (list, tuple)) and all(isinstance(c, Check) for c in checks)):
            checks = [Check(name=name, status="fail",
                            detail=f"its doctor check returned {checks!r:.80}, not a list of Check")]
    except (Exception, SystemExit) as exc:  # noqa: BLE001 — PAPI-11: never fatal
        checks = [Check(name=name, status="fail", detail=f"its doctor check raised {_what(exc)}")]
    return DoctorRow(name, "on", "", tuple(checks), bundled=bundled)


async def setup_action_rows(resolution: Resolution, name: str, paths: PluginPaths) -> tuple[Check, ...]:
    """Run NAME's setup_action() once, outside any session, against a throwaway context like
    doctor's (no LLM client). Contained like doctor(): an exception (SystemExit included) or a
    return that is not a list of Check is one failing row naming it. KeyboardInterrupt propagates.
    Called only by `plugins enable` on a terminal — never at start."""
    ctx = plugin_context(resolution, name, bus=EventBus(), registry=ToolRegistry(),
                         hooks=HookSystem(), llm=None, paths=paths)
    try:
        rows = await _call(resolution.classes[name]().setup_action, ctx)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 — PAPI-11: never fatal
        return (Check(name=name, status="fail", detail=f"its setup action raised {_what(exc)}"),)
    if not (isinstance(rows, (list, tuple)) and all(isinstance(c, Check) for c in rows)):
        return (Check(name=name, status="fail",
                      detail=f"its setup action returned {rows!r:.80}, not a list of Check"),)
    return tuple(rows)
