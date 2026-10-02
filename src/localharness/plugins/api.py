"""The LocalHarness plugin API (v1) — every type a plugin author touches, each defined once.

A plugin is a class: an in-code frozen manifest plus optional contribution methods that all default
to no-ops. Core decides what loads (a pure plan over BUILTIN_PLUGINS and discovered METADATA — a
plugin that is not enabled is never imported) and runs one fixed lifecycle — configure → tools →
start → running → stop in reverse — in dependency order. A plugin gets one context object
(PluginContext), never imports cli/start_cmd.py and never imports another plugin. The copyable
reference is examples/plugin-template/.

One contract, two kinds of author: the features that ship with LocalHarness (memory, web, image,
dispatch, autoresearch — each becomes a bundled plugin listed in plugins/builtin.py) and plugins you
write and install yourself. Where the rules differ between the two, the docstring says so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Protocol, Sequence, runtime_checkable,
)

from pydantic import AfterValidator, BaseModel, ConfigDict

if TYPE_CHECKING:
    from localharness.channels.base import ChannelAdapter
    from localharness.core.bus import EventBus
    from localharness.core.reduce import ReduceLevel
    from localharness.provider.client import LLMClient
    from localharness.tools.base import ToolProtocol
    from localharness.tools.hooks import HookSystem
    from localharness.tools.registry import ToolRegistry

PLUGIN_API_VERSION = "1"
"""The version of this API (PAPI-12). It changes only when a change here would break a plugin
written against the previous version; an added optional field or method does not change it."""

DEFAULT_REQUIRES_LOCALHARNESS = ">=0.15,<1"
"""The default `requires_localharness`: any release from 0.15 up to, not including, 1.0. It admits
0.15 because the development builds that carry this API still report 0.15.0."""

PLUGIN_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
"""A plugin name: a lower-case letter, then up to 63 lower-case letters, digits, `_` or `-`. The
name is also the plugin's config key (`<name>:` and `agent.<name>`), so it is one dot-path segment."""

GLOBAL_ONLY_KEY = "x-localharness-global-only"
GLOBAL_ONLY: dict[str, Any] = {GLOBAL_ONLY_KEY: True}
"""Mark a ConfigModel field that names a network endpoint, a credential or an access list:
`Field(default, json_schema_extra=GLOBAL_ONLY)` — a value for it in a project's settings is dropped
with a warning and the machine-level value stands (ENAB-02)."""


def _fullmatch(pattern: re.Pattern[str], rule: str) -> AfterValidator:
    """A pydantic validator: the whole value must match `pattern`, else a ValueError stating `rule`."""
    def check(value: str) -> str:
        if not pattern.fullmatch(value):
            raise ValueError(f"{value!r} is not {rule}")
        return value
    return AfterValidator(check)


_PluginName = Annotated[str, _fullmatch(
    PLUGIN_NAME_RE, "a plugin name (a lower-case letter, then up to 63 lower-case letters, digits, "
                    "'_' or '-')")]
_ImportTarget = Annotated[str, _fullmatch(
    re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(\.[A-Za-z_]\w*)*"),
    "an import target ('package.module:attribute')")]


class CliDescriptor(BaseModel):
    """One `localharness <name>` command a plugin adds (PAPI-06), mounted from the manifest alone.

    `name` is the word under `localharness` (lower case, `-` allowed); `help` is its one line in
    `localharness --help`, shown without importing anything; `target` ("package.module:attr", attr a
    typer.Typer) is imported only when the command runs. A disabled plugin's commands are absent."""

    model_config = ConfigDict(frozen=True)
    name: Annotated[str, _fullmatch(
        re.compile(r"[a-z][a-z0-9-]*"), "a command word (a lower-case letter, then lower-case "
                                        "letters, digits or '-')")]
    help: str
    target: _ImportTarget


class SlashDescriptor(BaseModel):
    """One REPL slash command a plugin adds (PAPI-07): a row in the one table that feeds the REPL
    dispatcher, /help, the input completer and the phone's command menu.

    `name` is "/example" (lower case — the REPL matches lower-cased input); `help` is its one line;
    `target` is "package.module:function" for `async def function(ctx: PluginContext, args: str) ->
    str | None`, imported on first use — the returned text is shown to the user. `target` may also
    be "package.module:Class.method" naming a method of the plugin's own class: it is called on the
    running instance with (ctx, args). A handler may return text, None, or a rich renderable (shown
    via the channel's renderable path). Both additive; PLUGIN_API_VERSION unchanged."""

    model_config = ConfigDict(frozen=True)
    name: Annotated[str, _fullmatch(
        re.compile(r"/[a-z][a-z0-9_-]*"), "a slash command ('/', a lower-case letter, then "
                                          "lower-case letters, digits, '_' or '-')")]
    help: str
    target: _ImportTarget


class SetupField(BaseModel):
    """One question `plugins enable NAME` asks on a terminal: `key` is a leaf of the plugin's
    ConfigModel, `prompt` the words shown, `default` the answer offered. Data, not a callback:
    the harness asks, checks and writes; the plugin does no I/O (a later setup wizard walks the
    same list across plugins)."""

    model_config = ConfigDict(frozen=True)
    key: str
    prompt: str
    default: str = ""


class PluginManifest(BaseModel):
    """What a plugin is, declared in code and frozen (PAPI-01). Core reads it before any of the
    plugin's methods run.

    - `kind`: "tools", "channel", "memory" (the memory slot — see MemorySlotPlugin) or "dev"
      (developer tooling: commands and checks).
    - `enabled_by_default` applies to BUNDLED plugins only. A plugin you install is off until you
      enable it globally — that enable is the operator's trust grant (SAFE-06).
    - `requires_extra`: the `localharness[<extra>]` install extra the plugin needs; without it the
      plugin is reported with the install command instead of loading.
    - `requires_localharness`: a PEP 440 specifier checked against the package version before
      anything else runs; out of range, the plugin is skipped and the reason shown.
    - `requires`: hard dependencies — they load first; if one is off, this plugin cannot load.
      `uses`: soft dependencies — they load first if they are on. A cycle is a load error.
    - `cli` / `slash`: the commands the plugin adds, mounted from these descriptors without
      importing the plugin until one runs.
    - `setup`: what `plugins enable` asks on a terminal when no --set is given (see SetupField).
    - `setup_help`: a few plain lines printed after that enable when the plugin's check does not
      pass: what to start, what files it needs, and optionally a prompt to paste into a coding agent.
    """

    model_config = ConfigDict(frozen=True)
    name: _PluginName
    version: str
    kind: Literal["tools", "channel", "memory", "dev"]
    enabled_by_default: bool = True
    requires_extra: str | None = None
    requires_localharness: str = DEFAULT_REQUIRES_LOCALHARNESS
    requires: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()
    cli: tuple[CliDescriptor, ...] = ()
    slash: tuple[SlashDescriptor, ...] = ()
    setup: tuple[SetupField, ...] = ()
    setup_help: str = ""


Availability = Literal["ready"] | tuple[Literal["unconfigured"], str]
AVAILABILITY_DOC = """What configure() reports: "ready", or ("unconfigured", "<the missing dot-path>")."""


class Check(BaseModel):
    """One `doctor` result from a plugin (PAPI-08): `name`, `status` pass / fail / skip, `detail`
    (what was found) and `hint` (what to do about it)."""

    model_config = ConfigDict(frozen=True)
    name: str
    status: Literal["pass", "fail", "skip"]
    detail: str = ""
    hint: str = ""


class ContextBudget(BaseModel):
    """Core's per-turn budget for the memory slot's prompt section: at most `max_chars` characters
    and at most `max_session_history` recent session-history entries."""

    model_config = ConfigDict(frozen=True)
    max_chars: int
    max_session_history: int


class ContextContribution(BaseModel):
    """What the memory slot adds to one turn's prompt: `sections`, each a (heading, markdown) pair
    that core renders as "## {heading}\\n{body}". Empty adds nothing."""

    model_config = ConfigDict(frozen=True)
    sections: tuple[tuple[str, str], ...] = ()


class BrowseQuery(BaseModel):
    """A memory search from the phone's memory screen or /memory (PAPI-04): free `text`, `tags`,
    a `min_confidence` floor, whether superseded versions are included, a `since`/`until` window and
    a `limit`."""

    model_config = ConfigDict(frozen=True)
    text: str = ""
    tags: tuple[str, ...] = ()
    min_confidence: float | None = None
    include_superseded: bool = False
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 50


@runtime_checkable
class MemoryBrowse(Protocol):
    """The memory slot's browse API (PAPI-04) — every verb the phone's memory screen and /memory use.

    Record shapes (pinned by the web phase, 46): a fact row is the dict
    {name, value, status, confidence, source, node_kind, tags, updated_at, provenance} with the full value
    (a caller clips for display). search -> [row]; get -> {"fact": row | None, "history": [row]}, or None
    when the name has neither; edit -> {"status": "edited" | "unchanged" | "missing", "name"} — written
    with source "user_edit" and provenance "user_edit@<epoch>" + ";<origin>" when origin is given;
    forget -> True when retired (recoverable, never deleted), False when there was no active fact;
    promote -> {"promoted": bool, "message": str}. `origin` is an additive optional argument: no
    PLUGIN_API_VERSION bump (runtime checks test attribute presence only — the 45 `setup` precedent).

    The bundled StoreBrowse also offers `store(name, content, *, confidence=1.0)` (seeding, used by the
    bench) and plugin-internal id-keyed reads used by the memory plugin's own commands. Neither is part
    of this Protocol; callers probe with getattr."""

    async def search(self, query: BrowseQuery) -> list[dict[str, Any]]: ...
    async def get(self, name: str) -> dict[str, Any] | None: ...          # the fact + "history"
    async def edit(self, name: str, content: str, origin: str = "") -> dict[str, Any]: ...  # user-edit provenance; origin marks the surface (e.g. "web")
    async def forget(self, name: str) -> bool: ...
    async def promote(self, name: str) -> dict[str, Any]: ...


@runtime_checkable
class MemoryWriteHandle(Protocol):
    """A subagent's write access to memory, bound per subagent by the slot occupant's
    bind_subagent() (PAPI-04), so cruncher gist persistence routes through the slot instead of a raw
    MemoryStore. v1 has one verb; the memory conversion binds the cruncher to it."""

    async def persist_reduce_trace(self, question: str, trace: Sequence[ReduceLevel]) -> None: ...


@dataclass(frozen=True)
class PluginPaths:
    """Where things live for one plugin — every path computed by core, never by the plugin.

    - `global_config_dir`: the always-global config layer (GUARDRAILS.md's home too).
    - `workspace`: the workspace layer's .localharness/, if one applies; else None.
    - `state_dir`: where this session's agent state lives (the workspace layer or the global dir).
    - `artifact_dir`: `<state_dir>/artifacts/<name>/` if the plugin sets wants_artifacts; else None
      (PAPI-10)."""

    global_config_dir: Path
    workspace: Path | None
    state_dir: Path
    artifact_dir: Path | None = None


@dataclass
class SessionInfo:
    """Who this session is, for a plugin that keeps per-session state (the memory plugin's store and
    session row). Set by core once per `start`; None outside a session (doctor, CLI commands). Not
    frozen: core sets `exit_reason` just before stop_plugins, and the plugin reads it in stop()."""

    agent_id: str
    division_id: str
    sitting_id: str
    model: str
    context_tokens: int
    budget: dict[str, Any]
    exit_reason: str = "complete"


@dataclass(frozen=True)
class PluginContext:
    """Everything a plugin receives — the v1 fields (PAPI-03) plus two additive optional fields,
    idle_llm and session. A plugin reaches the session
    only through this object: it never imports cli/start_cmd.py or another plugin.

    - `bus`: the session's EventBus.
    - `tools`: the session's ToolRegistry.
    - `hooks`: the HookSystem (pre_tool/post_tool only), or None if it failed to start. Register
      with `ctx.hooks.register_plugin(obj, name=<your plugin name>)` so a hook that raises is
      reported under your name.
    - `config` / `agent_config`: this plugin's validated ConfigModel (harness `<name>:`) and
      AgentConfigModel (`agent.<name>`) instances; None when the plugin declares no such model.
    - `paths`: PluginPaths.
    - `llm`: the session's LLM client; None outside a session (doctor, CLI commands).
    - `idle_llm`: an LLMTextAdapter over `llm` for cancellable background completions; None when
      `llm` is None (additive, not a v1 field — PAPI-03).
    - `session`: SessionInfo for the running session; None outside a session (additive; disclosed
      beside idle_llm)."""

    bus: EventBus
    tools: ToolRegistry
    hooks: HookSystem | None
    config: BaseModel | None
    agent_config: BaseModel | None
    paths: PluginPaths
    llm: LLMClient | None
    idle_llm: Any = None
    session: SessionInfo | None = None


class Plugin:
    """The base class of every plugin, bundled or installed. Every method is optional: the defaults
    do nothing, so a plugin overrides only what it contributes.

    Core runs one fixed lifecycle, in dependency order (`requires`, then `uses`): configure → tools
    → start → running → stop, the stops in reverse start order. A plugin that raises is disabled for
    the session and named in the startup warnings and in `doctor`; the harness keeps running
    (PAPI-11).

    Class attributes: `manifest` (required); `ConfigModel` validates the harness-level `<name>:`
    section and `AgentConfigModel` the `agent.<name>` section — the validated instances arrive as
    ctx.config and ctx.agent_config; `wants_artifacts` asks core for an artifact root (see
    artifact_root).
    """

    manifest: ClassVar[PluginManifest]
    ConfigModel: ClassVar[type[BaseModel] | None] = None
    AgentConfigModel: ClassVar[type[BaseModel] | None] = None
    wants_artifacts: ClassVar[bool] = False

    async def configure(self, ctx: PluginContext) -> Availability:
        """Validate settings and report availability (see AVAILABILITY_DOC). Must not start
        anything — doctor calls it outside a session, where ctx.llm is None."""
        return "ready"

    async def tools(self, ctx: PluginContext) -> list[ToolProtocol]:
        """The tools this plugin contributes. Tools must carry the SAFE-01 declarations (ingest,
        host, result_origin, gate_family — an undeclared axis fails closed); they register under
        their bare names at global scope with source_plugin set."""
        return []

    async def start(self, ctx: PluginContext) -> None:
        """Start what runs for the session: servers, schedulers, hook registrations on ctx.hooks."""
        return None

    async def stop(self, ctx: PluginContext) -> None:
        """Release what start() acquired. Plugins stop in reverse start order."""
        return None

    def doctor(self, ctx: PluginContext) -> list[Check]:
        """The checks `localharness doctor` runs for this plugin, after core's. ctx.llm is None."""
        return []

    def channels(self) -> dict[str, type[ChannelAdapter]]:
        """The channel kinds this plugin provides, by the name `start --channel` accepts."""
        return {}

    def artifact_root(self, ctx: PluginContext) -> Path | None:
        """Return the root you write to; core serves artifacts only if this equals the root it
        computed (ctx.paths.artifact_dir) — anything else disables your artifact serving for the
        session."""
        return ctx.paths.artifact_dir


class MemorySlotPlugin(Plugin):
    """The memory slot's occupant (PAPI-04): a plugin of kind "memory". Core holds at most one
    occupant, and the slot may be empty — the harness runs without memory.

    Core asks context() for each turn's memory section within the budget it sets; the phone's
    memory screen and /memory reach memory only through browse(); a subagent that persists work
    (the cruncher's gists) gets its handle from bind_subagent() instead of a raw store."""

    async def context(self, ctx: PluginContext, turn: str, budget: ContextBudget) -> ContextContribution:
        """This turn's memory section(s), within `budget`. Empty by default."""
        return ContextContribution()

    def browse(self) -> MemoryBrowse | None:
        """The browse API, or None when this occupant has nothing to browse."""
        return None

    def bind_subagent(self, ctx: PluginContext) -> MemoryWriteHandle | None:
        """A write handle for one subagent, or None — that subagent then persists nothing."""
        return None


def plugin_summary(cls: type) -> str:
    """WHAT IT DOES in `plugins list`: the first non-empty line of the plugin class's docstring
    (the manifest's fields are exactly the PAPI-01 set, so the one-line description lives here)."""
    return next((line.strip() for line in (cls.__doc__ or "").splitlines() if line.strip()), "")
