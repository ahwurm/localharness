# Spec 09: Hooks and Plugins

**Component:** `src/localharness/tools/hooks.py`, `src/localharness/plugins/`
**Requirements covered:** HOOK-01, HOOK-02, HOOK-03
**Dependencies:** `tools/registry.py` (spec 04), `core/events.py`
**Library:** pluggy 1.6.0, importlib.metadata (stdlib)
**Stability:** UNSTABLE (v1). The hook sections describe the live code; the Plugins section describes the implemented API in outline — `src/localharness/plugins/api.py` and `examples/plugin-template/` are the reference.

---

## Purpose

LocalHarness has two extension mechanisms:

1. **Tool hooks** — `pre_tool` and `post_tool`, dispatched through `pluggy`. They run before and after a tool call, for cross-cutting concerns such as audit logging, lint gates and metrics, and a `pre_tool` hook can veto the call. These two are the only pluggy hooks.

2. **Plugins** — typed classes with a manifest declared in code and one fixed lifecycle (`src/localharness/plugins/`). A plugin can contribute tools, CLI commands, slash commands, `doctor` checks, settings and artifacts, and a plugin of kind `"memory"` can fill the memory slot.

The two meet in one place: a plugin receives the session's `HookSystem` as `ctx.hooks` and registers any hook object it has in its `start()`.

---

## Hook Specifications

Hook specs are the contracts. They define what hooks exist and what arguments they receive. LocalHarness defines one `HookSpec` class with two hooks.

```python
# src/localharness/tools/hooks.py
import pluggy
from typing import Any

# The project name used for pluggy's internal namespace.
HARNESS_HOOKSPEC = pluggy.HookspecMarker("localharness")
HARNESS_HOOKIMPL = pluggy.HookimplMarker("localharness")


class HarnesHookSpec:
    """The two tool hooks — the only pluggy hooks LocalHarness keeps. A plugin's lifecycle
    (configure / tools / start / stop) comes from plugins/api.Plugin, not from pluggy.

    Implementations are plain functions, each called on its own and synchronously: an exception
    is caught and logged under the implementing plugin's name. pluggy wrapper implementations are
    not supported and are skipped with a warning.
    """

    @HARNESS_HOOKSPEC
    def pre_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        agent_id: str,
        division_id: str,
    ) -> None:
        """Called before a tool's run() method is invoked.

        Implementations MAY raise ToolVetoed to prevent execution. Any other exception is caught,
        logged with the plugin's name, and the call goes on.

        Args:
            name: Tool name as registered in ToolRegistry.
            arguments: Validated (post-Pydantic) arguments dict. Read-only.
            agent_id: ID of the agent making the call.
            division_id: Division of the calling agent.
        """

    @HARNESS_HOOKSPEC
    def post_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        agent_id: str,
        division_id: str,
    ) -> None:
        """Called after a tool's run() method returns. Observability only.

        Called whether or not run() succeeded; never called when a pre_tool hook vetoed the
        call (the tool did not run). An exception here (ToolVetoed included) is caught and logged
        with the plugin's name; the tool's result is unchanged.

        Args:
            name: Tool name.
            arguments: Validated arguments dict (same as passed to pre_tool).
            result: The ToolResult returned by run(). Read-only.
            agent_id: ID of the agent making the call.
            division_id: Division of the calling agent.
        """
```

### Hook Calling Convention

`HookSystem` calls each implementation of a hook **separately**, in pluggy's own order (the last registered runs first), passing only the arguments that implementation names:

- **`pre_tool`**: implementations run one by one until one raises `ToolVetoed`. That stops the call: the ones after it do not run, and the tool returns `ToolResult(success=False, error_type="permission_denied")` carrying the veto's message.
- **`post_tool`**: every implementation runs. A `ToolVetoed` raised here is logged and ignored — only `pre_tool` can veto.
- **Any other exception**, in either hook, is caught and logged as a warning naming the plugin (its name on the `HookSystem`) and the implementation's module, with the traceback. The remaining implementations and the tool call go on.
- **pluggy wrappers** (`wrapper=True` or `hookwrapper=True`) are not supported and are skipped with a warning naming the plugin.
- **Synchronous only.** Implementations are called and never awaited, so an `async def` implementation never runs: calling it only creates a coroutine, Python warns that the coroutine was never awaited, and a veto it would have raised is lost.

**Where hooks fire.** `localharness start` builds one `HookSystem` per session and wires it to the session's tool registry before any plugin starts; implementations registered later are called too. Hooks fire for tool calls dispatched through that registry, which are the calls of the agent you talk to. A subagent runs on a registry of its own, built from the tools it is allowed, and that registry has no hooks: **`pre_tool` and `post_tool` do not fire for a subagent's tool calls.**

**Raw registry hooks.** `ToolRegistry.register_pre_hook` and `register_post_hook` (spec 04) accept any callable, and the `HookSystem` registers its two callers that way. A callable registered there directly that raises anything but `ToolVetoed` is swallowed by the registry with no log line. Register hook objects on the `HookSystem` instead.

---

## Hook Implementation Pattern

Hook implementations are methods decorated with `@HARNESS_HOOKIMPL`:

```python
# Example: my_plugin/hooks.py
import logging
from typing import Any

from localharness.tools.base import ToolVetoed
from localharness.tools.hooks import HARNESS_HOOKIMPL

log = logging.getLogger(__name__)


class MyPluginHooks:
    """Example hook implementation."""

    @HARNESS_HOOKIMPL
    def pre_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        agent_id: str,
        division_id: str,
    ) -> None:
        # Block bash commands that contain 'rm -rf /'
        if name == "bash_exec":
            cmd = arguments.get("command", "")
            if "rm -rf /" in cmd:
                raise ToolVetoed(f"Destructive command blocked: {cmd!r}")

    @HARNESS_HOOKIMPL
    def post_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        agent_id: str,
        division_id: str,
    ) -> None:
        if not result.success:
            log.info("tool_failure agent=%s tool=%s", agent_id, name)
```

Hook implementations **do not** need to implement both hooks. Pluggy only calls methods that are decorated with `@HARNESS_HOOKIMPL`.

A plugin registers its hook object in `start()`, under its own name, so that a hook that raises is reported under that name:

```python
async def start(self, ctx: PluginContext) -> None:
    if ctx.hooks is not None:  # None only when the session's hook system failed to start
        ctx.hooks.register_plugin(MyPluginHooks(), name=self.manifest.name)
```

If the plugin fails at any lifecycle stage, the hook objects it registered through `ctx.hooks` are unregistered with it (see "Lifecycle and containment" below).

---

## `HookSystem` Class

The runtime manager: it holds the pluggy `PluginManager` with the two hookspecs and connects dispatch to a `ToolRegistry`.

```python
# src/localharness/tools/hooks.py (continued): the public surface
class HookSystem:
    def __init__(self) -> None:
        self.pm = pluggy.PluginManager("localharness")
        self.pm.add_hookspecs(HarnesHookSpec)

    def register_plugin(self, plugin: object, name: str | None = None) -> None:
        """Register a hook implementation object (dedup-safe). Pass `name` (a plugin passes its
        own plugin name) so a hook that raises is reported under it rather than an object id."""

    def wire_to_registry(self, registry: "ToolRegistry") -> None:
        """Register one async pre-hook and one async post-hook on `registry`; each calls every
        implementation of its hook separately (see Hook Calling Convention)."""
```

`HookSystem` discovers nothing. Hook objects are registered on it with `register_plugin`, which ignores an object that is already registered. Unregistering goes through pluggy directly (`hooks.pm.unregister(obj)`); the plugin lifecycle does it for a plugin that fails.

---

## Plugins

This section outlines the implemented API. `src/localharness/plugins/api.py` defines every type a plugin touches, once each and with docstrings, and `examples/plugin-template/` is a complete working plugin that the development install includes and the test suite exercises; start there. SECURITY.md ("Plugins") states the trust model. The example imports three LocalHarness modules: `localharness.plugins.api`, `localharness.tools.base` and `localharness.core.artifacts`.

A plugin that ships with LocalHarness (listed in `BUILTIN_PLUGINS`, `plugins/builtin.py`) and a plugin you install yourself use the same API and go through the same code; where they differ, it is said below. `BUILTIN_PLUGINS` holds two plugins today: `image` (off by default) and `web` (the phone app, on by default; needs the `web` install extra); the other bundled features have not converted yet.

### Discovery and enabling

Discovery (`plugins/discovery.py`) has two sources, both machine-level; a workspace is never one:

- an installed distribution with an entry point in the group **`localharness.plugins`**: its name is the plugin name and its value names the `Plugin` subclass (`"package.module:Class"`);
- a folder `<global config dir>/plugins/<name>/` whose `__init__.py` binds `plugin = <the Plugin subclass>`. It is imported by file location, under `localharness_folder_plugins.<name>`, without touching `sys.path`.

Discovery reads metadata only: entry-point names, distribution names and versions, folder names. A plugin you installed is **off** until enabled and is not imported before that. Its `<name>.enabled` is read from the machine-level layers only (the global `config.yaml` and `overrides.yaml`); a project's value is dropped with a warning. A bundled plugin's `<name>.enabled` is layered like any setting and defaults to the manifest's `enabled_by_default`.

- `localharness plugins list [--json]` shows every plugin (name, what it does, state, where it came from) and the command that turns an off or available one on; `plugins info NAME [--json]` shows one plugin and the settings it owns.
- `localharness plugins enable NAME [--set KEY=VALUE ...] [--workspace] [--no-input]` and `plugins disable NAME [--workspace]` write `NAME.enabled`, and any `--set` values after checking them with the plugin's `ConfigModel`, into one layer's `overrides.yaml`: the machine's, or with `--workspace` the project's. They never write a `config.yaml`. `--workspace` is refused for a plugin you installed and for a machine-level-only setting. The change takes effect at the next `localharness start`. Unlike `components set`, these commands write no audit event. A plain `enable` imports nothing; `enable NAME --set ...` imports a plugin you installed, to check the values. On a terminal, with no `--set`, it asks the plugin's `setup` questions, writes the answers the same way, runs the plugin's doctor check once, and prints `setup_help` if the check does not pass.

### Plugins written for 0.15

LocalHarness 0.15 loaded plugins in three ways, and none of them is loaded any more:

- an installed package's `localharness.tools` entry points, each naming a tool class, which `start` created and registered as a tool;
- an installed package's `localharness.hooks` entry points, each naming a hook class, which `start` created and registered on the hook system;
- a folder `<global config dir>/plugins/<dir>/` holding a `manifest.yaml` that listed tool and hook classes.

0.15's `start` loaded all of them in every session; nothing had to turn them on. 0.15 also declared three more hooks, `on_agent_start`, `on_agent_end` and `on_event`, which nothing in 0.15 ever called.

LocalHarness now imports nothing from such a plugin. Reading package metadata and file names only, it names each one in `localharness plugins list` (on stderr), in `localharness doctor` (a warning, not a failure) and in the startup summary of `localharness start`: a package with a `localharness.tools` or `localharness.hooks` entry point and no `localharness.plugins` one, as ``<package> <version> (`localharness.tools` entry point `<name>`) was built for the 0.15 plugin API and is no longer loaded``, and a `plugins/<dir>/` folder with a `manifest.yaml` and no `__init__.py`, by the manifest's path. The notice stops once the plugin is ported or removed (the package uninstalled, or the folder deleted).

To port one, with `examples/plugin-template/` as the model:

1. Write a `Plugin` subclass with a `PluginManifest` (see "Manifest and methods").
2. Expose it as a `localharness.plugins` entry point whose name is the manifest's name, in place of the old entry points, or as a folder `<global config dir>/plugins/<name>/` whose `__init__.py` binds `plugin = <the class>` (see "Discovery and enabling").
3. Return your tools from `tools(ctx)`, and give each tool's `ToolSchema` the four declarations; a tool that declares nothing is treated as the riskiest kind (see "Tool declarations and the gate-family rule"). A tool may not use a name the permission gate classifies by name (see "Lifecycle and containment").
4. Keep your `pre_tool` and `post_tool` implementations as they are, and register the object that holds them in `start()` with `ctx.hooks.register_plugin(obj, name=self.manifest.name)` (see "Hook Implementation Pattern").
5. `on_agent_start`, `on_agent_end` and `on_event` have no replacement hook; 0.15 never called them, so dropping them loses nothing. What exists instead: `start(ctx)` and `stop(ctx)` run once per session, and `ctx.bus` is the session's event bus. `ctx.bus.subscribe(TurnStarted, handler)`, with an event class from `localharness.core.events`, calls `handler` with each such event the session publishes, and returns a handle for `ctx.bus.unsubscribe`. The lifecycle does not remove subscriptions: unsubscribe in `stop()`. A plugin whose `start()` raised is never stopped, so a subscription it made stays for the session.
6. Install it, then run `localharness plugins enable <name>`: a plugin you install runs nothing until it is enabled.

### The load plan

`resolve(loader)` (`plugins/resolve.py`) discovers, reads each plugin's `enabled`, imports only the enabled plugins you installed, validates settings, and passes the result to `build_load_plan` (`plugins/plan.py`), which is pure: it reads no file and imports nothing. The same plan feeds `start`, the banner, `plugins list`, `doctor`, `components` and the CLI mount. Each plugin gets one state:

| State | Meaning |
|---|---|
| `on` | loads this session |
| `off` | ships with LocalHarness and is not enabled |
| `available` | installed, not enabled, never imported |
| `failed` | enabled, but importing it raised, its settings are invalid, or a plugin it `requires` is not on |
| `skipped` | enabled, but its `requires_localharness` range excludes this version |
| `refused` | breaks a plan rule: an invalid name, a name that is a core settings key or already taken, a class whose manifest is missing or names another plugin, kind `"memory"` without being a `MemorySlotPlugin`, two or more memory plugins on at once (all are refused), or a dependency cycle |
| `needs-extra` | enabled, but the `localharness[<extra>]` install extra it needs is missing |

The ON plugins are ordered dependencies first (`requires`, and `uses` naming a plugin that is on), ties in display order.

### Manifest and methods

A plugin is a subclass of `Plugin` whose class attribute `manifest` is a frozen, validated `PluginManifest`:

| Field | Meaning |
|---|---|
| `name` | the plugin name and its settings key: a lower-case letter, then up to 63 of `a-z`, `0-9`, `_`, `-` |
| `version` | the plugin's own version |
| `kind` | `"tools"`, `"channel"`, `"memory"` or `"dev"` |
| `enabled_by_default` | read for a plugin that ships with LocalHarness only |
| `requires_extra` | the `localharness[<extra>]` the plugin needs |
| `requires_localharness` | a PEP 440 range, default `>=0.15,<1`; out of range, the plugin is skipped. For a plugin you installed it is checked after the import, since the manifest is in its code |
| `requires` / `uses` | hard / soft dependencies on other plugins |
| `cli` / `slash` | `CliDescriptor` / `SlashDescriptor` tuples (see "Commands, slash commands and doctor") |
| `setup` | `SetupField` tuple — the questions `plugins enable` asks on a terminal when no `--set` is given: `key` (a leaf of `ConfigModel`), `prompt`, `default`. Data, not a callback: the harness asks, writes and checks |
| `setup_help` | a few plain lines printed after that enable when the plugin's doctor check does not pass |

`PLUGIN_API_VERSION` (`"1"`) names this API's version; it changes only with a change that would break a plugin. Nothing in the loader compares it with anything: `requires_localharness` is the check that runs. The other class attributes are `ConfigModel` and `AgentConfigModel` (pydantic models for the plugin's settings, or None) and `wants_artifacts` (default False). The first line of the class docstring is what `plugins list` shows. Every method is optional, and the defaults do nothing:

- `configure(ctx)` returns `"ready"` or `("unconfigured", "<missing dot-path>")`, and must not start anything.
- `tools(ctx)` returns the tools the plugin contributes.
- `start(ctx)` and `stop(ctx)` acquire and release what runs during the session.
- `doctor(ctx)` returns a list of `Check(name, status, detail, hint)`, with `status` one of `"pass"`, `"fail"`, `"skip"`.
- `channels()` returns channel classes by name. Nothing calls it yet. `start --channel` accepts core's channels (`terminal`, `acp`), the manifest name of every bundled plugin of kind `channel` (today `web`) and the legacy `discord` (until Discord becomes a plugin), read from class-level manifests before any plugin loads (`plugins/channels.py`). A plugin you install cannot add a channel (SECURITY.md, "`localharness web`").
- `artifact_root(ctx)` returns the root the plugin writes to (default: `ctx.paths.artifact_dir`).

`MemorySlotPlugin`, for kind `"memory"`, adds `context(ctx, turn, budget)`, `browse()` and `bind_subagent(ctx)`. `configure`, `tools`, `start`, `stop` and `doctor` may be `def` or `async def`: the lifecycle awaits a result when it is awaitable. `context()` must be `async def`, and `artifact_root()`, `browse()` and `bind_subagent()` must be plain `def`; the wrong kind counts as a failed call.

### Context and paths

Every method gets one `PluginContext` with the seven v1 fields, plus two additive optional ones: `bus` (the session's `EventBus`), `tools` (its `ToolRegistry`), `hooks` (its `HookSystem`, or None if that failed to start), `config` and `agent_config` (the plugin's validated `ConfigModel` and `AgentConfigModel` instances, or None), `paths`, `llm` (the session's LLM client; None outside a session, as in `doctor`), `idle_llm` (an adapter over `llm` for cancellable background completions; None when `llm` is None) and `session` (a `SessionInfo` naming the running session — agent, division, sitting, model, context tokens, budget, exit reason; None outside a session, as in `doctor` and CLI commands; core sets the exit reason before it stops the plugins). `session` goes beyond the v1 field list's letter, which named only the idle-LLM addition; both are additive with defaults, so `PLUGIN_API_VERSION` is unchanged. A plugin's CLI command gets no context: it is a plain Typer app. `paths` is a `PluginPaths` whose every path core computes: `global_config_dir`, `workspace` (or None), `state_dir` (the workspace layer when one applies, else the global directory) and `artifact_dir` (see Artifacts). The API asks a plugin never to import `cli/start_cmd.py` or another plugin; nothing enforces that.

### Lifecycle and containment

`localharness start` resolves the plugins once the built-in tools and the hook system are set up. `start_plugins` (`plugins/lifecycle.py`) then runs the ON plugins through four stages: create (the class is instantiated), `configure`, `tools`, `start`. Each stage runs over every plugin in dependency order before the next stage begins. After that the root agent's capability floor is applied to every registered tool, and then MCP servers connect. When the session ends, after the MCP servers shut down, `stop_plugins` stops the running plugins in reverse start order.

Every call into plugin code is contained (`except (Exception, SystemExit)`). A plugin is **failed for the session** when it raises, exits, or hands back something unusable: a `configure()` answer in neither form above, a `tools()` result that is not a list of tools, or a tool name that it uses twice, that is already registered, or that the permission gate classifies by name (`write`, `edit`, `bash_exec`, `python_exec`, `cruncher_exec`, `agent`, `web_fetch`). A failed plugin is named in the startup summary as `plugin <name>: <reason>`, the tools it registered and the hook objects it put on `ctx.hooks` are removed, and every plugin that `requires` it is failed too. An unconfigured plugin is reported as `plugin <name>: unconfigured — set <key>` and registers nothing. The harness keeps running; `KeyboardInterrupt` and task cancellation still propagate, after every plugin already running has been stopped (the interrupted plugin is not stopped: it cleans up what its `start()` opened). A plugin can report soft startup problems without failing: strings it leaves on an optional `startup_warnings` list attribute during `start()` are added to the startup warnings as written, whether `start()` returned or raised. A `stop()` that raises is logged, and the other plugins still stop.

What containment does not cover: a stage that hangs; a plugin whose `start()` raised, which is never stopped; and anything a plugin registers directly on `ctx.tools` (a tool, or a raw pre/post hook) instead of returning it from `tools()` or registering it on `ctx.hooks`. Such a tool carries no `source_plugin` and escapes the gate-family rule below, and it stays registered if the plugin fails.

When any plugin runs, the start banner adds `Plugins: <names>`. When plugins you installed are not enabled, it adds one line that names them all: ``i 1 plugin available, not enabled: <name> — run `localharness plugins enable <name>` to turn it on``, or, for two or more, ``i 2 plugins available, not enabled: <a>, <b> — run `localharness plugins enable <name>` to turn one on``.

### Settings

A plugin's harness-level settings live under `<name>:`, and its agent-level settings under `agent.<name>` in an agent's file. Core splits those sections off before it validates its own config, so a key that is neither a core setting nor a known plugin's is still an error. That includes a section left by a plugin that is no longer installed: its `<name>:` section is refused like a misspelled key, with an error that names the file, the line and the fix (``not a LocalHarness setting, and no installed plugin is named `<name>` — if a plugin you removed used it, reinstall that plugin or delete this section``), and `validate` reports a leftover `agent.<name>` entry the same way. `plugins list` names each leftover `<name>:` section in any config layer and each leftover `agent.<name>` entry in the machine's `overrides.yaml`, and `plugins info`, `enable` and `disable` for a removed plugin's name say where its settings still are. The plugin's own `ConfigModel` and `AgentConfigModel` validate them, and the validated instances arrive as `ctx.config` and `ctx.agent_config`. The harness-level section merges across the four config layers like any setting. A field marked `Field(..., json_schema_extra=GLOBAL_ONLY)` is machine-level only: a project's value for it is dropped with a warning. A field that names an endpoint, a credential or an access list should be marked. `GLOBAL_ONLY` on an `AgentConfigModel` field fails the plugin, because agent-level settings cannot enforce it. Plugin settings are listed by `components list` with `(plugin: <name>)` and set with `components set`.

### Tool declarations and the gate-family rule

Every `ToolSchema` carries four safety declarations, each with a fail-closed default: `ingest` (`"untrusted"` or `"none"`), `host` (`"dangerous"` or `"safe"`), `result_origin` (`"untrusted"` or `"trusted"`) and `gate_family` (None, or one of `"write"`, `"shell"`, `"code"`, `"delegate"`, `"network"`, `"allow"`). They are excluded from the schema sent to the model. The permission gate reads `gate_family`; the capability floor reads `ingest` and `host`, plus `gate_family` to find exec surfaces; the context store reads `result_origin`. The gate still recognises its own builtins by name, which is why a plugin may not reuse those names; every other tool is classified only by what it declares, and no safety reader looks at `source_plugin`. A tool that declares nothing is the worst case on every axis, so while the capability floor is on (the default) no agent may hold it: the root agent's floor strips it, with a startup warning naming the tool and its plugin, and any other agent configured with it is refused.

A plugin's tools register under their bare names at global scope, with `source_plugin` set. For a plugin you installed, and never for one that ships with LocalHarness, the lifecycle rewrites what reaches the gate before registering (`plugins/trust.py`): a `gate_family` in `THIRD_PARTY_CLAMPED_FAMILIES` (`allow`, `network`, `shell`, `write`) becomes undeclared, and a `group` starting with `mcp/` becomes `other`, each with a startup warning. `code` and `delegate` are kept. The clamped set is measured by a test over the real gate, not argued.

### Artifacts

A plugin sets `wants_artifacts = True`. Core then computes its root, `<state_dir>/artifacts/<name>/`, and passes it as `ctx.paths.artifact_dir`; a plugin that did not ask gets None. `write_artifact(root, name, data, mime)` in `localharness.core.artifacts` accepts only `image/png`, `image/jpeg` and `image/webp`, mints the id (`art-YYYYMMDD-HHMMSS-<6 hex>`, UTC), writes exactly one new file (never overwriting one) and returns an `ArtifactRef(plugin, kind="image", id, mime)`. After the start stage, core asks each running plugin's `artifact_root(ctx)`; unless the answer is the root core computed, that plugin's artifacts are not served this session. The web channel serves the accepted roots at `GET /api/artifacts/{plugin}/{artifact_id}` (SECURITY.md, "`localharness web`"). `ArtifactRef` is not an event; the one event that carries it is `Observation`, whose optional `artifact` field (web protocol 4) the agent loop fills from a successful tool result's `metadata["artifact"]` after validating it as an `ArtifactRef`, dropping anything that does not validate. The reference phone page shows an image artifact inline; the terminal does not display artifacts.

### The memory slot

`MemorySlot` (`plugins/slot.py`) holds at most one running `MemorySlotPlugin`, or nobody. When memory is on (the default), the occupant is the bundled `memory` plugin (`memory/plugin.py`): it opens its store in its own `start()`, contributes `memory_search`, `memory_get` and `remember`, and its `context()` renders the `## Division Context` and `## Agent Memory` sections. When the slot is occupied, the agent loop asks `context(turn, budget)` on every turn, right after the guardrails, and adds each non-empty section as `## <heading>`. The budget is core's ceiling, not a share: `max_chars` is the usable context window (the window minus the reply reserve) in characters, and `max_session_history` is at most 200 entries; the occupant renders within the smaller of its own `agent.memory` settings and that ceiling. Core does not truncate what comes back; the occupant is trusted to keep to the budget. Every call into the occupant is contained. `browse()` returns a `MemoryBrowse` (`plugins/api.py`): five verbs, `search`, `get`, `edit`, `forget` and `promote`, and it serves the phone's memory screen. The bundled plugin's browse class (`StoreBrowse`, `memory/browse.py`) also offers an optional `store(name, content, *, confidence=1.0)` seeding verb and id-keyed reads for its own commands; they are not part of the Protocol, so callers probe for them with `getattr`. `PLUGIN_API_VERSION` is unchanged: both additions are additive. `bind_subagent()` is asked once per cruncher run for a write handle; the bundled plugin returns none today, because subagent gist persistence is not implemented. With memory off (`memory.enabled: false`) the slot is empty unless you install and enable another memory plugin: no memory section, no memory tool and no `/memory` command exist, and the phone's memory routes answer 404. If the memory store cannot open, the plugin is failed for the session and the slot is empty in the same way. `/memory` is the plugin's own slash row (target `localharness.memory.plugin:MemoryPlugin.slash_memory`), called on the running plugin and reaching the store only through its browse class; `localharness memory` is its `CliDescriptor`, which opens the agent's store itself and wraps it in the same browse class. The bench starts the plugins through the same lifecycle over a throwaway config directory, for the scenarios that seed memory only, and seeds through the occupied slot's `browse().store`; it imports no memory module.

### Commands, slash commands and doctor

- **CLI commands.** Each `CliDescriptor(name, help, target)` of an ON plugin appears in `localharness --help`, read from the manifest. The module that `target` names (`"package.module:attr"`, a `typer.Typer`) is imported only when the command runs, and it runs as a subcommand of `localharness`; `localharness --help` lists plugin commands after the core ones. A core command wins a name clash, and when two plugins share a command name the first in start order wins, without a warning. The CLI resolves plugins only to list commands (`--help`) or to find a name no core command has. So a core command (`start`, `doctor`, `--version` and the rest) never loads a plugin to build the command list, while `localharness --help` imports each enabled plugin you installed: its package and plugin class, because the manifest is in code, and never a command module. A command that fails to import or raises prints `plugin <name>: command <cmd> …` and exits 1.
- **Slash commands.** Each `SlashDescriptor(name, help, target)` of a running plugin becomes a row in the one slash table (`cli/slash_commands.py`). The REPL dispatcher, `/help` and the input completer read that table, and the web channel's `/api/protocol` lists its rows in `commands[]`. A plugin's rows are in the table only while its session runs: they are added when the session starts and removed when it ends, so a client that reads `commands[]` before a session is live sees core's rows only. The reference phone page has no command menu. `target` names `async def f(ctx, args)`, imported the first time the command runs; a target `"package.module:Class.method"` that names a method of the plugin's own class is called on the running instance with `(ctx, args)`, which is how a handler reaches its plugin's live state. A handler may return text (shown as info), None or `""` (nothing shown), or a rich renderable (shown through the channel's renderable path). A name already in the table is skipped with a warning. A handler that raises, exits or returns anything else is reported to the user as a failure naming the plugin, and the session goes on. Plugin rows follow core rows, so `/help` and the completer list them after `/quit` and `/exit`. The Zed (ACP) channel does not read the table.
- **Doctor.** After core's checks, `localharness doctor` prints a Plugins section. An ON plugin is created and configured against a throwaway context (no LLM client, and a fresh bus, registry and hook system), and its `doctor()` checks are shown; a failing check is a doctor failure. An off or available plugin is listed with the command that turns it on. `skipped`, `needs-extra` and unconfigured plugins are warnings, and `failed` and `refused` ones are failures. Doctor never starts a plugin, and it does not show which plugin tools the root agent's floor will strip.
