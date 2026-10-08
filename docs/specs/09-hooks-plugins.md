# Spec 09: Hooks and Plugins

**Component:** `src/localharness/tools/hooks.py`, `src/localharness/plugins/`
**Requirements covered:** HOOK-01, HOOK-02, HOOK-03
**Dependencies:** `tools/registry.py` (spec 04), `core/events.py`
**Library:** pluggy 1.6.0, importlib.metadata (stdlib)
**Stability:** v1 — `PLUGIN_API_VERSION = "1"`. Changes since the first release have been additive only (each listed under "Additive fields"); a change that would break a plugin bumps the version. The code is authoritative: `src/localharness/plugins/api.py` defines the API and `examples/plugin-template/` is a working plugin, and `tests/unit/test_docs_convergence.py` checks this spec against the code in both directions (every name here exists there, and every name there is here).

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

This section outlines the implemented API. `src/localharness/plugins/api.py` defines every type a plugin touches, once each and with docstrings, and `examples/plugin-template/` is a complete working plugin that the development install includes and the test suite exercises; start there. [SECURITY.md, "Plugins"](../../SECURITY.md#plugins) states the trust model; this spec states the mechanism. The example imports three LocalHarness modules: `localharness.plugins.api`, `localharness.tools.base` and `localharness.core.artifacts`.

A plugin that ships with LocalHarness (listed in `BUILTIN_PLUGINS`, `plugins/builtin.py`) and a plugin you install yourself use the same API and go through the same code; where they differ, it is said below.

### The bundled plugins

`BUILTIN_PLUGINS` is the one list of what ships; every reader (the loader, the banner, `plugins list`, `doctor`, `components`) reaches it through `bundled_plugins()`. It holds five plugins, in this order:

- **`image`** — kind `tools`, off by default, no install extra. Makes pictures with a local ComfyUI server: the `generate_image` tool and the `localharness generate-image` command. It sets `wants_artifacts`, so its pictures are artifacts (see "Artifacts"). Its `configure()` reports it unconfigured until `image.comfyui_url` is set. Its setup step asks for that address, checks ComfyUI, and on a failed check prints `setup_help` and its `agent_prompt` filled with the address and the GPU the machine reports. Off: no tool, and `generate-image` prints the off-plugin hint and exits 4 (see "Errors and refusals").
- **`mobile`** — kind `channel`, on by default, needs the `mobile` install extra. The phone app: the `localharness mobile` command serves the phone UI and its event API, and its `channels()` names the `mobile` channel. Its `start()` and `stop()` do nothing, because the server's life belongs to `localharness mobile`. Its setup step asks one machine-level setting, `mobile.public_url`: the address the pairing QR sends a phone to (left empty, `localharness mobile` guesses it). The bind address and `--allow-unsafe-bind` are never settings. Off: `localharness mobile` prints the off-plugin hint and exits 4.
- **`memory`** — kind `memory`, on by default, no install extra. Fills the memory slot (see "The memory slot"): the `memory_search`, `memory_get` and `remember` tools, the memory sections of each turn's prompt, background consolidation, the `/memory` slash command and the `localharness memory` command. Its setup step asks no questions: its `setup_action` downloads the embedding model, after asking, while the model is missing. Off (`memory.enabled: false`): the slot is empty and none of these exist; `localharness memory` prints the off-plugin hint and exits 4.
- **`dispatch`** — kind `channel`, on by default, needs the `dispatch` install extra. Chat platforms as a channel, today Discord: its manifest's `channels` is `("discord",)`, so `start --channel discord` drives the agent from allowlisted chat messages and posts the replies back. It builds the channel through `make_channel`, which hands the channel its validated settings, reports its deprecation lines through `startup_warnings` and a `warn` doctor check, and opens no network in `start()` (the gateway connects when the REPL starts the channel). It adds no tool and no command. Its setup step asks `discord.token` (not echoed) and `discord.allow`, and its `agent_prompt` never holds the token. Off: `start --channel discord` is refused (see "Errors and refusals"), while `start --help` still lists `discord`.
- **`autoresearch`** — kind `dev`, on by default, no install extra. Provides three commands, `autoresearch`, `experiment` and `propose`, and adds nothing to a session (no tools, no slash commands, no start or stop work); its `doctor()` check names the configured proposer model and address, never the key, and does not contact it. It owns the core settings `proposer:` and `sentinel:` under those names (see `sections` below). Its setup step asks `proposer.base_url`, `proposer.model` and `proposer.api_key` (secret: not echoed, empty for a local server), keys under its `sections` that are written together in one checked write; the proposer may be the main model again at a local address (the same model id as `provider.default_model` is accepted) or a cloud API with its key, and its `setup_action` asks the proposer for its model list once, reading the address and key from the machine-level config only and printing the host before the request goes, while `doctor()` still never contacts it. `proposer.base_url` and `proposer.api_key` are machine-level only. Off (`plugins disable autoresearch`): its commands leave `--help` and print the off-plugin hint (exit 4) when run, its `proposer.*`/`sentinel.*` rows leave `components list` (the `autoresearch.enabled` row stays), and nothing on disk is deleted. The bench is core and imports nothing from it.

Every bundled plugin's module is imported for every `--help`, `doctor` and `plugins list`, so at module level it imports only the plugin API (and its own settings model); its heavy parts are imported inside its methods or when its command runs. For example `localharness --help`, `doctor` and `plugins list` import only the `autoresearch` plugin class, and the experiment archive, scipy and the command modules load when a command runs; the memory and autoresearch packages re-export lazily for the same reason. `tests/unit/test_import_direction.py` keeps core from importing a plugin module; `plugins/builtin.py` is the one exception.

### Discovery and enabling

Discovery (`plugins/discovery.py`) has two sources, both machine-level; a workspace is never one:

- an installed distribution with an entry point in the group **`localharness.plugins`**: its name is the plugin name and its value names the `Plugin` subclass (`"package.module:Class"`);
- a folder `<global config dir>/plugins/<name>/` whose `__init__.py` binds `plugin = <the Plugin subclass>`. It is imported by file location, under `localharness_folder_plugins.<name>`, without touching `sys.path`.

Discovery reads metadata only: entry-point names, distribution names and versions, folder names. A plugin you installed is **off** until enabled and is not imported before that. Its `<name>.enabled` is read from the machine-level layers only (the global `config.yaml` and `overrides.yaml`); a project's value is dropped with a warning. A bundled plugin's `<name>.enabled` is layered like any setting and defaults to the manifest's `enabled_by_default`.

- `localharness plugins list [--json]` shows every plugin (name, what it does, state, where it came from) and the command that turns an off or available one on; `plugins info NAME [--json]` shows one plugin and the settings it owns.
- `localharness plugins enable NAME [--set KEY=VALUE ...] [--workspace] [--no-input]` and `plugins disable NAME [--workspace]` write `NAME.enabled`, and any `--set` values after checking them with the plugin's `ConfigModel`, into one layer's `overrides.yaml`: the machine's, or with `--workspace` the project's. They never write a `config.yaml`. `--workspace` is refused for a plugin you installed and for a machine-level-only setting. The change takes effect at the next `localharness start`. Unlike `components set`, these commands write no audit event. A plain `enable` imports nothing; `enable NAME --set ...` imports a plugin you installed, to check the values. On a terminal, with no `--set` and no `--workspace`, it runs the plugin's setup step: it asks the `setup` questions (a question that is not secret offers the stored value, else its `default`; a secret one never shows what is stored; an empty answer is not written), writes the answers the same way, runs `setup_action(ctx)` after asking the manifest's `setup_action` question when there is one, and runs the plugin's doctor check once. The setup action runs only while that check does not pass yet, or when answers were written in this run. When the check does not pass, or a row of the setup action fails, it prints `setup_help`, then the plugin's `agent_prompt` with the values filled in; `next_steps` print on every outcome, and the command exits 0 either way. A plugin missing its install extra gets one line naming the extra and no questions. A setup key under one of a bundled plugin's `sections` is a core setting: it is written at its own path, and all such keys are checked together in one write. Without a terminal, or with `--no-input`, it asks nothing, runs no setup action, and names the next step.
- In a running terminal session, `/plugins enable NAME` and `/plugins disable NAME` make the same write and take effect at once: the session ends, the plugin's setup step runs on the plain terminal (its questions only while the plugin is not set up yet), and the session restarts with the plugin's flag changed and the conversation carried over; only the plugin lifecycle registers anything. A plugin that already runs and is set up, one a project pins, or one missing its install extra gets one line and no restart, and so does any `/plugins enable` or `/plugins disable` while a call is parked. Other channels answer one line: setup questions are asked on the terminal only.

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
| `setup` | `SetupField` tuple — the questions `plugins enable` asks on a terminal when no `--set` is given: `key` (a leaf of `ConfigModel`, or for a bundled plugin a dotted path under one of its `sections`), `prompt`, `default`, `secret` (default False; when True the answer is not echoed as it is typed). Data, not a callback: the harness asks, writes and checks |
| `setup_help` | a few plain lines printed after that enable when the plugin's doctor check does not pass |
| `next_steps` | plain lines printed after the plugin's setup step on every outcome, saying what to do next (default empty) |
| `agent_prompt` | the text printed under "Or paste this into your coding agent to set it up for your hardware:" after a check that does not pass, and by `plugins info NAME` in its text view at any time; no file is written. It may name `{config_dir}`, `{machine}` (this machine's GPU as the machine reports it, or nothing) and its own setup keys that are not secret and have a `default`, each filled with the stored value or that default. Any other name, or a secret key, is refused when the manifest is built, so a token never appears in a printed prompt (default empty) |
| `setup_action` | the yes/no question asked on a terminal before the plugin's `setup_action(ctx)` runs; a question about a download names its size. Empty runs it without asking (default empty) |
| `channels` | for kind `"channel"`: the names `start --channel` accepts from this plugin (default empty, meaning the plugin's own name). `dispatch` declares `("discord",)` |
| `sections` | bundled plugins only: pre-existing top-level core settings the plugin owns under their old names (default empty). While the plugin is on, `components list` shows those rows as `(plugin: <name>)` and `plugins info` lists them (`sections` in its `--json`); while it is off, the rows leave the catalogue and its `<name>.enabled` row stays. The config loader validates the section either way. `autoresearch` declares `("proposer", "sentinel")`. A plugin you install that declares `sections` is refused at load (`declares sections (claiming core settings), which only bundled plugins may do`) |

`PLUGIN_API_VERSION` (`"1"`) names this API's version; it changes only with a change that would break a plugin. Everything added since the first release is optional, with a default that leaves an existing plugin unchanged, so the version is still `"1"` (see "Additive fields"). Nothing in the loader compares it with anything: `requires_localharness` is the check that runs. The other class attributes are `ConfigModel` and `AgentConfigModel` (pydantic models for the plugin's settings, or None) and `wants_artifacts` (default False). The first line of the class docstring is what `plugins list` shows. Every method is optional, and the defaults do nothing:

- `configure(ctx)` returns an `Availability`: `"ready"` or `("unconfigured", "<missing dot-path>")`. It must not start anything: `doctor` calls it outside a session, where `ctx.llm` is None.
- `tools(ctx)` returns the tools the plugin contributes.
- `start(ctx)` and `stop(ctx)` acquire and release what runs during the session.
- `doctor(ctx)` returns a list of `Check(name, status, detail, hint)`, with `status` one of `"pass"`, `"fail"`, `"skip"`, `"warn"`. A `warn` check is printed with a warning sign and its hint, and is never counted as a doctor failure (the dispatch plugin uses it for its deprecation lines).
- `setup_action(ctx)` returns a list of `Check`, like `doctor(ctx)`: the work the plugin's setup step does after its questions, such as a download or a check that an endpoint answers. Core calls it only from `plugins enable` on a terminal, after asking the manifest's `setup_action` question when there is one, and only while the plugin's doctor check does not pass yet or when answers were written in that run, so pair it with a check that says whether the work is still needed; never at `start` and never from `doctor`. The call is contained like `doctor()`: an exception becomes one failing row, and core prints the rows as `doctor` prints its checks. `ctx.llm` is None. A destination that receives a credential is read from the machine-level config only (`ctx.paths.global_config_dir`, no workspace layer), never through a project's `.localharness/`, which loads without a prompt inside the project.
- `channels()` returns channel classes by name. `start --channel` accepts core's channels (`terminal`, `acp`) and, for each bundled plugin of kind `channel`, the names in its manifest's `channels` (or its own name when that is empty): today `mobile` and `discord`. A typo is refused from those class-level manifests before any config is read (`plugins/channels.py`). After the plugins resolve, `accepted_channels(resolution)` keeps only the channels of plugins whose state is `on`, and `start` refuses a known channel whose plugin is off (``channel 'discord' is provided by the dispatch plugin, which is off — run `localharness plugins enable dispatch` ``), is missing its install extra (naming the `localharness[<extra>]` to install), or resolved on but did not start (naming the reason). It never falls back to the terminal. `localharness start --help` still lists every bundled channel name, including one whose plugin is off. To build the channel, `start` calls the running plugin's `make_channel(name, bus)` if it has one (so the plugin can hand the channel its own validated settings), else `channels()[name](bus=bus, config={})`, and prints the channel's `start_banner` if it is not empty. A channel class that sets `bare_mode_command = True` takes a plain `mode <name>` message as the `/mode` command (Discord does; the terminal does not). `start` and the REPL name no chat platform. A plugin you install cannot add a channel (SECURITY.md, "`localharness mobile`").
- `artifact_root(ctx)` returns the root the plugin writes to (default: `ctx.paths.artifact_dir`).

`MemorySlotPlugin`, for kind `"memory"`, adds `context(ctx, turn, budget)`, `browse()` and `bind_subagent(ctx)`. `configure`, `tools`, `start`, `stop`, `doctor` and `setup_action` may be `def` or `async def`: the lifecycle awaits a result when it is awaitable. `context()` must be `async def`, and `artifact_root()`, `browse()` and `bind_subagent()` must be plain `def`; the wrong kind counts as a failed call.

### Additive fields

Each row is an addition made after the first release. Each is optional: a plugin written before it keeps working unchanged, which is why `PLUGIN_API_VERSION` is still `"1"`. "First needed by" names the bundled plugin whose conversion added it; "Presence-checked" says whether core probes for it (`getattr`) or reads a field whose default is the old behaviour.

| Field | What it is for | First needed by | Presence-checked |
|---|---|---|---|
| `PluginManifest.setup` / `setup_help`, `SetupField` | the questions `plugins enable` asks on a terminal, and the lines printed when the doctor check after them does not pass | `image` | no: fields, default empty |
| `MemoryBrowse.edit` `origin` | an optional argument that marks which surface made a user's edit (`"mobile"`, `"cli"`) in the fact's provenance | `memory` (for the phone's memory screen) | no: an optional argument, default `""` |
| `PluginContext.idle_llm` / `session`, `SessionInfo` | cancellable background completions; who the running session is | `memory` (consolidation and `remember`; its store and session row) | no: fields, default None |
| `startup_warnings` | soft startup problems a plugin reports without failing | `memory` | yes: an optional list attribute, read with `getattr` after `start()` |
| a slash `target` `"package.module:Class.method"` naming a method of the plugin's own class; a rich renderable as a slash reply | a slash handler reaching its running plugin's state, and replying with more than text | `memory` (`/memory`) | no: read from the target's shape and the reply's type |
| `PluginManifest.channels`, `accepted_channels` | the names `start --channel` accepts from a channel plugin; the channels of the plugins that are on | `dispatch` | no: a field, default empty (meaning the plugin's own name) |
| the `"warn"` value of `Check.status` | a doctor line that is shown with its hint and never counted as a failure | `dispatch` (its deprecation lines) | no: a new value of an existing field |
| `SetupField.secret` | a setup answer that is not echoed as it is typed | `dispatch` (the bot token) | no: a field, default False |
| `ChannelAdapter.bare_mode_command` / `start_banner` | a channel that takes a plain `mode <name>` message as `/mode`; the line `start` prints for the channel | `dispatch` (Discord) | no: class attributes with defaults (`False`, `""`) |
| `make_channel(name, bus)` | a plugin builds its own channel, so it can hand the channel its validated settings | `dispatch` | yes: called when the running plugin has it, else `channels()[name](bus=bus, config={})` |
| `PluginManifest.sections` | a bundled plugin owns pre-existing core settings under their old names | `autoresearch` | no: a field, default empty; a plugin you install that declares it is refused |
| the off-plugin command hint | a bundled plugin's command that runs while the plugin is off names the fix and exits 4 | `autoresearch` | not a field: read from the static manifest |
| `PluginManifest.next_steps` / `agent_prompt` / `setup_action`, `Plugin.setup_action` | what to do after a plugin's setup step, a prompt to paste into a coding agent with the user's own values filled in, and setup work the harness runs for the plugin (a download, a check that an endpoint answers) | `memory`, `image`, `autoresearch` | no: fields with empty defaults, and a method whose default returns no rows |

### Context and paths

Every method gets one `PluginContext` with the seven v1 fields, plus two additive optional ones: `bus` (the session's `EventBus`), `tools` (its `ToolRegistry`), `hooks` (its `HookSystem`, or None if that failed to start), `config` and `agent_config` (the plugin's validated `ConfigModel` and `AgentConfigModel` instances, or None), `paths`, `llm` (the session's LLM client; None outside a session, as in `doctor`), `idle_llm` (an adapter over `llm` for cancellable background completions; None when `llm` is None) and `session` (a `SessionInfo` naming the running session, for a plugin that keeps per-session state; None outside a session, as in `doctor` and CLI commands). Both additions default to None, so `PLUGIN_API_VERSION` is unchanged. Core sets `SessionInfo` once per `start`, with these fields: `agent_id` (the agent's name), `division_id` (its division, `"default"` when it has none), `sitting_id`, `model` (the resolved model), `context_tokens` (the context window), `budget` (the agent's budget settings, as a dict) and `exit_reason` (default `"complete"`). It is not frozen: core sets `exit_reason` (`"interrupt"` or `"error"` when the session did not end normally) just before it stops the plugins, and a plugin reads it in `stop()`. A plugin's CLI command gets no context: it is a plain Typer app. `paths` is a `PluginPaths` whose every path core computes: `global_config_dir`, `workspace` (or None), `state_dir` (the workspace layer when one applies, else the global directory) and `artifact_dir` (see Artifacts). The API asks a plugin never to import `cli/start_cmd.py` or another plugin; nothing enforces that.

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

A plugin sets `wants_artifacts = True`. Core then computes its root, `<state_dir>/artifacts/<name>/`, and passes it as `ctx.paths.artifact_dir`; a plugin that did not ask gets None. `write_artifact(root, name, data, mime)` in `localharness.core.artifacts` accepts only `image/png`, `image/jpeg` and `image/webp`, mints the id (`art-YYYYMMDD-HHMMSS-<6 hex>`, UTC), writes exactly one new file (never overwriting one) and returns an `ArtifactRef(plugin, kind="image", id, mime)`. After the start stage, core asks each running plugin's `artifact_root(ctx)`; unless the answer is the root core computed, that plugin's artifacts are not served this session. The mobile channel serves the accepted roots at `GET /api/artifacts/{plugin}/{artifact_id}` (SECURITY.md, "`localharness mobile`"). `ArtifactRef` is not an event; the one event that carries it is `Observation`, whose optional `artifact` field (web protocol 4) the agent loop fills from a successful tool result's `metadata["artifact"]` after validating it as an `ArtifactRef`, dropping anything that does not validate. The reference phone page shows an image artifact inline; the terminal does not display artifacts.

### The memory slot

`MemorySlot` (`plugins/slot.py`) holds at most one running `MemorySlotPlugin`, or nobody. When memory is on (the default), the occupant is the bundled `memory` plugin (`memory/plugin.py`): it opens its store in its own `start()`, contributes `memory_search`, `memory_get` and `remember`, and its `context()` renders the `## Division Context` and `## Agent Memory` sections. When the slot is occupied, the agent loop asks `context(turn, budget)` on every turn, right after the guardrails. `budget` is a `ContextBudget` (`max_chars`, `max_session_history`), and the answer is a `ContextContribution` whose `sections` are (heading, markdown) pairs; core adds each non-empty one as `## <heading>`, and an empty contribution adds nothing. The budget is core's ceiling, not a share: `max_chars` is the usable context window (the window minus the reply reserve) in characters, and `max_session_history` is at most 200 entries; the occupant renders within the smaller of its own `agent.memory` settings and that ceiling. Core does not truncate what comes back; the occupant is trusted to keep to the budget. Every call into the occupant is contained. `browse()` returns a `MemoryBrowse` (`plugins/api.py`): five verbs, `search`, `get`, `edit`, `forget` and `promote`, and it serves the phone's memory screen. The bundled plugin's browse class (`StoreBrowse`, `memory/browse.py`) also offers an optional `store(name, content, *, confidence=1.0)` seeding verb and id-keyed reads for its own commands; they are not part of the Protocol, so callers probe for them with `getattr`. Both additions are listed under "Additive fields". `bind_subagent()` is asked once per cruncher run for a `MemoryWriteHandle`, a subagent's write access to memory with one verb, `persist_reduce_trace(question, trace)`; the bundled plugin returns none today, because subagent gist persistence is not implemented. With memory off (`memory.enabled: false`) the slot is empty unless you install and enable another memory plugin: no memory section, no memory tool and no `/memory` command exist, and the phone's memory routes answer 404. If the memory store cannot open, the plugin is failed for the session and the slot is empty in the same way. `/memory` is the plugin's own slash row (target `localharness.memory.plugin:MemoryPlugin.slash_memory`), called on the running plugin and reaching the store only through its browse class; `localharness memory` is its `CliDescriptor`, which opens the agent's store itself and wraps it in the same browse class. The bench starts the plugins through the same lifecycle over a throwaway config directory, for the scenarios that seed memory only, and seeds through the occupied slot's `browse().store`; it imports no memory module.

### Commands, slash commands and doctor

- **CLI commands.** Each `CliDescriptor(name, help, target)` of an ON plugin appears in `localharness --help`, read from the manifest. The module that `target` names (`"package.module:attr"`, a `typer.Typer`) is imported only when the command runs, and it runs as a subcommand of `localharness`; `localharness --help` lists plugin commands after the core ones. A core command wins a name clash, and when two plugins share a command name the first in start order wins, without a warning. The CLI resolves plugins only to list commands (`--help`) or to find a name no core command has. So a core command (`start`, `doctor`, `--version` and the rest) never loads a plugin to build the command list, while `localharness --help` imports each enabled plugin you installed: its package and plugin class, because the manifest is in code, and never a command module. A command that fails to import or raises prints `plugin <name>: command <cmd> …` and exits 1. A command of a bundled plugin that is off is not listed in `--help`, but its name is known from the manifest, so running it (with any arguments, `--help` included) prints ``command '<cmd>' is provided by the <plugin> plugin, which is off — run `localharness plugins enable <plugin>` `` to stderr, imports nothing of the plugin, and exits 4, outside the `experiment` verdict codes 0-3. This applies to bundled plugins only: a plugin you installed but have not enabled gets Click's `No such command` (exit 2), and so does an off plugin's command when the config cannot be read.
- **Slash commands.** Each `SlashDescriptor(name, help, target)` of a running plugin becomes a row in the one slash table (`cli/slash_commands.py`). The REPL dispatcher, `/help` and the input completer read that table, and the mobile channel's `/api/protocol` lists its rows in `commands[]`. A plugin's rows are in the table only while its session runs: they are added when the session starts and removed when it ends, so a client that reads `commands[]` before a session is live sees core's rows only. The reference phone page has no command menu. `target` names `async def f(ctx, args)`, imported the first time the command runs; a target `"package.module:Class.method"` that names a method of the plugin's own class is called on the running instance with `(ctx, args)`, which is how a handler reaches its plugin's live state. A handler may return text (shown as info), None or `""` (nothing shown), or a rich renderable (shown through the channel's renderable path). A name already in the table is skipped with a warning. A handler that raises, exits or returns anything else is reported to the user as a failure naming the plugin, and the session goes on. Plugin rows follow core rows, so `/help` and the completer list them after `/quit` and `/exit`. The Zed (ACP) channel does not read the table.
- **Doctor.** After core's checks, `localharness doctor` prints a Plugins section. An ON plugin is created and configured against a throwaway context (no LLM client, and a fresh bus, registry and hook system), and its `doctor()` checks are shown; a failing check is a doctor failure. An off or available plugin is listed with the command that turns it on. `skipped` and unconfigured plugins are warnings, as is a `needs-extra` plugin you installed; a bundled plugin waiting on its install extra is information, since nobody opted in. `failed` and `refused` ones are failures. Doctor never starts a plugin, and it does not show which plugin tools the root agent's floor will strip.

### Errors and refusals

Every shape below is copied from the code; `{...}` marks what is filled in.

- **Load plan.** `plugins list`, `doctor` and the startup summary show a plugin that is not on as `{state} — {reason}` (an off or available one as `{state} — turn on: localharness plugins enable {name}`). The reasons:
  - `failed`: `could not be imported: {Type}: {message}`; `it was not imported`; `invalid settings — {detail}`; `it marks agent.{name}.{path} machine-level only, which agent-level settings cannot enforce`; `requires {name}, which is {not installed | available but not enabled | missing its install extra | its state}`.
  - `skipped`: `its requires_localharness '{spec}' is not a version range`; `requires localharness {spec}, this is {version}`.
  - `needs-extra`: ``install `localharness[{extra}]` to use it``.
  - `refused`: `'{name}' is not a valid plugin name (a lower-case letter, then up to 63 lower-case letters, digits, '_' or '-')`; `its name collides with the core settings key '{name}'`; `the name '{name}' is already taken ({by whom})`; `it was found as '{name}' but its class {declares no manifest | names '{other}' in its manifest}`; `its kind is "memory" but it is not a MemorySlotPlugin`; `declares sections (claiming core settings), which only bundled plugins may do`; `{a}, {b} each claim the memory slot, which holds one plugin — turn all but one off`; `dependency cycle: {a} → {b} → {a}`.
- **Startup.** A plugin failed for the session: `plugin {name}: {reason}`; an unconfigured one: `plugin {name}: unconfigured — set {key}`. Both are warnings: the session starts.
- **Settings.** A section no core key and no installed plugin owns is refused, naming the file and line: ``not a LocalHarness setting, and no installed plugin is named `{key}` — if a plugin you removed used it, reinstall that plugin or delete this section`` (``not a LocalHarness setting — did you mean `{key}`?`` when a core key is close, and `not an agent setting` in place of `not a LocalHarness setting` for an agent-level key). A project's value for a machine-level-only field is dropped with a warning: `ignoring {name}.{path} in {file}: only the global config may set it`, or, when a project value would replace a whole section holding such a field, `ignoring {name}.{replaced} in {file}: it would replace {name}.{path}, which only the global config may set`. The deprecated memory switch warns `org.memory_enabled is deprecated — use memory.enabled (read from {files})`. Which settings are machine-level only, and why, is SECURITY.md's (see [Plugins](../../SECURITY.md#plugins)).
- **Channels** (`start --channel`, never a fall-back to the terminal). A name no core channel and no bundled channel plugin declares, before any config is read: `unknown channel '{given}'; choose one of: {known}`. A channel served by its own command (`mobile`, `acp`): ``the {name} channel is served by its own command, because {why}. Run `localharness {name}` instead of `localharness start --channel {name}`.`` A known channel whose plugin is not on, after the plugins resolve: `channel '{name}' is provided by the {plugin} plugin, which is {state} — {fix}`, where `{state}` is the plan state (or `missing its install extra`) and `{fix}` is the enable command or the plan's reason. A channel whose plugin resolved on but did not start: `channel '{name}' is provided by the {plugin} plugin, which did not start — {reason}`.
- **Commands.** A command of a bundled plugin that is off prints ``command '{name}' is provided by the {plugin} plugin, which is off — run `{fix}` `` to stderr and exits 4 (exit codes: [spec 10, "Exit Codes"](10-cli.md#exit-codes)); it imports nothing of the plugin. Two cases do not get that hint: when the config cannot be read the command is unknown to Click (`No such command`, exit 2), and a plugin command that cannot be imported or raises prints `plugin {plugin}: command {name} could not be imported: {Type}: {message}` (or `raised` in place of `could not be imported:`) and exits 1. A plugin you installed but have not enabled gets Click's `No such command` (exit 2).

### Design notes

**Declarations, not names.** The permission gate, the capability floor and the context store decide what a plugin's tool may do from what the tool declares (`ingest`, `host`, `result_origin`, `gate_family`), not from its name or from which plugin sent it. Each declaration defaults to its riskiest value, so a tool that says nothing gets the least. This is the principle Saltzer and Schroeder call fail-safe defaults: "Base access decisions on permission rather than exclusion." ([The Protection of Information in Computer Systems, "Basic Principles Of Information Protection"](https://web.mit.edu/Saltzer/www/publications/protection/Basic.html)). The capability floor applies it once, to the registered tools at start, before the agent runs. The same paper's complete mediation, "Every access to every object must be checked for authority.", is what one gate reading one declaration aims at; the known gaps are stated where they live: the gate still recognises its own builtins by name, and a tool registered directly on `ctx.tools` escapes the gate-family rule (see "Lifecycle and containment"). The trust stance is [SECURITY.md's](../../SECURITY.md#plugins).

**One declaration, three readers.** Each safety fact about a tool is written once, on its `ToolSchema`, and three parts of core read it: the gate (`gate_family`), the capability floor (`ingest`, `host`, and `gate_family` to find exec surfaces) and the context store (`result_origin`). If each kept its own list of tools, the lists could disagree, and the gate could allow a tool that the floor never saw as dangerous. With one source, changing a declaration changes what all three see. For a plugin you installed, `plugins/trust.py` rewrites the declaration before it is registered, so the clamp is applied once too, ahead of all three readers.

**Discovery reads metadata; installed is not enabled.** Plugins are found through the packaging entry-points mechanism: "Entry points are a mechanism for an installed distribution to advertise components it provides to be discovered and used by other code", stored as metadata "read at runtime with importlib.metadata" ([PyPA, Entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/)). LocalHarness reads only that metadata to list a plugin you installed, and imports it only once it is enabled at machine level. Installing a package therefore grants it nothing: until it is enabled it runs no code in LocalHarness, which is Saltzer and Schroeder's least privilege ("Every program and every user of the system should operate using the least set of privileges necessary to complete the job."). The cost is at the other end: once enabled, its package and plugin class are imported for `localharness --help`, because the manifest is in code.

**The memory slot holds one plugin.** This is a LocalHarness design choice; no prior-art source for it was found. Memory is cross-cutting: it writes into every turn's prompt, inside one budget, and owns the store that the memory tools, `/memory` and the phone's memory screen all reach. Two occupants would mean two answers to "what does the agent remember" and two writers to one budget, so the slot holds at most one, and two memory plugins on at once are all refused rather than one being picked without telling you. The cost is that two memory plugins cannot be combined.

**Additive versioning.** `PLUGIN_API_VERSION` stays `"1"` because every change since the first release added something optional whose default is the old behaviour (see "Additive fields"). [RFC 9413, "Maintaining Robust Protocols"](https://www.rfc-editor.org/rfc/rfc9413.html) (Thomson and Schinazi, June 2023) says that tolerating unexpected input "is no longer considered best practice in all scenarios", and that "A well-designed extensibility mechanism establishes clear rules for the handling of elements like new messages or parameters". The rule here is that clear for core reading an older plugin. It is not strict the other way: an older LocalHarness silently ignores a manifest field it does not know, so a plugin that depends on an addition should say so with `requires_localharness`, which is the check that runs.
