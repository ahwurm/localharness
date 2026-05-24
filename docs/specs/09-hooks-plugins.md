# Spec 09: Hooks and Plugins

**Component:** `src/localharness/tools/hooks.py`, `src/localharness/plugins/`
**Requirements covered:** HOOK-01, HOOK-02, HOOK-03
**Dependencies:** `tools/registry.py` (spec 04), `core/events.py`
**Library:** pluggy 1.6.0, importlib.metadata (stdlib)
**Stability:** UNSTABLE (v1)

---

## Purpose

LocalHarness has two orthogonal extension mechanisms:

1. **Tool plugins** — packages that contribute new tools to the `ToolRegistry`. Discovered via `importlib.metadata` entry points. Every tool plugin registers its tools under the group `localharness.tools`. This is the mechanism for tool packs: "install this package, get these tools."

2. **Hook plugins** — packages that attach behavior to lifecycle events. Implemented via `pluggy`. Hooks fire before/after tool calls, at agent start/end, and on bus events. This is the mechanism for cross-cutting concerns: audit logging, lint gates, risk annotation, metrics.

A plugin package may implement both mechanisms (contribute tools AND register hooks), but the mechanisms are independent. A pure tool plugin need not implement any hooks. A pure hook plugin need not register any tools.

---

## Hook Specifications

Hook specs are the contracts. They define what hooks exist, what arguments they receive, and their calling convention. LocalHarness defines one `HookSpec` class.

```python
# src/localharness/tools/hooks.py
import pluggy
from typing import Any

# The project name used for pluggy's internal namespace.
HARNESS_HOOKSPEC = pluggy.HookspecMarker("localharness")
HARNESS_HOOKIMPL = pluggy.HookimplMarker("localharness")


class HarnesHookSpec:
    """Pluggy hook specifications for LocalHarness.

    All hooks are optional. Plugin implementations may implement any subset.
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

        Implementations MAY raise ToolVetoed to prevent execution. Any other
        exception is caught, logged, and does not block execution (plugins
        must not crash the harness on hook failure).

        Raise ToolVetoed to block:
            raise ToolVetoed("Reason the tool call is denied")

        Args:
            name: Tool name as registered in ToolRegistry.
            arguments: Validated (post-Pydantic) arguments dict. Read-only.
            agent_id: ID of the agent making the call.
            division_id: Division of the calling agent.

        Returns:
            None. Return value is ignored. Raise ToolVetoed to block.
        """

    @HARNESS_HOOKSPEC
    def post_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: "ToolResult",
        agent_id: str,
        division_id: str,
    ) -> None:
        """Called after a tool's run() method returns.

        Always called, even if run() returned success=False. Never called if
        pre_tool raised ToolVetoed (tool did not run).

        Exceptions raised here are caught, logged, and ignored. post_tool hooks
        must never block or fail loudly — they are for observability only.

        Args:
            name: Tool name.
            arguments: Validated arguments dict (same as passed to pre_tool).
            result: The ToolResult returned by run(). Read-only.
            agent_id: ID of the agent making the call.
            division_id: Division of the calling agent.
        """

    @HARNESS_HOOKSPEC
    def on_agent_start(
        self,
        agent_id: str,
        division_id: str,
        task: str,
        iteration_budget: int,
    ) -> None:
        """Called once when an agent loop starts a new turn.

        Exceptions raised here are caught and logged. Agent loop continues.

        Args:
            agent_id: Agent being started.
            division_id: Division of the agent.
            task: The task string passed to the agent.
            iteration_budget: max_actions configured for this agent.
        """

    @HARNESS_HOOKSPEC
    def on_agent_end(
        self,
        agent_id: str,
        division_id: str,
        summary: str,
        iterations_used: int,
        success: bool,
        error: str | None,
    ) -> None:
        """Called once when an agent loop ends (success or failure).

        Exceptions raised here are caught and logged.

        Args:
            agent_id: Agent that finished.
            division_id: Division of the agent.
            summary: The summary text the agent produced (if success=True).
            iterations_used: Number of ReAct iterations consumed.
            success: False if the loop ended due to budget, stuck detection, or exception.
            error: Error message if success=False, else None.
        """

    @HARNESS_HOOKSPEC
    def on_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        agent_id: str | None,
    ) -> None:
        """Called for every event emitted on the event bus.

        This is a broad hook for monitoring and audit. Implementations should
        be fast — they are called on the event bus dispatch path.

        Note: This hook is called synchronously on the event loop. If the
        implementation needs to do I/O (e.g. write to a database), it must
        schedule it as a background task, not await it inline.

        Args:
            event_type: Event class name, e.g. "Action", "Observation", "Heartbeat".
            event_data: Event fields as a dict. Structure varies by event_type.
            agent_id: The agent that emitted the event, or None for system events.
        """
```

### Hook Calling Convention

- **`pre_tool`**: `firstresult=False` — all implementations called. Any raising `ToolVetoed` blocks execution.
- **`post_tool`**: `firstresult=False` — all implementations called.
- **`on_agent_start`**: `firstresult=False`.
- **`on_agent_end`**: `firstresult=False`.
- **`on_event`**: `firstresult=False`.

No hook uses `firstresult=True` — all implementations always fire.

---

## Hook Implementation Pattern

Plugin authors implement hooks by decorating methods with `@HARNESS_HOOKIMPL`:

```python
# Example: my_plugin/hooks.py
from localharness.tools.hooks import HARNESS_HOOKIMPL, ToolVetoed
from typing import Any

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
        # Emit a metric (fire-and-forget via background task)
        import asyncio
        if not result.success:
            asyncio.get_event_loop().call_soon(
                lambda: print(f"[METRIC] tool_failure agent={agent_id} tool={name}")
            )

    @HARNESS_HOOKIMPL
    def on_agent_end(
        self,
        agent_id: str,
        division_id: str,
        summary: str,
        iterations_used: int,
        success: bool,
        error: str | None,
    ) -> None:
        print(f"[AUDIT] {agent_id} finished success={success} iters={iterations_used}")
```

Hook implementations **do not** need to implement all hook methods. Pluggy only calls methods that are decorated with `@HARNESS_HOOKIMPL`.

---

## `HookSystem` Class

The `HookSystem` is the runtime manager: it creates the pluggy `PluginManager`, discovers and registers all plugins, and exposes the `pm` (PluginManager) for the `ToolRegistry` and agent loop to call.

```python
# src/localharness/tools/hooks.py (continued)
import importlib.metadata
import structlog

log = structlog.get_logger(__name__)


class HookSystem:
    """Manages plugin discovery, registration, and hook dispatch.

    Instantiated once at harness startup. The ToolRegistry receives references
    to pre_tool and post_tool callers via register_pre_hook / register_post_hook.
    The agent loop calls on_agent_start and on_agent_end directly via pm.hook.
    """

    def __init__(self) -> None:
        self.pm = pluggy.PluginManager("localharness")
        self.pm.add_hookspecs(HarnesHookSpec)
        self._loaded_plugins: list[str] = []

    async def discover_and_register(self) -> None:
        """Discover all plugins and register their hook implementations.

        Discovery happens in two passes:
        1. importlib.metadata entry points (group="localharness.hooks") for hook-only plugins
        2. Plugin manifests (loaded in plugin discovery flow) for combined tool+hook plugins

        This method is idempotent — safe to call multiple times (re-registration
        of the same plugin name is a no-op with a warning).
        """
        # Hook-only plugins register via entry_points group "localharness.hooks"
        eps = importlib.metadata.entry_points(group="localharness.hooks")
        for ep in eps:
            await self._load_hook_plugin(ep)

    async def _load_hook_plugin(self, ep: importlib.metadata.EntryPoint) -> None:
        plugin_name = ep.name
        try:
            impl_class = ep.load()
            instance = impl_class()
            if not self.pm.is_registered(instance):
                self.pm.register(instance, name=plugin_name)
                self._loaded_plugins.append(plugin_name)
                log.info("hook_plugin_loaded", plugin=plugin_name)
            else:
                log.warning("hook_plugin_already_registered", plugin=plugin_name)
        except Exception as exc:
            # Plugin load failure is non-fatal. Log and continue.
            # If a hook plugin fails to load, the harness still runs —
            # hooks are optional. A missing security hook IS dangerous,
            # so log at ERROR level.
            log.error(
                "hook_plugin_load_failed",
                plugin=plugin_name,
                error=str(exc),
                exc_info=True,
            )

    def register_impl(self, instance: object, name: str) -> None:
        """Register a hook implementation instance directly.

        Used by the plugin loader (PluginLoader) after loading a combined
        tool+hook plugin from its manifest.

        Args:
            instance: Object with @HARNESS_HOOKIMPL-decorated methods.
            name: Unique plugin name for pluggy tracking.
        """
        if self.pm.is_registered(instance):
            log.warning("hook_impl_already_registered", name=name)
            return
        self.pm.register(instance, name=name)
        self._loaded_plugins.append(name)

    def wire_to_registry(self, registry: "ToolRegistry") -> None:
        """Connect hook dispatch to ToolRegistry.

        Must be called after discover_and_register() and before any agents run.
        """
        async def pre_hook_caller(
            name: str, arguments: dict, agent_id: str, **_: Any
        ) -> None:
            # pm.hook.pre_tool is a sync call to pluggy's caller infrastructure.
            # Each hookimpl may be sync or async. LocalHarness requires sync hookimpls
            # for pre_tool (they run on the hot path). Async hookimpls are NOT
            # supported for pre_tool — pluggy does not await coroutines by default.
            self.pm.hook.pre_tool(
                name=name,
                arguments=arguments,
                agent_id=agent_id,
                division_id=_get_division(agent_id),
            )

        async def post_hook_caller(
            name: str, arguments: dict, result: Any, agent_id: str, **_: Any
        ) -> None:
            self.pm.hook.post_tool(
                name=name,
                arguments=arguments,
                result=result,
                agent_id=agent_id,
                division_id=_get_division(agent_id),
            )

        registry.register_pre_hook(pre_hook_caller)
        registry.register_post_hook(post_hook_caller)

    def call_agent_start(
        self, agent_id: str, division_id: str, task: str, iteration_budget: int
    ) -> None:
        """Called by the agent loop at the start of each turn."""
        try:
            self.pm.hook.on_agent_start(
                agent_id=agent_id,
                division_id=division_id,
                task=task,
                iteration_budget=iteration_budget,
            )
        except Exception as exc:
            log.warning("on_agent_start_hook_error", error=str(exc))

    def call_agent_end(
        self,
        agent_id: str,
        division_id: str,
        summary: str,
        iterations_used: int,
        success: bool,
        error: str | None,
    ) -> None:
        """Called by the agent loop at the end of each turn."""
        try:
            self.pm.hook.on_agent_end(
                agent_id=agent_id,
                division_id=division_id,
                summary=summary,
                iterations_used=iterations_used,
                success=success,
                error=error,
            )
        except Exception as exc:
            log.warning("on_agent_end_hook_error", error=str(exc))

    def call_on_event(
        self, event_type: str, event_data: dict, agent_id: str | None
    ) -> None:
        """Called by the event bus subscriber after each event emission."""
        try:
            self.pm.hook.on_event(
                event_type=event_type,
                event_data=event_data,
                agent_id=agent_id,
            )
        except Exception as exc:
            log.warning("on_event_hook_error", error=str(exc), event_type=event_type)

    @property
    def loaded_plugin_names(self) -> list[str]:
        return list(self._loaded_plugins)
```

**Note on async hooks:** pluggy does not natively await coroutines. LocalHarness requires all `pre_tool` and `post_tool` hook implementations to be synchronous. `on_event` implementations must also be synchronous (they schedule async work via `asyncio.get_event_loop().call_soon()` or create tasks). This constraint is documented in `HarnesHookSpec` and enforced by a type check at registration time in `register_impl()`.

---

## Plugin Manifest Format

Every plugin package (tool plugin, hook plugin, or combined) includes a `manifest.yaml` at its package root. This is the human-readable declaration of what the plugin provides.

```yaml
# manifest.yaml schema
# Located at: <package_root>/manifest.yaml

# Required fields:
name: my-search-tools          # Unique plugin name. snake_case or kebab-case.
version: "1.2.0"               # Semver.
description: "Web search tools via Exa API"  # One-line description.
author: "Your Name <you@example.com>"

# Stability marker. LocalHarness v1 marks its own extension API as UNSTABLE.
# Plugin authors declare their own stability level.
stability: UNSTABLE             # UNSTABLE | STABLE | DEPRECATED

# Minimum LocalHarness version required.
requires_localharness: ">=0.1.0"

# Python package dependencies (passed to pip/uv at install time).
dependencies:
  - exa-py>=1.0.0

# Tools this plugin contributes (tool plugins only).
# Each entry is loaded by the plugin loader and registered in ToolRegistry.
tools:
  - name: exa_search            # Must match ToolSchema.name returned by info()
    scope: global               # global | division | agent
    # If scope=division or scope=agent, add:
    # division_id: financial
    # agent_id: morning-briefing
    entrypoint: my_search_tools.tools:ExaSearchTool  # Python import path
    description: "Search the web using Exa semantic search"

  - name: exa_crawl
    scope: global
    entrypoint: my_search_tools.tools:ExaCrawlTool
    description: "Crawl a URL and return its content"

# Hook implementations this plugin contributes (hook plugins only).
hooks:
  - entrypoint: my_search_tools.hooks:MySearchHooks
    # Class must have @HARNESS_HOOKIMPL-decorated methods.

# Optional: Configuration schema this plugin accepts.
# If present, users may configure the plugin under plugins.<name> in
# ~/.localharness/config.yaml. Values are passed to tool constructors.
config_schema:
  type: object
  properties:
    api_key:
      type: string
      description: "Exa API key"
    timeout_s:
      type: number
      default: 30
  required: [api_key]
```

### Manifest validation rules

- `name` must be unique across all loaded plugins. Collision on load raises `PluginConflictError`.
- `version` must be valid semver. Invalid versions fail validation.
- `stability: UNSTABLE` is the only valid value for v1 plugins. If a plugin declares `STABLE`, the harness emits a warning (the harness API itself is UNSTABLE; stable plugins cannot make stability guarantees).
- `entrypoint` format: `package.module:ClassName`. The class must be importable and must satisfy `ToolProtocol` (for tools) or have at least one `@HARNESS_HOOKIMPL` method (for hooks).
- `requires_localharness` is checked against `localharness.__version__` at load time. Version mismatch raises `PluginVersionError` (non-fatal — logged and skipped).

---

## Plugin Discovery via importlib.metadata

```python
# src/localharness/plugins/loader.py
import importlib.metadata
import importlib
from pathlib import Path
import yaml
import structlog

log = structlog.get_logger(__name__)


class PluginLoader:
    """Discovers, validates, and loads all plugins.

    Two discovery paths:
    1. Entry points: packages declare tools via pyproject.toml entry_points.
       Group "localharness.tools" for tool classes.
       Group "localharness.hooks" for hook implementation classes.
    2. Manifest-based: packages include manifest.yaml; the loader finds all
       installed packages with a manifest.yaml at their package root.
       (Used for combined tool+hook plugins and config-aware plugins.)

    At startup, PluginLoader is called before ToolRegistry.register_builtin_tools()
    but after the registry and hook system are constructed.
    """

    def __init__(self, registry: "ToolRegistry", hook_system: "HookSystem") -> None:
        self._registry = registry
        self._hook_system = hook_system
        self._loaded: dict[str, "PluginManifest"] = {}

    async def discover_all(self) -> None:
        """Run full discovery. Order:
            1. Discover tool entry points
            2. Discover hook entry points
            3. Discover manifest-based plugins (catches combined plugins)
        """
        await self._discover_tool_entry_points()
        await self._hook_system.discover_and_register()
        await self._discover_manifest_plugins()

    async def _discover_tool_entry_points(self) -> None:
        """Load tools registered under group 'localharness.tools'."""
        eps = importlib.metadata.entry_points(group="localharness.tools")
        for ep in eps:
            await self._load_tool_entry_point(ep)

    async def _load_tool_entry_point(self, ep: importlib.metadata.EntryPoint) -> None:
        tool_name = ep.name
        try:
            tool_class = ep.load()
            instance = tool_class()
            schema = instance.info()
            await self._registry.register(instance, scope=schema.scope or "global")
            log.info("tool_plugin_loaded_via_entrypoint", tool=tool_name)
        except Exception as exc:
            log.error(
                "tool_plugin_entrypoint_failed",
                tool=tool_name,
                error=str(exc),
                exc_info=True,
            )

    async def _discover_manifest_plugins(self) -> None:
        """Find all installed packages with a manifest.yaml."""
        for dist in importlib.metadata.distributions():
            # Check if this distribution has a manifest.yaml file listed
            for file in dist.files or []:
                if file.name == "manifest.yaml":
                    manifest_path = file.locate()
                    await self._load_manifest_plugin(manifest_path, dist.name)
                    break

    async def _load_manifest_plugin(
        self, manifest_path: Path, dist_name: str
    ) -> None:
        try:
            manifest_data = yaml.safe_load(manifest_path.read_text())
            manifest = PluginManifest(**manifest_data)
        except Exception as exc:
            log.error(
                "manifest_parse_failed",
                dist=dist_name,
                path=str(manifest_path),
                error=str(exc),
            )
            return

        if manifest.name in self._loaded:
            log.warning("manifest_plugin_already_loaded", name=manifest.name)
            return

        # Version check
        if not _version_compatible(manifest.requires_localharness):
            log.warning(
                "manifest_plugin_version_mismatch",
                plugin=manifest.name,
                requires=manifest.requires_localharness,
            )
            return

        # Load tools from manifest
        for tool_entry in manifest.tools:
            await self._load_manifest_tool(tool_entry, manifest.name)

        # Load hooks from manifest
        for hook_entry in manifest.hooks:
            await self._load_manifest_hook(hook_entry, manifest.name)

        self._loaded[manifest.name] = manifest
        log.info("manifest_plugin_loaded", name=manifest.name, version=manifest.version)

    async def _load_manifest_tool(self, entry: "ToolEntry", plugin_name: str) -> None:
        try:
            module_path, class_name = entry.entrypoint.rsplit(":", 1)
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            instance = tool_class()
            await self._registry.register(
                instance,
                scope=entry.scope,
                division_id=getattr(entry, "division_id", None),
                agent_id=getattr(entry, "agent_id", None),
            )
            log.info(
                "manifest_tool_registered", tool=entry.name, plugin=plugin_name
            )
        except Exception as exc:
            log.error(
                "manifest_tool_load_failed",
                tool=entry.name,
                plugin=plugin_name,
                error=str(exc),
                exc_info=True,
            )

    async def _load_manifest_hook(self, entry: "HookEntry", plugin_name: str) -> None:
        try:
            module_path, class_name = entry.entrypoint.rsplit(":", 1)
            module = importlib.import_module(module_path)
            hook_class = getattr(module, class_name)
            instance = hook_class()
            self._hook_system.register_impl(instance, name=f"{plugin_name}.{class_name}")
            log.info(
                "manifest_hook_registered", plugin=plugin_name, class_name=class_name
            )
        except Exception as exc:
            log.error(
                "manifest_hook_load_failed",
                plugin=plugin_name,
                error=str(exc),
                exc_info=True,
            )
```

### Pydantic models for manifest parsing

```python
from pydantic import BaseModel
from typing import Literal

class ToolEntry(BaseModel):
    name: str
    scope: Literal["global", "division", "agent"] = "global"
    entrypoint: str  # "package.module:ClassName"
    description: str = ""
    division_id: str | None = None
    agent_id: str | None = None

class HookEntry(BaseModel):
    entrypoint: str  # "package.module:ClassName"

class PluginManifest(BaseModel):
    name: str
    version: str
    description: str
    author: str = ""
    stability: Literal["UNSTABLE", "STABLE", "DEPRECATED"] = "UNSTABLE"
    requires_localharness: str = ">=0.1.0"
    dependencies: list[str] = []
    tools: list[ToolEntry] = []
    hooks: list[HookEntry] = []
    config_schema: dict | None = None
```

---

## Plugin Lifecycle

```
discover_all() called at harness startup
  │
  ├─ [Phase 1] Entry point scan: importlib.metadata.entry_points()
  │    group="localharness.tools"  → each → ToolRegistry.register()
  │    group="localharness.hooks"  → each → HookSystem.register_impl()
  │
  ├─ [Phase 2] Manifest scan: iterate all installed distributions
  │    For each dist with manifest.yaml:
  │      ├─ Parse + validate PluginManifest (Pydantic)
  │      ├─ Version check against localharness.__version__
  │      ├─ Load tools → ToolRegistry.register()
  │      └─ Load hooks → HookSystem.register_impl()
  │
  └─ [Phase 3] HookSystem.wire_to_registry(registry)
       Connects pluggy pm.hook.pre_tool / post_tool to ToolRegistry dispatch
```

**Activate**: Plugins are "active" once registered. There is no separate activate step — registered tools are immediately available in the ToolRegistry; registered hooks are immediately called by pluggy.

**No hot-reload in v1.** Plugin changes require harness restart. Dynamic reload is v2.

---

## Tool Plugins vs Hook Plugins

| Dimension | Tool Plugin | Hook Plugin |
|-----------|-------------|-------------|
| Discovery group | `localharness.tools` entry point OR manifest `tools:` section | `localharness.hooks` entry point OR manifest `hooks:` section |
| Base contract | Implements `ToolProtocol` (`info()` + `run()`) | Has `@HARNESS_HOOKIMPL` methods on hookspecs |
| Effect | Adds tools to ToolRegistry | Fires callbacks at lifecycle events |
| Failure impact | Tool not available; logged at ERROR | Hook not called for that event; logged at ERROR |
| Can veto tool calls | No (tools don't observe other tools) | Yes, via `pre_tool` raising `ToolVetoed` |
| Async support | Yes — `run()` is an async method | No — hookimpl methods must be synchronous (schedule async work via `call_soon`) |
| State | Stateless preferred; stateful tools use own lock | Stateless strongly preferred |

---

## Plugin Versioning and Stability

All v1 LocalHarness hook and tool APIs are marked **UNSTABLE**. This means:

- The `HarnesHookSpec` signatures may change between minor versions.
- `ToolProtocol` may gain new required methods.
- `manifest.yaml` schema may gain required fields.

Plugin authors must pin `requires_localharness` to a specific minor version, e.g. `>=0.1.0,<0.2.0`.

The harness enforces this:
- `PluginLoader._load_manifest_plugin()` checks `requires_localharness` against `localharness.__version__`.
- Incompatible plugins are **skipped with a WARNING**, not a hard failure.
- The `localharness doctor` command reports all skipped plugins.

Stability will be bumped to STABLE when the hook API reaches v1.0 (post first stable release).

---

## Error Handling Reference

| Situation | Behavior |
|-----------|----------|
| Tool entry point fails to import | `log.error(...)`, tool not registered, harness continues |
| Tool entry point class not found | `log.error(...)`, tool not registered, harness continues |
| Tool name collision in registry | `log.warning(...)`, second registration skipped |
| Manifest parse error | `log.error(...)`, entire plugin skipped |
| Manifest version mismatch | `log.warning(...)`, plugin skipped |
| Hook entry point fails to import | `log.error(...)`, hook not registered, harness continues |
| `pre_tool` raises `ToolVetoed` | Tool execution blocked; `ToolResult(error_type="permission_denied")` returned |
| `pre_tool` raises other exception | `log.warning(...)`, exception swallowed, execution continues |
| `post_tool` raises any exception | `log.warning(...)`, exception swallowed, result returned unchanged |
| `on_agent_start` raises | `log.warning(...)`, agent loop continues |
| `on_agent_end` raises | `log.warning(...)`, end handling continues |
| `on_event` raises | `log.warning(...)`, event processing continues |
| Plugin `config_schema` missing required field | `log.error(...)`, plugin's tools registered but with default config |

**The harness never crashes due to a plugin.** All plugin errors are contained.

---

## Example Plugin: Complete Working Implementation

This section is a complete, working plugin that contributes one tool (`lint_python`) and one hook (a post_tool hook that logs tool failures to a file). Copy it as a template.

### File layout

```
localharness-lint-plugin/
├── pyproject.toml
├── manifest.yaml
└── localharness_lint/
    ├── __init__.py
    ├── tools.py
    └── hooks.py
```

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "localharness-lint-plugin"
version = "0.1.0"
description = "Lint tool and failure-logging hook for LocalHarness"
requires-python = ">=3.12"
dependencies = []

# Entry points for discovery without manifest.yaml.
# The manifest.yaml is the primary mechanism; entry points are the fallback.
[project.entry-points."localharness.tools"]
lint_python = "localharness_lint.tools:LintPythonTool"

[project.entry-points."localharness.hooks"]
lint_failure_logger = "localharness_lint.hooks:LintFailureLoggerHooks"
```

### `manifest.yaml`

```yaml
name: localharness-lint-plugin
version: "0.1.0"
description: "Lint Python files and log tool failures"
author: "Example Author <example@example.com>"
stability: UNSTABLE
requires_localharness: ">=0.1.0,<0.2.0"
dependencies: []

tools:
  - name: lint_python
    scope: global
    entrypoint: localharness_lint.tools:LintPythonTool
    description: "Run ruff linter on a Python file or directory"

hooks:
  - entrypoint: localharness_lint.hooks:LintFailureLoggerHooks
```

### `localharness_lint/tools.py`

```python
import asyncio
import shutil
from pathlib import Path
from localharness.tools.base import Tool, ToolSchema, ToolResult


class LintPythonTool(Tool):
    """Run ruff linter on a Python file or directory.

    Returns lint output (empty string = no issues). Fails with an error result
    if ruff is not installed.
    """

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="lint_python",
            description=(
                "Run the ruff linter on a Python file or directory. "
                "Returns lint findings or empty string if clean."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to Python file or directory to lint.",
                    },
                    "fix": {
                        "type": "boolean",
                        "description": "Apply auto-fixes where possible.",
                        "default": False,
                    },
                },
                "required": ["path"],
            },
            destructive=False,
            estimated_tokens=300,
            version="0.1.0",
        )

    async def _execute(self, path: str, fix: bool = False) -> ToolResult:
        if shutil.which("ruff") is None:
            return self.err(
                "ruff is not installed. Install with: pip install ruff",
                error_type="execution_error",
            )

        target = Path(path).resolve()
        if not target.exists():
            return self.err(f"Path does not exist: {target}", error_type="not_found")

        cmd = ["ruff", "check", str(target)]
        if fix:
            cmd.append("--fix")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        output = stdout.decode("utf-8", errors="replace").strip()

        if proc.returncode == 0:
            return self.ok("(no lint issues)")
        return self.ok(output, exit_code=proc.returncode)
```

### `localharness_lint/hooks.py`

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from localharness.tools.hooks import HARNESS_HOOKIMPL


class LintFailureLoggerHooks:
    """Log tool failures to ~/.localharness/tool-failures.jsonl."""

    LOG_PATH = Path.home() / ".localharness" / "tool-failures.jsonl"

    @HARNESS_HOOKIMPL
    def post_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        agent_id: str,
        division_id: str,
    ) -> None:
        """Log failed tool calls to a JSONL file."""
        if result.success:
            return  # Only log failures

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "division_id": division_id,
            "tool": name,
            "error_type": result.error_type,
            "error": result.error,
        }

        # Synchronous append — safe because POSIX O_APPEND writes are atomic
        # for records < PIPE_BUF (4096 bytes). Our records are well under that.
        self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self.LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    @HARNESS_HOOKIMPL
    def on_agent_end(
        self,
        agent_id: str,
        division_id: str,
        summary: str,
        iterations_used: int,
        success: bool,
        error: str | None,
    ) -> None:
        """Print a summary line when an agent finishes."""
        status = "OK" if success else f"FAILED: {error}"
        print(
            f"[lint-plugin] {agent_id} finished after {iterations_used} iterations: {status}"
        )
```

### Install and verify

```bash
# Development install
cd localharness-lint-plugin
uv pip install -e .

# Verify discovery
localharness doctor
# Expected output includes:
# [PASS] Plugin: localharness-lint-plugin 0.1.0 (tools: lint_python, hooks: lint_failure_logger)
```

---

## Harness Startup Sequence (hooks-relevant excerpt)

```python
# src/localharness/tools/__init__.py

async def build_tool_system(
    harness_config: "HarnessConfig",
) -> tuple["ToolRegistry", "HookSystem", "PluginLoader"]:
    """Construct and wire the full tool system. Called once at startup."""
    from localharness.tools.registry import ToolRegistry
    from localharness.tools.hooks import HookSystem
    from localharness.plugins.loader import PluginLoader
    from localharness.tools.builtin import register_builtin_tools

    registry = ToolRegistry(
        default_timeout_s=harness_config.tools.default_timeout_s,
        result_size_cap_chars=harness_config.tools.result_size_cap_chars,
    )
    hook_system = HookSystem()
    loader = PluginLoader(registry, hook_system)

    # Phase 1: Register built-in tools (always present, not pluggable)
    await register_builtin_tools(registry)

    # Phase 2: Discover and register all plugins
    await loader.discover_all()

    # Phase 3: Wire hooks to registry dispatch
    hook_system.wire_to_registry(registry)

    return registry, hook_system, loader
```
