# Spec 04: Tool System

**Component:** `src/localharness/tools/`
**Requirements covered:** TOOL-01, TOOL-02, TOOL-03, TOOL-04
**Dependencies:** `core/events.py`, `config/models.py`
**Stability:** UNSTABLE (v1)

---

## Purpose

The tool system is the execution layer between the agent loop and the external world. It provides:

1. A minimal, uniform interface (`info()` / `run()`) that every tool — built-in or third-party — must implement.
2. A registry that resolves which tools are available to a given agent based on its scope configuration (global → division → agent → MCP).
3. Pydantic-driven argument validation before any `run()` call fires.
4. A thread-safe async execution model with timeouts.
5. Pre/post hook integration points (hooks are defined separately in spec 09; this spec defines where they are called).

The tool system does **not** manage hook implementations, MCP transport (spec 04b), or the agent loop ReAct cycle. It is a pure service: given a tool name and arguments, validate and execute.

**Source pattern:** OpenCode minimal `info()/run()` interface. OpenHands typed tool system (V1 Nov 2025).

---

## Data Structures

### `ToolParameter`

A single parameter in a tool's JSON schema.

```python
# src/localharness/tools/base.py
from pydantic import BaseModel, ConfigDict
from typing import Any, Literal

class ToolParameter(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["string", "integer", "number", "boolean", "array", "object"]
    description: str
    # For string params with a fixed set of values:
    enum: list[str] | None = None
    # For array params:
    items: dict[str, Any] | None = None
    # For object params:
    properties: dict[str, "ToolParameter"] | None = None
    required: list[str] | None = None
    # Constraints:
    min_length: int | None = None
    max_length: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    default: Any | None = None
```

### `ToolSchema`

The self-description returned by `info()`. Matches the OpenAI function-calling schema format so it can be passed directly to the LLM provider without transformation.

```python
class ToolSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Unique identifier — must match the name used in tool registry keys.
    # Convention: snake_case, e.g. "glob", "bash_exec", "exa_search"
    name: str

    # One-sentence description shown in LLM tool list. Be precise — vague
    # descriptions degrade tool selection accuracy on local models.
    description: str

    # JSON Schema object for the parameters. Must have:
    #   "type": "object"
    #   "properties": { ... }
    #   "required": [ ... ]  (list only truly required params)
    parameters: dict[str, Any]

    # Scope this tool was registered at. Set by ToolRegistry at registration time.
    # Agents see this in get_tools_for_agent() output for diagnostics.
    scope: Literal["global", "division", "agent", "mcp"] = "global"

    # Approximate token cost of calling this tool (input + typical output).
    # Used by context manager to warn if tool set exceeds budget.
    # Estimate conservatively. None = unknown.
    estimated_tokens: int | None = None

    # Version string for the tool implementation. Semver preferred.
    version: str = "1.0.0"

    # If True, the tool makes irreversible changes (writes, deletes, network calls).
    # Used by the permission evaluator and inline risk annotation hook.
    destructive: bool = False
```

### `ToolResult`

The structured return value from `run()`. Always a `ToolResult`, never a bare string. The agent loop unwraps `output` when building the tool_result message for the LLM.

```python
class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    # The primary string output. Always present, even if empty.
    # For binary data: base64-encode and note in output.
    output: str

    # True if run() completed without error. False triggers error handling in loop.
    success: bool = True

    # Human-readable error message if success=False. None if success=True.
    error: str | None = None

    # Error category for structured handling by the agent loop.
    error_type: Literal[
        "validation_error",    # Argument validation failed (Pydantic)
        "execution_error",     # Tool raised an exception during run()
        "timeout_error",       # run() exceeded its timeout
        "permission_denied",   # Permission evaluator blocked the call
        "not_found",           # Tool name not in registry for this agent
    ] | None = None

    # Actual wall-clock milliseconds for this run() call. Set by ToolRegistry.
    duration_ms: int | None = None

    # Whether the output was truncated (tool result budget cap applied).
    truncated: bool = False

    # Original length before truncation, if truncated=True.
    original_length: int | None = None

    # Arbitrary metadata the tool wants to attach (e.g. file path, line count).
    # Not shown to LLM; available to hooks.
    metadata: dict[str, Any] = {}
```

---

## Tool Interface (Protocol)

Every tool — built-in, plugin, or MCP-wrapped — must satisfy this protocol. The `Tool` class is the implementation base; the `ToolProtocol` is the structural type used in type hints.

```python
# src/localharness/tools/base.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class ToolProtocol(Protocol):
    def info(self) -> ToolSchema:
        """Return the tool's self-description. Must be pure (no side effects).
        Called once at registration time and cached. May be called again if
        the registry is refreshed (MCP reconnect)."""
        ...

    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with validated keyword arguments.

        Contract:
        - kwargs are pre-validated by ToolRegistry before this is called.
          run() may assume all required params are present and type-correct.
        - run() must return a ToolResult. It must never raise — catch all
          exceptions internally and return ToolResult(success=False, ...).
        - run() must respect the timeout set in ToolRegistry._default_timeout_s.
          Use asyncio.wait_for() internally for sub-operations if needed.
        - run() must be safe to call concurrently (multiple agents may invoke
          the same tool instance in parallel).
        """
        ...
```

### Abstract Base Class

Plugin authors should subclass `Tool` rather than implementing `ToolProtocol` directly. The base class provides helpers for result construction.

```python
import asyncio
from abc import ABC, abstractmethod

class Tool(ABC):
    """Base class for all LocalHarness tools.

    Subclass and implement info() and _execute(). Do not override run() —
    it handles timeout wrapping and exception normalization.
    """

    # Override in subclass if this tool needs a different timeout.
    # None = use ToolRegistry default (30s).
    timeout_s: float | None = None

    @abstractmethod
    def info(self) -> ToolSchema:
        ...

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Implementation goes here. May raise — base class catches all exceptions."""
        ...

    async def run(self, **kwargs: Any) -> ToolResult:
        timeout = self.timeout_s or 30.0
        try:
            return await asyncio.wait_for(self._execute(**kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult(
                output="",
                success=False,
                error=f"Tool '{self.info().name}' timed out after {timeout}s",
                error_type="timeout_error",
            )
        except Exception as exc:
            return ToolResult(
                output="",
                success=False,
                error=str(exc),
                error_type="execution_error",
            )

    # Helpers for subclasses:

    def ok(self, output: str, **metadata: Any) -> ToolResult:
        return ToolResult(output=output, success=True, metadata=metadata)

    def err(
        self,
        message: str,
        error_type: str = "execution_error",
        **metadata: Any,
    ) -> ToolResult:
        return ToolResult(
            output="",
            success=False,
            error=message,
            error_type=error_type,  # type: ignore[arg-type]
            metadata=metadata,
        )
```

---

## ToolRegistry

The registry is the central service that maps tool names to implementations, enforces scope, and dispatches calls with validation.

```python
# src/localharness/tools/registry.py
import asyncio
import time
from collections.abc import Callable
from typing import Any

class ToolRegistry:
    """Thread-safe tool registry with scope resolution.

    Instantiated once at harness startup and shared across agent loops.
    Each call to get_tools_for_agent() returns a scoped view — a frozen
    dict of {name: ToolSchema} — without mutating registry state.
    """

    # Warn if any agent is given more than this many tools. Context window
    # degradation on local models begins above 15 tools (see FEATURES.md).
    TOOL_COUNT_WARNING_THRESHOLD: int = 15

    def __init__(
        self,
        default_timeout_s: float = 30.0,
        result_size_cap_chars: int = 50_000,
    ) -> None:
        # Nested dict: scope → tool_name → tool_instance
        # _tools["global"]["glob"] = GlobTool()
        self._tools: dict[str, dict[str, ToolProtocol]] = {
            "global": {},
            "division": {},
            "agent": {},
            "mcp": {},
        }
        # Cached schemas: tool_name → ToolSchema (populated on register)
        self._schemas: dict[str, ToolSchema] = {}
        # Division-scoped tools: division_id → {tool_name: tool_instance}
        self._division_tools: dict[str, dict[str, ToolProtocol]] = {}
        # Agent-scoped tools: agent_id → {tool_name: tool_instance}
        self._agent_tools: dict[str, dict[str, ToolProtocol]] = {}

        self._default_timeout_s = default_timeout_s
        self._result_size_cap_chars = result_size_cap_chars
        self._lock = asyncio.Lock()

        # Pre/post hook callables registered by the hook system (spec 09).
        # Called by dispatch() before and after run().
        self._pre_hooks: list[Callable] = []
        self._post_hooks: list[Callable] = []

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    async def register(
        self,
        tool: ToolProtocol,
        scope: str = "global",
        division_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Register a tool at the given scope.

        Args:
            tool: Must satisfy ToolProtocol. info() is called immediately to
                  cache the schema.
            scope: "global" | "division" | "agent" | "mcp"
            division_id: Required when scope="division".
            agent_id: Required when scope="agent".

        Raises:
            ValueError: If scope is "division" but division_id is None,
                        or scope is "agent" but agent_id is None,
                        or tool name is already registered at same scope+id.
            TypeError: If tool does not satisfy ToolProtocol.
        """
        if not isinstance(tool, ToolProtocol):
            raise TypeError(
                f"{type(tool).__name__} does not satisfy ToolProtocol "
                "(must implement info() and run())"
            )

        schema = tool.info()
        name = schema.name

        async with self._lock:
            if scope == "global":
                if name in self._tools["global"]:
                    raise ValueError(f"Tool '{name}' already registered at global scope")
                self._tools["global"][name] = tool
                self._schemas[name] = schema

            elif scope == "division":
                if division_id is None:
                    raise ValueError("division_id required for division-scoped tools")
                bucket = self._division_tools.setdefault(division_id, {})
                if name in bucket:
                    raise ValueError(
                        f"Tool '{name}' already registered for division '{division_id}'"
                    )
                bucket[name] = tool
                # Division schemas keyed as "division:{division_id}:{name}"
                self._schemas[f"division:{division_id}:{name}"] = schema

            elif scope == "agent":
                if agent_id is None:
                    raise ValueError("agent_id required for agent-scoped tools")
                bucket = self._agent_tools.setdefault(agent_id, {})
                if name in bucket:
                    raise ValueError(
                        f"Tool '{name}' already registered for agent '{agent_id}'"
                    )
                bucket[name] = tool
                self._schemas[f"agent:{agent_id}:{name}"] = schema

            elif scope == "mcp":
                # MCP tools live in the global MCP bucket. Name collisions with
                # higher-priority scopes are resolved at get_tools_for_agent() time.
                self._tools["mcp"][name] = tool
                self._schemas[f"mcp:{name}"] = schema

            else:
                raise ValueError(f"Unknown scope: '{scope}'")

    async def unregister(self, name: str, scope: str = "global", **scope_kwargs: str) -> None:
        """Remove a tool from the registry. Used by MCP integration on server disconnect."""
        async with self._lock:
            if scope == "global":
                self._tools["global"].pop(name, None)
                self._schemas.pop(name, None)
            elif scope == "mcp":
                self._tools["mcp"].pop(name, None)
                self._schemas.pop(f"mcp:{name}", None)
            elif scope == "division":
                division_id = scope_kwargs["division_id"]
                self._division_tools.get(division_id, {}).pop(name, None)
                self._schemas.pop(f"division:{division_id}:{name}", None)
            elif scope == "agent":
                agent_id = scope_kwargs["agent_id"]
                self._agent_tools.get(agent_id, {}).pop(name, None)
                self._schemas.pop(f"agent:{agent_id}:{name}", None)

    # -------------------------------------------------------------------------
    # Scope Resolution
    # -------------------------------------------------------------------------

    def get_tools_for_agent(
        self,
        agent_id: str,
        division_id: str,
        tool_config: "ToolConfig",  # from config/models.py
    ) -> dict[str, ToolSchema]:
        """Resolve the effective tool set for an agent.

        Resolution order (lower wins — later layers override earlier):
            1. Global tools (always included unless denied)
            2. Division tools for this agent's division
            3. Agent-specific tools added via tool_config.add
            4. MCP tools (lowest priority — overridden by any named tool above)

        Then apply:
            - tool_config.add: force-include named tools regardless of scope
            - tool_config.deny: remove named tools from the resolved set

        The "inherit" field in tool_config selects which scopes to pull from:
            inherit: [global]          — only global tools
            inherit: [global, division] — global + division (default)
            inherit: []                — start from empty (tool_config.add is the full set)

        Warn to stderr if result count > TOOL_COUNT_WARNING_THRESHOLD.

        Returns:
            Dict mapping tool name → ToolSchema. Immutable view.
        """
        resolved: dict[str, ToolProtocol] = {}

        inherit = tool_config.inherit if tool_config.inherit is not None else ["global", "division"]

        # Step 1: Add global tools if inherited
        if "global" in inherit:
            resolved.update(self._tools["global"])

        # Step 2: Add MCP tools (lowest priority — overridden by named scopes)
        if "mcp" in inherit or True:  # MCP always visible unless explicitly denied
            for name, tool in self._tools["mcp"].items():
                if name not in resolved:
                    resolved[name] = tool

        # Step 3: Add division tools if inherited
        if "division" in inherit:
            division_bucket = self._division_tools.get(division_id, {})
            resolved.update(division_bucket)

        # Step 4: Agent-specific tools (always applied regardless of inherit)
        agent_bucket = self._agent_tools.get(agent_id, {})
        resolved.update(agent_bucket)

        # Step 5: Force-add tools listed in tool_config.add
        for name in (tool_config.add or []):
            tool = self._find_tool_by_name(name)
            if tool is None:
                import warnings
                warnings.warn(
                    f"Agent '{agent_id}' tool_config.add contains unknown tool '{name}'",
                    stacklevel=2,
                )
            else:
                resolved[name] = tool

        # Step 6: Apply deny list (deny always wins — Claude Code deny-first pattern)
        for name in (tool_config.deny or []):
            resolved.pop(name, None)

        # Warning threshold
        if len(resolved) > self.TOOL_COUNT_WARNING_THRESHOLD:
            import warnings
            warnings.warn(
                f"Agent '{agent_id}' has {len(resolved)} tools "
                f"(threshold: {self.TOOL_COUNT_WARNING_THRESHOLD}). "
                "Context window degradation likely on models with <32K context. "
                "Use tool_config.deny to remove unused tools.",
                stacklevel=2,
            )

        # Return schemas only (not instances — callers should not call run() directly)
        return {name: tool.info() for name, tool in resolved.items()}

    def _find_tool_by_name(self, name: str) -> ToolProtocol | None:
        """Search all scopes for a tool by name. Returns first match."""
        for bucket in [
            self._tools["global"],
            self._tools["mcp"],
            *self._division_tools.values(),
            *self._agent_tools.values(),
        ]:
            if name in bucket:
                return bucket[name]
        return None

    def _get_tool_for_agent(
        self,
        name: str,
        agent_id: str,
        division_id: str,
        tool_config: "ToolConfig",
    ) -> ToolProtocol | None:
        """Get the executable tool instance for dispatch. Applies deny list."""
        if name in (tool_config.deny or []):
            return None
        # Check agent-specific first (highest priority)
        if name in self._agent_tools.get(agent_id, {}):
            return self._agent_tools[agent_id][name]
        # Then division
        if name in self._division_tools.get(division_id, {}):
            return self._division_tools[division_id][name]
        # Then global
        if name in self._tools["global"]:
            return self._tools["global"][name]
        # Then MCP
        if name in self._tools["mcp"]:
            return self._tools["mcp"][name]
        return None

    # -------------------------------------------------------------------------
    # Dispatch (validation + execution)
    # -------------------------------------------------------------------------

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        agent_id: str,
        division_id: str,
        tool_config: "ToolConfig",
    ) -> ToolResult:
        """Validate arguments and execute a tool.

        This is the single call site for tool execution. The agent loop calls
        dispatch() for every tool_call extracted from the LLM response.

        Flow:
            1. Resolve tool instance for this agent (scope + deny check)
            2. Validate arguments against the tool's JSON schema via Pydantic
            3. Call pre-hooks (from hook system — spec 09)
            4. Execute tool.run(**validated_args)
            5. Apply result size cap
            6. Call post-hooks
            7. Return ToolResult

        Pre-hook veto: If any pre-hook raises ToolVetoed, dispatch() returns
        a ToolResult(success=False, error_type="permission_denied") without
        calling run(). The exception message is used as the error string.

        Args:
            name: Tool name as it appears in the LLM tool_call.
            arguments: Raw arguments dict from LLM response (strings, numbers, etc.)
            agent_id: ID of the calling agent (for scope resolution and hooks).
            division_id: Division of the calling agent.
            tool_config: Agent's tool configuration (for scope resolution).

        Returns:
            ToolResult. Never raises.
        """
        start_ms = int(time.monotonic() * 1000)

        # Step 1: Resolve tool
        tool = self._get_tool_for_agent(name, agent_id, division_id, tool_config)
        if tool is None:
            return ToolResult(
                output="",
                success=False,
                error=f"Tool '{name}' not found or not permitted for agent '{agent_id}'",
                error_type="not_found",
            )

        # Step 2: Validate arguments
        validated = self._validate_arguments(name, arguments, tool.info())
        if isinstance(validated, ToolResult):
            # Validation failed — validated is already a ToolResult(success=False)
            return validated

        # Step 3: Pre-hooks
        for hook in self._pre_hooks:
            try:
                await _maybe_await(hook(name=name, arguments=validated, agent_id=agent_id))
            except ToolVetoed as exc:
                return ToolResult(
                    output="",
                    success=False,
                    error=str(exc),
                    error_type="permission_denied",
                    duration_ms=int(time.monotonic() * 1000) - start_ms,
                )
            except Exception:
                # Pre-hook exceptions (other than ToolVetoed) are logged but do not
                # block execution. Post-hook exceptions are always non-blocking.
                pass  # Hooks log their own errors via structlog

        # Step 4: Execute
        result = await tool.run(**validated)

        # Step 5: Apply result size cap
        if len(result.output) > self._result_size_cap_chars:
            result = ToolResult(
                output=result.output[: self._result_size_cap_chars],
                success=result.success,
                error=result.error,
                error_type=result.error_type,
                duration_ms=result.duration_ms,
                truncated=True,
                original_length=len(result.output),
                metadata=result.metadata,
            )

        result = result.model_copy(
            update={"duration_ms": int(time.monotonic() * 1000) - start_ms}
        )

        # Step 6: Post-hooks (never blocking)
        for hook in self._post_hooks:
            try:
                await _maybe_await(
                    hook(name=name, arguments=validated, result=result, agent_id=agent_id)
                )
            except Exception:
                pass

        return result

    def _validate_arguments(
        self, tool_name: str, arguments: dict[str, Any], schema: ToolSchema
    ) -> dict[str, Any] | ToolResult:
        """Validate arguments against the tool's JSON schema using Pydantic.

        Returns the (possibly coerced) arguments dict on success, or a
        ToolResult(success=False, error_type="validation_error") on failure.

        Implementation note: Build a dynamic Pydantic model from schema.parameters
        using pydantic.create_model(). Cache the model per tool_name so it is
        only built once.
        """
        from pydantic import ValidationError, create_model

        if not hasattr(self, "_validator_cache"):
            self._validator_cache: dict[str, type[BaseModel]] = {}

        if tool_name not in self._validator_cache:
            self._validator_cache[tool_name] = _build_validator_model(
                tool_name, schema.parameters
            )

        model_cls = self._validator_cache[tool_name]
        try:
            validated_model = model_cls(**arguments)
            return validated_model.model_dump(exclude_none=False)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            return ToolResult(
                output="",
                success=False,
                error=f"Tool '{tool_name}' argument validation failed: {errors}",
                error_type="validation_error",
            )

    # -------------------------------------------------------------------------
    # Hook registration (called by HookSystem in spec 09)
    # -------------------------------------------------------------------------

    def register_pre_hook(self, fn: Callable) -> None:
        """Register a pre-dispatch hook. Called by HookSystem at startup."""
        self._pre_hooks.append(fn)

    def register_post_hook(self, fn: Callable) -> None:
        """Register a post-dispatch hook. Called by HookSystem at startup."""
        self._post_hooks.append(fn)
```

### `ToolConfig` (from `config/models.py`)

Referenced by `ToolRegistry.get_tools_for_agent()`. Defined in the config spec but reproduced here for clarity:

```python
class ToolConfig(BaseModel):
    # Scopes to inherit tools from. Default: global + division.
    inherit: list[Literal["global", "division", "mcp"]] = ["global", "division"]
    # Additional tools to include by name.
    add: list[str] = []
    # Tools to exclude by name. Deny always wins (deny-first from Claude Code).
    deny: list[str] = []
```

### `ToolVetoed` exception

Raised by pre-hooks to block tool execution:

```python
class ToolVetoed(Exception):
    """Raised by a pre_tool hook to veto execution.
    The exception message is returned as the ToolResult error string."""
```

---

## Scope Resolution Algorithm

The resolution algorithm in `get_tools_for_agent()` implements the following precedence. "Later" means higher priority — later additions override earlier ones.

```
Priority (lowest to highest):
  MCP tools        (auto-discovered, plugged in at mcp scope)
  Global tools     (built-in + globally registered plugins)
  Division tools   (shared within a division)
  Agent tools      (agent-specific additions)
  tool_config.add  (explicit force-includes; can add from any scope)

After priority resolution:
  tool_config.deny (deny always wins regardless of priority)
```

YAML example showing how this resolves:

```yaml
# Division: financial has registered "portfolio_query" at division scope.
# Global scope has: glob, grep, read, write, bash_exec.
# MCP has: exa_search, exa_crawl.

name: morning-briefing
division: financial
tools:
  inherit: [global, division]
  add: [exa_search]       # Force-include from MCP scope
  deny: [write, bash_exec]  # Never allow writes or shell from this agent
```

Resolved tool set: `glob, grep, read, portfolio_query, exa_search`

- `write` and `bash_exec` denied even though they are in global scope.
- `exa_crawl` not included (only `exa_search` was force-added from MCP).
- `portfolio_query` included via division inheritance.

---

## Built-in Tools

All built-in tools are registered at `scope="global"` during harness startup by `register_builtin_tools(registry: ToolRegistry)` in `tools/builtin/__init__.py`.

### `GlobTool`

```python
# src/localharness/tools/builtin/glob_tool.py
import glob as _glob
from pathlib import Path

class GlobTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="glob",
            description=(
                "Find files matching a glob pattern. Returns newline-separated "
                "absolute paths. Use ** for recursive matching."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. 'src/**/*.py' or '*.yaml'",
                    },
                    "base_dir": {
                        "type": "string",
                        "description": (
                            "Directory to resolve pattern against. "
                            "Defaults to current working directory."
                        ),
                        "default": ".",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 500,
                        "minimum": 1,
                        "maximum": 5000,
                    },
                },
                "required": ["pattern"],
            },
            destructive=False,
            estimated_tokens=200,
        )

    async def _execute(self, pattern: str, base_dir: str = ".", limit: int = 500) -> ToolResult:
        base = Path(base_dir).resolve()
        if not base.exists():
            return self.err(f"base_dir does not exist: {base}")
        matches = sorted(base.glob(pattern))[:limit]
        if not matches:
            return self.ok("(no matches)")
        return self.ok("\n".join(str(p) for p in matches), match_count=len(matches))
```

### `GrepTool`

```python
# src/localharness/tools/builtin/grep_tool.py
import re
import asyncio
from pathlib import Path

class GrepTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="grep",
            description=(
                "Search file contents for a regex pattern. Returns matching lines "
                "with file path and line number. Searches recursively if path is a directory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter when path is a directory, e.g. '*.py'.",
                        "default": "*",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive matching.",
                        "default": False,
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context before and after each match.",
                        "default": 0,
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return.",
                        "default": 200,
                        "minimum": 1,
                        "maximum": 2000,
                    },
                },
                "required": ["pattern", "path"],
            },
            destructive=False,
            estimated_tokens=500,
        )

    async def _execute(
        self,
        pattern: str,
        path: str,
        glob: str = "*",
        ignore_case: bool = False,
        context_lines: int = 0,
        limit: int = 200,
    ) -> ToolResult:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return self.err(f"Invalid regex: {exc}", error_type="validation_error")

        target = Path(path).resolve()
        if not target.exists():
            return self.err(f"Path does not exist: {target}")

        files = [target] if target.is_file() else sorted(target.rglob(glob))
        lines_out: list[str] = []
        total = 0

        for f in files:
            if not f.is_file():
                continue
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, f.read_text, "utf-8", "replace"
                )
            except OSError:
                continue
            file_lines = text.splitlines()
            for i, line in enumerate(file_lines):
                if compiled.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(file_lines), i + context_lines + 1)
                    for j in range(start, end):
                        lines_out.append(f"{f}:{j+1}: {file_lines[j]}")
                    total += 1
                    if total >= limit:
                        lines_out.append(f"... (limit {limit} reached)")
                        return self.ok("\n".join(lines_out), match_count=total, truncated=True)

        if not lines_out:
            return self.ok("(no matches)")
        return self.ok("\n".join(lines_out), match_count=total)
```

### `ReadTool`

```python
# src/localharness/tools/builtin/read_tool.py

class ReadTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="read",
            description=(
                "Read file contents. Returns the file as a string with line numbers "
                "prepended (format: 'N\\t<line>'). Supports optional line range."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "First line to read (1-indexed). Default: 1.",
                        "default": 1,
                        "minimum": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Default: 2000.",
                        "default": 2000,
                        "minimum": 1,
                        "maximum": 10000,
                    },
                },
                "required": ["path"],
            },
            destructive=False,
            estimated_tokens=800,
        )

    async def _execute(self, path: str, offset: int = 1, limit: int = 2000) -> ToolResult:
        import asyncio
        from pathlib import Path

        target = Path(path).resolve()
        if not target.exists():
            return self.err(f"File not found: {target}", error_type="not_found")
        if target.is_dir():
            return self.err(f"Path is a directory, not a file: {target}")

        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, target.read_text, "utf-8", "replace"
            )
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

        all_lines = text.splitlines()
        total_lines = len(all_lines)
        start = max(0, offset - 1)
        selected = all_lines[start : start + limit]

        numbered = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))
        return self.ok(
            numbered,
            total_lines=total_lines,
            lines_returned=len(selected),
        )
```

### `WriteTool`

```python
# src/localharness/tools/builtin/write_tool.py

class WriteTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="write",
            description=(
                "Write or overwrite a file. Creates parent directories if needed. "
                "Returns the absolute path written and byte count."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode. Default: overwrite.",
                        "default": "overwrite",
                    },
                },
                "required": ["path", "content"],
            },
            destructive=True,
            estimated_tokens=150,
        )

    async def _execute(self, path: str, content: str, mode: str = "overwrite") -> ToolResult:
        import asyncio
        from pathlib import Path

        target = Path(path).resolve()

        # Deny writes to known sensitive paths (defense-in-depth; permission
        # evaluator also enforces this, but tool self-enforces as belt-and-suspenders).
        forbidden_suffixes = {".env", ".secret", ".token", ".pem", ".key"}
        if target.suffix in forbidden_suffixes or target.name.startswith(".env"):
            return self.err(
                f"Write to credential/secret file blocked: {target}",
                error_type="permission_denied",
            )

        target.parent.mkdir(parents=True, exist_ok=True)

        open_mode = "a" if mode == "append" else "w"
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: target.open(open_mode, encoding="utf-8").write(content)
            )
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

        return self.ok(
            f"Written {len(content.encode())} bytes to {target}",
            path=str(target),
            bytes_written=len(content.encode()),
        )
```

### `BashExecTool`

```python
# src/localharness/tools/builtin/bash_tool.py

class BashExecTool(Tool):
    # Bash is the highest-risk tool. Default timeout is shorter than the
    # registry default to prevent runaway processes. Agents should set
    # explicit timeout_s if they need longer-running commands.
    timeout_s: float = 60.0

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="bash_exec",
            description=(
                "Execute a bash command and return combined stdout+stderr. "
                "Working directory is the harness working directory. "
                "Environment inherits from the harness process."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute.",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "Command timeout in seconds. Max 300.",
                        "default": 60.0,
                        "minimum": 1.0,
                        "maximum": 300.0,
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command. Defaults to CWD.",
                        "default": ".",
                    },
                },
                "required": ["command"],
            },
            destructive=True,
            estimated_tokens=300,
        )

    async def _execute(
        self, command: str, timeout_s: float = 60.0, working_dir: str = "."
    ) -> ToolResult:
        import asyncio
        from pathlib import Path

        cwd = Path(working_dir).resolve()
        if not cwd.exists():
            return self.err(f"working_dir does not exist: {cwd}")

        timeout = min(timeout_s, 300.0)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(
                    output=f"Command timed out after {timeout}s: {command}",
                    success=False,
                    error=f"Timeout after {timeout}s",
                    error_type="timeout_error",
                )
        except OSError as exc:
            return self.err(str(exc))

        output = stdout.decode("utf-8", errors="replace")
        return self.ok(
            output or "(no output)",
            exit_code=proc.returncode,
            command=command,
        )
```

### Built-in tool registration

```python
# src/localharness/tools/builtin/__init__.py
from localharness.tools.registry import ToolRegistry

async def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools at global scope. Call once at harness startup."""
    from .glob_tool import GlobTool
    from .grep_tool import GrepTool
    from .read_tool import ReadTool
    from .write_tool import WriteTool
    from .bash_tool import BashExecTool

    for tool in [GlobTool(), GrepTool(), ReadTool(), WriteTool(), BashExecTool()]:
        await registry.register(tool, scope="global")
```

---

## Pydantic Validation on Dispatch

The `_validate_arguments()` method in `ToolRegistry` builds a Pydantic model dynamically from the tool's `parameters` JSON Schema. This is done once per tool name and cached in `_validator_cache`.

### `_build_validator_model` implementation

```python
# src/localharness/tools/registry.py (module-level helper)
from pydantic import create_model
from pydantic.fields import FieldInfo
from typing import Any

_JSON_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

def _build_validator_model(tool_name: str, parameters: dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic BaseModel subclass from a JSON Schema object.

    Handles:
    - Required vs optional fields (uses JSON Schema "required" array)
    - Default values from "default" key
    - Type coercion via _JSON_SCHEMA_TYPE_MAP
    - Nested objects: treated as dict[str, Any] (not deeply validated)

    Does NOT handle:
    - $ref, allOf, anyOf, oneOf — not needed for tool schemas
    - Pattern, format constraints — validation is structural, not semantic
    """
    properties = parameters.get("properties", {})
    required_fields = set(parameters.get("required", []))
    field_definitions: dict[str, Any] = {}

    for field_name, field_schema in properties.items():
        py_type = _JSON_SCHEMA_TYPE_MAP.get(field_schema.get("type", "string"), Any)
        default = field_schema.get("default", ...)
        is_required = field_name in required_fields
        if not is_required and default is ...:
            default = None
            py_type = py_type | None  # type: ignore[operator]

        field_definitions[field_name] = (
            py_type,
            FieldInfo(default=default, description=field_schema.get("description", "")),
        )

    return create_model(f"_{tool_name}_Args", **field_definitions)
```

---

## Tool Count Warning

The threshold `ToolRegistry.TOOL_COUNT_WARNING_THRESHOLD = 15` is enforced in `get_tools_for_agent()`. When exceeded, a `UserWarning` is emitted. The harness CLI also surfaces this during `localharness validate` by calling `get_tools_for_agent()` for each configured agent and checking the returned count.

Rationale: Research on local models (from FEATURES.md) shows context window degradation with >20 tools. The warning fires at 15 to give teams room to reduce before hitting the cliff.

---

## Thread Safety and Async Execution Model

- `ToolRegistry._lock` is an `asyncio.Lock`. All mutations (`register`, `unregister`) acquire this lock.
- `get_tools_for_agent()` does NOT acquire the lock (reads only; Python dict reads are GIL-safe). If registration and dispatch happen concurrently in tests, a brief inconsistency window is acceptable (tool either appears or doesn't; it will not cause data corruption).
- `dispatch()` does NOT acquire the registry lock. Multiple agents may dispatch concurrently.
- Individual tool `run()` methods must be coroutine-safe. The `Tool` base class does not synchronize across concurrent `run()` calls; stateless tools are naturally safe; stateful tools must use their own locks.
- `BashExecTool` and `ReadTool` use `asyncio.get_event_loop().run_in_executor(None, ...)` to run blocking I/O in the default thread pool. This is correct — do not use `asyncio.to_thread()` directly as it bypasses the event loop's executor setting.

---

## Error Handling Reference

| Situation | Behavior |
|-----------|----------|
| Tool name not in registry for agent | `ToolResult(success=False, error_type="not_found")` |
| Tool name in deny list | `ToolResult(success=False, error_type="not_found")` (treat same as not found — don't reveal deny list) |
| Argument validation failure | `ToolResult(success=False, error_type="validation_error")` with field-level error detail |
| Pre-hook raises `ToolVetoed` | `ToolResult(success=False, error_type="permission_denied")` |
| Pre-hook raises other exception | Exception is logged; execution continues |
| `run()` raises exception | Caught by `Tool.run()` wrapper; `ToolResult(success=False, error_type="execution_error")` |
| `run()` exceeds timeout | `ToolResult(success=False, error_type="timeout_error")` |
| Result exceeds size cap | Output truncated; `truncated=True` set; `original_length` set |
| Post-hook raises | Exception logged; result returned unchanged |
| `register()` name collision | `ValueError` raised (programming error — fail fast) |
| `register()` invalid tool object | `TypeError` raised |

---

## Configuration

`ToolRegistry` is constructed once in `src/localharness/tools/__init__.py` and injected into the agent loop constructor. The constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_timeout_s` | `float` | `30.0` | Default `run()` timeout. Individual tools may override via `Tool.timeout_s`. |
| `result_size_cap_chars` | `int` | `50_000` | Maximum characters in `ToolResult.output` before truncation. Approximately 12K tokens at 4 chars/token. |

These values may be overridden via the global harness config (`~/.localharness/config.yaml`) under:

```yaml
tools:
  default_timeout_s: 30
  result_size_cap_chars: 50000
```

---

## Implementation Notes

1. **Cache `info()` results.** The schema returned by `info()` is cached in `_schemas` at registration time. `info()` is never called again unless the tool is re-registered (e.g., MCP reconnect). This means `info()` implementations can do non-trivial work (file reads, etc.) without performance concern.

2. **`_validate_arguments` cache.** The Pydantic model built by `_build_validator_model` is expensive to construct. The `_validator_cache` dict persists these for the life of the registry. No eviction needed — the model count is bounded by registered tool count.

3. **`run_in_executor` thread pool.** The default thread pool size is `min(32, os.cpu_count() + 4)`. For harnesses running >10 concurrent agents each doing file I/O, consider setting `loop.set_default_executor(ThreadPoolExecutor(max_workers=N))` at startup.

4. **MCP tool names must not collide with built-in names.** MCP tools have lower priority than built-ins, but name collisions cause confusing behavior. At MCP registration time, warn if an MCP tool name shadows a global tool.

5. **The deny list is agent-scoped, not global.** `tool_config.deny` applies only to the agent whose config it appears in. One agent denying `bash_exec` does not affect other agents' access to it.

6. **`asyncio.Lock` vs `threading.Lock`.** The registry uses `asyncio.Lock`. This is correct for the v1 single-process event loop model. If the harness is ever extended to use `asyncio.run_in_executor` for agent loops themselves (i.e., agent loops in threads), switch to `threading.RLock` for the registry and ensure all async tool implementations are thread-safe.
