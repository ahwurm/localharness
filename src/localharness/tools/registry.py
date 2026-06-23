"""ToolRegistry: scope resolution, Pydantic dispatch, and hook integration."""
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo

from localharness.tools.base import Tool, ToolProtocol, ToolResult, ToolSchema, ToolVetoed

_JSON_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_validator_model(tool_name: str, parameters: dict[str, Any]) -> type[BaseModel]:
    """Build a dynamic Pydantic model from a JSON Schema object."""
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


async def _maybe_await(result: Any) -> Any:
    import asyncio
    if asyncio.iscoroutine(result):
        return await result
    return result


class ToolRegistry:
    """Thread-safe tool registry with scope resolution."""

    TOOL_COUNT_WARNING_THRESHOLD: int = 15

    def __init__(
        self,
        default_timeout_s: float = 30.0,
        result_size_cap_chars: int = 50_000,
    ) -> None:
        self._tools: dict[str, dict[str, ToolProtocol]] = {
            "global": {},
            "division": {},
            "agent": {},
            "mcp": {},
        }
        self._schemas: dict[str, ToolSchema] = {}
        self._division_tools: dict[str, dict[str, ToolProtocol]] = {}
        self._agent_tools: dict[str, dict[str, ToolProtocol]] = {}
        self._default_timeout_s = default_timeout_s
        self._result_size_cap_chars = result_size_cap_chars
        self._lock = __import__("asyncio").Lock()
        self._pre_hooks: list[Callable] = []
        self._post_hooks: list[Callable] = []
        self._validator_cache: dict[str, type[BaseModel]] = {}

    async def register(
        self,
        tool: ToolProtocol,
        scope: str = "global",
        division_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
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
                    raise ValueError(f"Tool '{name}' already registered for division '{division_id}'")
                bucket[name] = tool
                self._schemas[f"division:{division_id}:{name}"] = schema

            elif scope == "agent":
                if agent_id is None:
                    raise ValueError("agent_id required for agent-scoped tools")
                bucket = self._agent_tools.setdefault(agent_id, {})
                if name in bucket:
                    raise ValueError(f"Tool '{name}' already registered for agent '{agent_id}'")
                bucket[name] = tool
                self._schemas[f"agent:{agent_id}:{name}"] = schema

            elif scope == "mcp":
                self._tools["mcp"][name] = tool
                self._schemas[f"mcp:{name}"] = schema

            else:
                raise ValueError(f"Unknown scope: '{scope}'")

    async def unregister(self, name: str, scope: str = "global", **scope_kwargs: str) -> None:
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

    def rebind_global(self, tool: ToolProtocol) -> None:
        """Overwrite a global-scope tool IN PLACE (per-agent store binding).

        Unlike register(), this REPLACES an existing global entry instead of raising — used to bind
        store-backed verb tools (web_fetch / web_page_query / tool_result_get) to an agent's OWN
        ContentStore. No-ops if the tool isn't already present, so it never grants a capability the
        agent's toolset withheld. Synchronous direct-write (mirrors from_allowed)."""
        name = tool.info().name
        if name in self._tools["global"]:
            self._tools["global"][name] = tool
            self._schemas[name] = tool.info()

    def get_tools_for_agent(
        self,
        agent_id: str,
        division_id: str,
        tool_config: Any,  # ToolConfig from config/models.py
    ) -> dict[str, ToolSchema]:
        resolved: dict[str, ToolProtocol] = {}
        inherit = tool_config.inherit if tool_config.inherit is not None else ["global", "division"]

        if "global" in inherit:
            resolved.update(self._tools["global"])

        # MCP always visible unless denied
        for name, tool in self._tools["mcp"].items():
            if name not in resolved:
                resolved[name] = tool

        if "division" in inherit:
            resolved.update(self._division_tools.get(division_id, {}))

        # Agent-specific tools always applied
        resolved.update(self._agent_tools.get(agent_id, {}))

        # Force-add
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

        # Deny list wins
        for name in (tool_config.deny or []):
            resolved.pop(name, None)

        if len(resolved) > self.TOOL_COUNT_WARNING_THRESHOLD:
            import warnings
            warnings.warn(
                f"Agent '{agent_id}' has {len(resolved)} tools "
                f"(threshold: {self.TOOL_COUNT_WARNING_THRESHOLD}). "
                "Context window degradation likely on models with <32K context.",
                stacklevel=2,
            )

        from localharness.tools.capabilities import assert_no_coresidence, floor_enabled
        if floor_enabled():
            # Mark mcp-bucket tools so an MCP ingestion tool (e.g. fetch) co-resident with a
            # host-dangerous tool is caught here too. (Plugins resolved via 'global' scope register
            # bare and are a named residual — see capabilities.assert_no_coresidence.)
            check_names = {("mcp:" + n if n in self._tools["mcp"] else n) for n in resolved}
            assert_no_coresidence(check_names, agent_id=agent_id)

        return {name: tool.info() for name, tool in resolved.items()}

    def _find_tool_by_name(self, name: str) -> ToolProtocol | None:
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
        tool_config: Any,
    ) -> ToolProtocol | None:
        if name in (tool_config.deny or []):
            return None
        if name in self._agent_tools.get(agent_id, {}):
            return self._agent_tools[agent_id][name]
        if name in self._division_tools.get(division_id, {}):
            return self._division_tools[division_id][name]
        if name in self._tools["global"]:
            return self._tools["global"][name]
        if name in self._tools["mcp"]:
            return self._tools["mcp"][name]
        return None

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        agent_id: str,
        division_id: str,
        tool_config: Any,
    ) -> ToolResult:
        start_ms = int(time.monotonic() * 1000)

        tool = self._get_tool_for_agent(name, agent_id, division_id, tool_config)
        if tool is None:
            return ToolResult(
                output="",
                success=False,
                error=f"Tool '{name}' not found or not permitted for agent '{agent_id}'",
                error_type="not_found",
            )

        validated = self._validate_arguments(name, arguments, tool.info())
        if isinstance(validated, ToolResult):
            return validated

        for hook in self._pre_hooks:
            try:
                await _maybe_await(hook(name=name, arguments=validated, agent_id=agent_id, division_id=division_id))
            except ToolVetoed as exc:
                return ToolResult(
                    output="",
                    success=False,
                    error=str(exc),
                    error_type="permission_denied",
                    duration_ms=int(time.monotonic() * 1000) - start_ms,
                )
            except Exception:
                pass

        result = await tool.run(**validated)

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

        result = result.model_copy(update={"duration_ms": int(time.monotonic() * 1000) - start_ms})

        for hook in self._post_hooks:
            try:
                await _maybe_await(
                    hook(name=name, arguments=validated, result=result, agent_id=agent_id, division_id=division_id)
                )
            except Exception:
                pass

        return result

    def _validate_arguments(
        self, tool_name: str, arguments: dict[str, Any], schema: ToolSchema
    ) -> dict[str, Any] | ToolResult:
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

    def register_pre_hook(self, fn: Callable) -> None:
        self._pre_hooks.append(fn)

    def register_post_hook(self, fn: Callable) -> None:
        self._post_hooks.append(fn)

    # ------------------------------------------------------------------
    # Bench-runner helpers (Plan 12-04 Task 1)
    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        """Return True if `name` resolves to a registered tool in any scope.

        Accepts bare names (`exa_search`), MCP-prefixed names (`mcp:fetch`),
        and plugin-prefixed names (`plugin:PLUGIN.TOOL`). The prefix forms
        strip down to the bare TOOL name for resolution because plugin tools
        register at scope="global" under their bare name (see plugins/loader.py).
        """
        from localharness.bench.schema import parse_tool_name
        try:
            _source, tool_name, _plugin = parse_tool_name(name)
        except ValueError:
            tool_name = name
        # Also keep the raw form so MCP/plugin lookups can find prefixed
        # registrations if any backend chose to store them prefixed.
        return (
            tool_name in self._tools["global"]
            or tool_name in self._tools["mcp"]
            or name in self._tools["global"]
            or name in self._tools["mcp"]
        )

    @classmethod
    def from_allowed(
        cls,
        allowed: list[str],
        base_registry: "ToolRegistry | None" = None,
    ) -> "ToolRegistry":
        """Build a registry containing only the tools named in `allowed`.

        `allowed` entries use the source-prefix convention from
        bench.schema.parse_tool_name (`bare`, `mcp:TOOL`, `plugin:PLUGIN.TOOL`).
        For each entry the bare TOOL name is resolved against `base_registry`'s
        scope='global' (where plugins/loader.py and register_builtin_tools both
        register) and re-registered under both bare and prefixed forms so
        downstream dispatch resolves whichever form the agent loop uses.

        `base_registry` must not be None.  Passing None previously returned an
        empty registry — a silent foot-gun where a caller that forgot
        ``_get_base_registry()`` would hand the agent loop a zero-tool registry
        and every tool dispatch would fail silently.  Pass the builtin registry
        (``await _get_base_registry()``) or an explicit empty one built with
        ``ToolRegistry()`` when you deliberately want no base tools.
        """
        from localharness.bench.schema import parse_tool_name

        if base_registry is None:
            raise ValueError(
                "from_allowed() requires an explicit base_registry; "
                "pass the builtin registry (await _get_base_registry()) or "
                "ToolRegistry() when no base tools are desired. "
                "Passing None previously silently returned a zero-tool registry."
            )
        out = cls()

        for entry in allowed:
            try:
                _source, tool_name, _plugin = parse_tool_name(entry)
            except ValueError:
                tool_name = entry

            tool = (
                base_registry._tools["global"].get(tool_name)
                or base_registry._tools["mcp"].get(tool_name)
                or base_registry._tools["global"].get(entry)
                or base_registry._tools["mcp"].get(entry)
            )
            if tool is None:
                continue

            # Register under bare name in global scope (sync — bypass async lock
            # because from_allowed is invoked during bench-loop construction)
            if tool_name not in out._tools["global"]:
                out._tools["global"][tool_name] = tool
                out._schemas[tool_name] = tool.info()
            # Also register under the prefixed form if the entry was prefixed
            if entry != tool_name and entry not in out._tools["global"]:
                out._tools["global"][entry] = tool
                out._schemas[entry] = tool.info()

        from localharness.tools.capabilities import assert_no_coresidence, floor_enabled
        if floor_enabled():
            # Detect ingest by SOURCE: keep the mcp:/plugin: prefix so an MCP/plugin ingestion tool
            # (e.g. mcp:fetch, plugin:research_tools.exa_search) is flagged untrusted-ingest — not
            # just the 3 built-in web verbs. Intent-based (checks the declared `allowed`) so a
            # co-resident config is rejected regardless of whether each tool happens to be installed.
            check_names: set[str] = set()
            for entry in allowed:
                try:
                    source, tool_name, _p = parse_tool_name(entry)
                except ValueError:
                    source, tool_name = "builtin", entry
                check_names.add(entry if source in ("mcp", "plugin") else tool_name)
            assert_no_coresidence(check_names)

        return out
