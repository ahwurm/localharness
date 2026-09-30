"""ToolRegistry: scope resolution, Pydantic dispatch, and hook integration."""
import difflib
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo

from localharness.tools.base import Tool, ToolProtocol, ToolResult, ToolSchema, ToolVetoed
from localharness.tools.capabilities import (
    IngestViaExecError,
    assert_no_coresidence,
    assert_no_ingest_via_exec,
    floor_enabled,
    ingests_untrusted,
    is_exec,
)

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


def _bare_name(entry: str) -> str:
    """An allowed-list entry's TOOL name: `mcp:TOOL` and `plugin:PLUGIN.TOOL` strip to TOOL."""
    from localharness.bench.schema import parse_tool_name
    try:
        return parse_tool_name(entry)[1]
    except ValueError:
        return entry


async def _maybe_await(result: Any) -> Any:
    import asyncio
    if asyncio.iscoroutine(result):
        return await result
    return result


log = logging.getLogger(__name__)


class ContributedTool:
    """A plugin-contributed tool as the registry holds it (PAPI-05, PAPI-11).

    info() is the plugin tool's own schema with `source_plugin` stamped, and any loader override
    (such as SAFE-06's gate-family clamp for a plugin you installed) applied, on EVERY call, so
    every reader sees one schema: the gate (agent/loop.py reads tool.info() per call), the
    capability floor, the context store, /api/tools. The safety readers never read source_plugin
    (CORE-04). run() turns an exception the plugin's tool raises, SystemExit included, into an
    attributed error result: a plugin never takes a turn, or the harness, down. Cancellation still
    propagates. Other attribute reads fall through to the plugin's own tool."""

    def __init__(self, inner: ToolProtocol, source_plugin: str,
                 overrides: Mapping[str, Any] | None = None) -> None:
        self._inner = inner
        self._source_plugin = source_plugin
        self._overrides = dict(overrides or {})

    def info(self) -> ToolSchema:
        return self._inner.info().model_copy(
            update={**self._overrides, "source_plugin": self._source_plugin})

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._inner.run(**kwargs)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 — PAPI-11: attributed, never fatal
            return self._attributed(ToolResult(output="", success=False, error_type="execution_error",
                                               error=str(exc)), type(exc).__name__, exc)
        # A Tool subclass's run() already caught what its _execute raised and marked it (tools/base.py);
        # an error the tool chose to return carries no mark and is its own words.
        raised = None if result.success else result.metadata.get("raised")
        return self._attributed(result, raised) if isinstance(raised, str) else result

    def _attributed(self, result: ToolResult, raised: str, exc: BaseException | None = None) -> ToolResult:
        name = self.info().name
        log.warning("tool %r from plugin %r raised %s; returned as a tool error",
                    name, self._source_plugin, raised, exc_info=exc)
        return result.model_copy(update={
            "error": f"tool {name!r} from plugin {self._source_plugin!r} failed: {raised}: {result.error}"})

    def __getattr__(self, attr: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(attr)
        return getattr(inner, attr)


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
        *,
        source_plugin: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """A plugin's tool registers bare at global scope, exactly as a builtin does (PAPI-05), with
        `source_plugin` naming the plugin. It is then held as a ContributedTool, which stamps that
        provenance and the loader's `overrides` onto every info() read and contains its errors."""
        if not isinstance(tool, ToolProtocol):
            raise TypeError(
                f"{type(tool).__name__} does not satisfy ToolProtocol "
                "(must implement info() and run())"
            )
        if overrides and source_plugin is None:
            raise ValueError("schema overrides apply only to a plugin-contributed tool "
                             "(pass source_plugin)")
        if source_plugin is not None:
            tool = ContributedTool(tool, source_plugin, overrides)

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

        schemas = {name: tool.info() for name, tool in resolved.items()}
        if floor_enabled():
            # Judged by what each resolved tool DECLARES, whatever scope or name it arrived under:
            # an MCP tool declares ingest in the wrapper's code, a plugin tool inherited through
            # 'global' under a bare name declares its own (or fails closed) — no residual left.
            assert_no_coresidence(schemas.values(), agent_id=agent_id)
        return schemas

    def schema_of(self, name: str) -> ToolSchema | None:
        """The live schema of a registered tool, any scope, or None. An allowed-list form
        (`mcp:TOOL`, `plugin:PLUGIN.TOOL`) resolves to its bare TOOL, as from_allowed resolves it."""
        tool = self._find_tool_by_name(name) or self._find_tool_by_name(_bare_name(name))
        return tool.info() if tool is not None else None

    def global_schemas(self) -> list[ToolSchema]:
        """Every global-scope tool's live schema — what the root capability floor reads."""
        return [tool.info() for tool in self._tools["global"].values()]

    def result_origin(self, name: str) -> Literal["untrusted", "trusted"]:
        """What the context store marks an evicted body of `name` (SAFE-03): the tool's declared
        result_origin; a name no registered tool answers to fails closed (untrusted)."""
        schema = self.schema_of(name)
        return schema.result_origin if schema is not None else "untrusted"

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

    def _agent_has_ingest(self, agent_id: str, division_id: str, tool_config: Any) -> bool:
        """Does this agent reach a tool that declares ingest: untrusted — i.e. is it DESIGNATED an
        ingester?

        Resolved through the same per-agent path dispatch uses (so deny wins), over every tool in
        the global, mcp, the agent's division and the agent's own buckets, and judged by the
        DECLARATION: the web verbs, an MCP tool, a plugin's search tool and a tool that declares
        nothing all count, under whatever name they are registered.
        """
        names = {*self._tools["global"], *self._tools["mcp"],
                 *self._division_tools.get(division_id, {}), *self._agent_tools.get(agent_id, {})}
        return any(
            (tool := self._get_tool_for_agent(n, agent_id, division_id, tool_config)) is not None
            and ingests_untrusted(tool.info())
            for n in names
        )

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

    # difflib's own default ratio; below it "did you mean" turns into noise.
    SUGGEST_CUTOFF: float = 0.6

    def _unknown_tool(
        self, name: str, agent_id: str, division_id: str, tool_config: Any
    ) -> ToolResult:
        """The two cases dispatch used to fold into one string, told apart: a tool that exists
        but is not this agent's is `permission_denied` (retrying will not help; a config change
        will); a name nothing answers to is `not_found` with the nearest callable names — by
        spelling (difflib) or by GROUP, since a model that wants to delegate asks for `delegate`,
        which is the group of the tool named `agent`. Measured live (2026-09-14): three
        identical retries of `delegate`, and a human reading the old message concluded the tool
        "needs to be trusted"."""
        if self._find_tool_by_name(name) is not None:
            return ToolResult(
                output="", success=False, error_type="permission_denied",
                error=f"Tool '{name}' exists but is not permitted for agent '{agent_id}'",
            )
        visible = self.get_tools_for_agent(agent_id, division_id, tool_config)
        wanted = name.lower()
        by_group = [n for n, s in visible.items()
                    if wanted in (s.group.lower(), s.group.lower().rsplit(".", 1)[-1])]
        close = difflib.get_close_matches(name, list(visible), n=3, cutoff=self.SUGGEST_CUTOFF)
        near = list(dict.fromkeys([*by_group, *close]))
        tail = (f" Did you mean: {', '.join(near)}?" if near
                else f" Callable tools: {', '.join(sorted(visible)) or 'none'}.")
        return ToolResult(
            output="", success=False, error_type="not_found",
            error=f"Unknown tool '{name}'.{tail}",
        )

    def lookup_tool(
        self,
        name: str,
        agent_id: str,
        division_id: str,
        tool_config: Any,
    ) -> ToolProtocol | None:
        """The tool `dispatch` WOULD run, without running it.

        The permission gate needs a tool's schema (its `destructive` flag and `group`) and its
        declared `timeout_s` BEFORE the call happens — same resolution order as `dispatch`, so
        the gate can never judge a different tool than the one that executes.
        """
        return self._get_tool_for_agent(name, agent_id, division_id, tool_config)

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
            return self._unknown_tool(name, agent_id, division_id, tool_config)

        schema = tool.info()
        validated = self._validate_arguments(name, arguments, schema)
        if isinstance(validated, ToolResult):
            return validated

        # Ingest gate (owner ruling 2026-09-17): an agent DENIED the web verbs may not fetch
        # remote content through an exec tool instead. Keyed off the agent's own DESIGNATION at
        # the one dispatch chokepoint, so EVERY agent inherits it with no per-agent config —
        # including a specialist the model writes itself at runtime. Exec-ness is DECLARED.
        if is_exec(schema):
            try:
                assert_no_ingest_via_exec(
                    schema,
                    validated,
                    agent_id=agent_id,
                    has_ingest=self._agent_has_ingest(agent_id, division_id, tool_config),
                )
            except IngestViaExecError as exc:
                log.warning("ingest-via-exec BLOCKED: agent=%s tool=%s", agent_id, name)
                return ToolResult(
                    output="",
                    success=False,
                    error=str(exc),
                    error_type="permission_denied",
                    duration_ms=int(time.monotonic() * 1000) - start_ms,
                )

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

        if result.error and len(result.error) > self._result_size_cap_chars:
            # .error was never capped here — the old event-slice bounded it by accident (#133).
            # MCP tools mirror whole server responses into .error, so cap at the choke point.
            result = result.model_copy(update={
                "error": result.error[: self._result_size_cap_chars]
                + f"\n… [error truncated: {len(result.error):,} chars total]",
            })

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
        register at scope="global" under their bare name (see plugins/lifecycle.py).
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
        scope='global' (where plugins/lifecycle.py and register_builtin_tools both
        register) and re-registered under both bare and prefixed forms so
        downstream dispatch resolves whichever form the agent loop uses.

        `base_registry` must not be None.  Passing None previously returned an
        empty registry — a silent foot-gun where a caller that forgot
        ``_get_base_registry()`` would hand the agent loop a zero-tool registry
        and every tool dispatch would fail silently.  Pass the builtin registry
        (``await _get_base_registry()``) or an explicit empty one built with
        ``ToolRegistry()`` when you deliberately want no base tools.
        """
        if base_registry is None:
            raise ValueError(
                "from_allowed() requires an explicit base_registry; "
                "pass the builtin registry (await _get_base_registry()) or "
                "ToolRegistry() when no base tools are desired. "
                "Passing None previously silently returned a zero-tool registry."
            )
        out = cls()

        unresolved_external: list[str] = []
        for entry in allowed:
            tool_name = _bare_name(entry)

            tool = (
                base_registry._tools["global"].get(tool_name)
                or base_registry._tools["mcp"].get(tool_name)
                or base_registry._tools["global"].get(entry)
                or base_registry._tools["mcp"].get(entry)
            )
            if tool is None:
                if entry.startswith(("mcp:", "plugin:")):
                    unresolved_external.append(entry)
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

        if floor_enabled():
            # The RESOLVED tools are judged by what they declare. An mcp:/plugin: entry the base
            # registry could not resolve is judged by its declared INTENT — the MCP wrapper's
            # posture, ingest untrusted — so a config naming an external ingestion tool beside a
            # host-dangerous one is rejected whether or not that tool happens to be installed (the
            # check the floor has always made). Its host class is unknowable until it is installed.
            intents = [ToolSchema(name=entry, description="named in allowed, not installed",
                                  parameters={}, ingest="untrusted", host="safe",
                                  result_origin="untrusted") for entry in unresolved_external]
            assert_no_coresidence([*(t.info() for t in out._tools["global"].values()), *intents])

        return out
