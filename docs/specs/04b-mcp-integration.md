# Spec 04b: MCP Integration

**Component:** `src/localharness/tools/mcp.py`
**Requirements covered:** TOOL-05
**Dependencies:** `tools/registry.py` (spec 04), `config/models.py`, `core/events.py`
**Library:** `mcp` (official Python SDK), Python 3.12+
**Stability:** UNSTABLE (v1)

---

## Purpose

LocalHarness integrates with MCP (Model Context Protocol) servers to dynamically extend its tool set. Users configure MCP servers in their agent YAML; at startup the harness connects, lists available tools, wraps each as a LocalHarness `Tool`, and registers them in `ToolRegistry` at `scope="mcp"`. The result: agents can use MCP-provided tools with no code changes — just a config entry.

MCP became the de facto tool interoperability standard in 2025 (adopted by OpenAI, Google, Microsoft, LangChain). Supporting it gives LocalHarness access to the growing MCP ecosystem without implementing each tool individually.

**Transports supported (v1):**
- `stdio` — spawns a local process and communicates over stdin/stdout. Used for local MCP servers.
- `streamable-http` — HTTP-based transport for remote MCP servers. This is the replacement for the older SSE transport (MCP spec updated 2025-11-25). SSE transport is explicitly NOT supported.

**Out of scope for v1:**
- SSE transport (deprecated in MCP spec 2025-11-25; use streamable-http).
- MCP resource and prompt primitives (only tools are consumed).
- MCP server sampling requests (servers asking the harness for LLM completions).
- Authentication negotiation beyond static auth tokens.

---

## MCP Configuration in Agent YAML

Users declare MCP servers at the agent level or the division level. Division-level declarations are inherited by all agents in that division unless overridden.

```yaml
# Agent YAML: ~/.localharness/agents/research-agent.yaml
name: research-agent
division: research
role: "Research topics using web search and URL crawling"
model: inherit

mcp_servers:
  - name: exa            # Logical name — used in logs and tool name prefix
    transport: stdio
    command: uvx          # Command to spawn the MCP server process
    args: [mcp-server-exa]
    env:
      EXA_API_KEY: "${EXA_API_KEY}"   # Env var interpolation at config load time
    # Optional: only register tools matching these names (allowlist)
    tool_filter:
      allow: [exa_search, exa_crawl]
      # deny: [...]  # Alternatively, block specific tools
    # Optional: prepend server name to tool name to avoid collisions
    prefix_tools: true    # exa_search → exa__exa_search; false = bare name

  - name: github-mcp
    transport: streamable-http
    url: "http://localhost:3000/mcp"
    headers:
      Authorization: "Bearer ${GITHUB_TOKEN}"
    tool_filter:
      deny: [github_delete_repo, github_delete_branch]  # Too destructive
    prefix_tools: false
    # Optional: reconnect policy
    reconnect:
      max_attempts: 3
      backoff_s: 5.0

# Division-level MCP servers (inherited by all agents in this division unless overridden)
# Specified in ~/.localharness/divisions/research.yaml, same schema.
```

### `MCPServerConfig` (Pydantic model in `config/models.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal

class MCPToolFilter(BaseModel):
    # Allowlist: if set, only these tool names are registered.
    allow: list[str] = []
    # Denylist: these tool names are never registered.
    # deny takes precedence over allow.
    deny: list[str] = []

class MCPReconnectConfig(BaseModel):
    max_attempts: int = 3
    backoff_s: float = 5.0

class MCPServerConfig(BaseModel):
    # Logical name for this server. Used in logs, error messages, and tool name prefix.
    name: str

    # Transport type.
    transport: Literal["stdio", "streamable-http"]

    # stdio-specific:
    command: str | None = None          # Executable to run (e.g. "uvx", "python")
    args: list[str] = []               # Arguments to command
    env: dict[str, str] = {}           # Additional environment variables

    # streamable-http-specific:
    url: str | None = None             # Full MCP server URL
    headers: dict[str, str] = {}       # HTTP headers (auth tokens, etc.)

    # Tool filtering: applied before registration in ToolRegistry
    tool_filter: MCPToolFilter = Field(default_factory=MCPToolFilter)

    # If True, tool names are prefixed: "{server_name}__{tool_name}"
    # Prevents name collisions when multiple MCP servers expose tools with the same name.
    # Recommendation: set True when using multiple MCP servers.
    prefix_tools: bool = True

    # Max tools to register from this server.
    # Prevents runaway context cost if a server exposes hundreds of tools.
    max_tools: int = 20

    # Reconnect policy on connection loss.
    reconnect: MCPReconnectConfig = Field(default_factory=MCPReconnectConfig)
```

### Environment variable interpolation

Config loader resolves `"${VAR_NAME}"` strings in `env` and `headers` fields at load time:

```python
import os
import re

def _interpolate_env(value: str) -> str:
    """Replace ${VAR} with os.environ[VAR]. Raises ValueError if var not set."""
    def replacer(match: re.Match) -> str:
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            raise ValueError(
                f"MCP config references env var '${{{var}}}' but it is not set"
            )
        return val
    return re.sub(r"\$\{([^}]+)\}", replacer, value)
```

---

## MCP Client Implementation

```python
# src/localharness/tools/mcp.py
import asyncio
import structlog
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool as MCPToolDef

from localharness.tools.base import Tool, ToolSchema, ToolResult
from localharness.tools.registry import ToolRegistry
from localharness.config.models import MCPServerConfig

log = structlog.get_logger(__name__)


class MCPToolWrapper(Tool):
    """Wraps an MCP tool definition as a LocalHarness Tool.

    Each MCP tool discovered from a server gets one MCPToolWrapper instance.
    The wrapper holds a reference to the MCPServerClient that manages the
    server connection; it delegates run() to the client's execute() method.

    MCPToolWrapper instances are registered in ToolRegistry at scope="mcp".
    """

    def __init__(
        self,
        mcp_tool: MCPToolDef,
        server_client: "MCPServerClient",
        registered_name: str,
    ) -> None:
        self._mcp_tool = mcp_tool
        self._server_client = server_client
        self._registered_name = registered_name  # May include prefix

    def info(self) -> ToolSchema:
        # MCP tools expose name, description, and inputSchema (JSON Schema).
        # Map directly to ToolSchema.
        return ToolSchema(
            name=self._registered_name,
            description=self._mcp_tool.description or "(no description)",
            parameters=self._mcp_tool.inputSchema or {
                "type": "object",
                "properties": {},
                "required": [],
            },
            scope="mcp",
            destructive=True,  # Conservative default — MCP tools may have side effects
            version="mcp-discovered",
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Delegate execution to the MCPServerClient."""
        return await self._server_client.execute_tool(
            tool_name=self._mcp_tool.name,  # Original MCP name, not prefixed
            arguments=kwargs,
        )


class MCPServerClient:
    """Manages the connection to one MCP server.

    Responsibilities:
    - Establish transport connection (stdio or streamable-http)
    - Negotiate the MCP protocol (initialize handshake)
    - List available tools
    - Execute tool calls
    - Handle reconnection on transport failure

    One MCPServerClient per configured MCP server. All tool wrappers for a
    given server share the same client instance.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._session: ClientSession | None = None
        self._tools: dict[str, MCPToolDef] = {}
        self._transport_ctx = None  # Context manager for the transport
        self._connected = False
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish transport + MCP session. Raises on permanent failure."""
        async with self._lock:
            if self._connected:
                return
            await self._connect_inner()

    async def _connect_inner(self) -> None:
        config = self._config
        try:
            if config.transport == "stdio":
                if not config.command:
                    raise ValueError(
                        f"MCP server '{config.name}': transport=stdio requires 'command'"
                    )
                params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env={**dict(__import__("os").environ), **config.env},
                )
                self._transport_ctx = stdio_client(params)

            elif config.transport == "streamable-http":
                if not config.url:
                    raise ValueError(
                        f"MCP server '{config.name}': transport=streamable-http requires 'url'"
                    )
                self._transport_ctx = streamablehttp_client(
                    config.url,
                    headers=config.headers,
                )

            else:
                raise ValueError(f"Unsupported MCP transport: {config.transport!r}")

            read, write, *_ = await self._transport_ctx.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
            self._connected = True
            log.info("mcp_server_connected", server=config.name, transport=config.transport)

        except Exception as exc:
            self._connected = False
            log.error(
                "mcp_server_connect_failed",
                server=config.name,
                error=str(exc),
                exc_info=True,
            )
            raise

    async def disconnect(self) -> None:
        """Gracefully close the MCP session and transport."""
        async with self._lock:
            if not self._connected:
                return
            try:
                if self._session:
                    await self._session.__aexit__(None, None, None)
                if self._transport_ctx:
                    await self._transport_ctx.__aexit__(None, None, None)
            except Exception as exc:
                log.warning(
                    "mcp_server_disconnect_error", server=self._config.name, error=str(exc)
                )
            finally:
                self._session = None
                self._transport_ctx = None
                self._connected = False
                log.info("mcp_server_disconnected", server=self._config.name)

    # -------------------------------------------------------------------------
    # Tool discovery
    # -------------------------------------------------------------------------

    async def list_tools(self) -> list[MCPToolDef]:
        """Call MCP list_tools and cache the result.

        Returns the raw MCP tool definitions. MCPClientManager applies
        filtering and creates MCPToolWrapper instances.

        Raises:
            MCPConnectionError: If the session is not connected.
            MCPProtocolError: If the server returns a malformed response.
        """
        if not self._connected or not self._session:
            raise MCPConnectionError(
                f"MCP server '{self._config.name}' is not connected"
            )
        try:
            response = await self._session.list_tools()
            self._tools = {t.name: t for t in response.tools}
            log.info(
                "mcp_tools_listed",
                server=self._config.name,
                count=len(self._tools),
                names=list(self._tools.keys()),
            )
            return response.tools
        except Exception as exc:
            raise MCPProtocolError(
                f"MCP server '{self._config.name}' list_tools failed: {exc}"
            ) from exc

    # -------------------------------------------------------------------------
    # Tool execution
    # -------------------------------------------------------------------------

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        """Execute an MCP tool by its original (unprefixed) name.

        Handles reconnection on transport failure. Returns ToolResult on
        success or failure — never raises.
        """
        for attempt in range(self._config.reconnect.max_attempts + 1):
            try:
                return await self._execute_once(tool_name, arguments)
            except (MCPConnectionError, MCPTransportError) as exc:
                if attempt >= self._config.reconnect.max_attempts:
                    log.error(
                        "mcp_tool_failed_all_retries",
                        server=self._config.name,
                        tool=tool_name,
                        attempts=attempt + 1,
                    )
                    return ToolResult(
                        output="",
                        success=False,
                        error=f"MCP server '{self._config.name}' unreachable after "
                              f"{attempt + 1} attempts: {exc}",
                        error_type="execution_error",
                    )
                log.warning(
                    "mcp_reconnecting",
                    server=self._config.name,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                await asyncio.sleep(self._config.reconnect.backoff_s)
                try:
                    await self.disconnect()
                    await self._connect_inner()
                except Exception:
                    pass  # Will retry
            except Exception as exc:
                return ToolResult(
                    output="",
                    success=False,
                    error=f"MCP tool '{tool_name}' execution error: {exc}",
                    error_type="execution_error",
                )

        # Should not reach here
        return ToolResult(output="", success=False, error="Unreachable", error_type="execution_error")

    async def _execute_once(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self._connected or not self._session:
            raise MCPConnectionError(f"Server '{self._config.name}' not connected")
        try:
            response = await self._session.call_tool(tool_name, arguments=arguments)
        except Exception as exc:
            # Distinguish transport failures (retry) from protocol errors (don't retry)
            error_str = str(exc).lower()
            if any(kw in error_str for kw in ["connection", "transport", "eof", "broken pipe"]):
                self._connected = False
                raise MCPTransportError(str(exc)) from exc
            raise MCPProtocolError(str(exc)) from exc

        # MCP tool response has a list of content items (text, image, embedded resource)
        # LocalHarness uses the text content only for v1.
        text_parts: list[str] = []
        is_error = False

        for content in response.content:
            if hasattr(content, "text"):
                text_parts.append(content.text)
            elif hasattr(content, "type") and content.type == "text":
                text_parts.append(getattr(content, "text", ""))

        if response.isError:
            is_error = True

        combined = "\n".join(text_parts) if text_parts else "(empty response)"
        return ToolResult(
            output=combined,
            success=not is_error,
            error=combined if is_error else None,
            error_type="execution_error" if is_error else None,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def server_name(self) -> str:
        return self._config.name
```

---

## MCPClientManager

The `MCPClientManager` orchestrates all MCP server connections for a harness instance. It is constructed once and shared across all agent loops.

```python
class MCPClientManager:
    """Manages all MCP server connections and tool registration.

    One MCPClientManager per harness instance. Agents do not hold direct
    references to MCPServerClient — they use the ToolRegistry as normal.

    Lifecycle:
        manager = MCPClientManager(registry)
        await manager.startup(server_configs)  # Connect + register all tools
        # ... harness runs ...
        await manager.shutdown()               # Graceful disconnect all servers
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._clients: dict[str, MCPServerClient] = {}

    async def startup(self, server_configs: list[MCPServerConfig]) -> None:
        """Connect to all configured MCP servers and register their tools.

        Failure to connect to one server does not prevent other servers from
        connecting. Failed servers are logged at ERROR and excluded.

        Called once at harness startup, after ToolRegistry is initialized and
        built-in tools are registered.
        """
        for config in server_configs:
            await self._connect_server(config)

    async def _connect_server(self, config: MCPServerConfig) -> None:
        client = MCPServerClient(config)
        try:
            await client.connect()
        except Exception as exc:
            log.error(
                "mcp_server_startup_failed",
                server=config.name,
                error=str(exc),
            )
            return  # Don't add to _clients; server is unavailable

        # List tools
        try:
            raw_tools = await client.list_tools()
        except MCPProtocolError as exc:
            log.error(
                "mcp_list_tools_failed",
                server=config.name,
                error=str(exc),
            )
            await client.disconnect()
            return

        # Apply filter and cap
        filtered = self._apply_filter(raw_tools, config)
        if len(filtered) > config.max_tools:
            log.warning(
                "mcp_tool_cap_applied",
                server=config.name,
                total=len(filtered),
                cap=config.max_tools,
            )
            filtered = filtered[: config.max_tools]

        # Register each tool in ToolRegistry
        registered_count = 0
        for mcp_tool in filtered:
            registered_name = (
                f"{config.name}__{mcp_tool.name}" if config.prefix_tools else mcp_tool.name
            )
            # Warn if name would shadow a non-MCP tool
            existing = self._registry._find_tool_by_name(registered_name)
            if existing is not None:
                existing_schema = existing.info()
                if existing_schema.scope != "mcp":
                    log.warning(
                        "mcp_tool_name_shadows_higher_priority_tool",
                        mcp_tool=registered_name,
                        existing_scope=existing_schema.scope,
                        server=config.name,
                    )

            wrapper = MCPToolWrapper(
                mcp_tool=mcp_tool,
                server_client=client,
                registered_name=registered_name,
            )
            try:
                await self._registry.register(wrapper, scope="mcp")
                registered_count += 1
            except ValueError as exc:
                # Name collision at MCP scope — log and skip
                log.warning(
                    "mcp_tool_registration_skipped",
                    tool=registered_name,
                    reason=str(exc),
                )

        self._clients[config.name] = client
        log.info(
            "mcp_server_ready",
            server=config.name,
            tools_registered=registered_count,
        )

    def _apply_filter(
        self, tools: list[MCPToolDef], config: MCPServerConfig
    ) -> list[MCPToolDef]:
        """Apply tool_filter.allow and tool_filter.deny to the raw tool list.

        Allow list takes precedence: if allow is non-empty, only tools in
        the allow list pass. Then deny list removes remaining tools.
        Deny always wins over allow.
        """
        allow = set(config.tool_filter.allow)
        deny = set(config.tool_filter.deny)

        result = []
        for tool in tools:
            if allow and tool.name not in allow:
                continue
            if tool.name in deny:
                continue
            result.append(tool)

        return result

    async def shutdown(self) -> None:
        """Gracefully disconnect all MCP servers.

        Called on harness shutdown. Each disconnect is attempted independently;
        failure of one does not prevent others from disconnecting.
        """
        for name, client in self._clients.items():
            try:
                await client.disconnect()
            except Exception as exc:
                log.warning("mcp_shutdown_error", server=name, error=str(exc))
        self._clients.clear()
        log.info("mcp_all_servers_disconnected")

    @property
    def connected_servers(self) -> list[str]:
        """Names of currently-connected MCP servers."""
        return [name for name, c in self._clients.items() if c.is_connected]

    async def reconnect_server(self, server_name: str) -> bool:
        """Manually trigger reconnection for a named server.

        Returns True if reconnection succeeded.
        Used by `localharness doctor` and future health monitoring.
        """
        client = self._clients.get(server_name)
        if client is None:
            log.error("mcp_reconnect_unknown_server", server=server_name)
            return False
        try:
            await client.disconnect()
            await client.connect()
            return client.is_connected
        except Exception as exc:
            log.error(
                "mcp_manual_reconnect_failed", server=server_name, error=str(exc)
            )
            return False
```

---

## Tool Discovery Protocol

The protocol executed for each configured MCP server during `MCPClientManager.startup()`:

```
1. Spawn transport
   stdio: asyncio subprocess via StdioServerParameters
   streamable-http: HTTP client session to URL

2. MCP initialization handshake
   Client → Server: initialize(protocolVersion, capabilities, clientInfo)
   Server → Client: InitializeResult(protocolVersion, capabilities, serverInfo)
   Client → Server: initialized notification

3. Tool listing
   Client → Server: tools/list (paginated; follow nextCursor if present)
   Server → Client: ListToolsResult(tools: [...], nextCursor?)
   Loop until nextCursor is None.

4. Tool wrapping
   For each MCPToolDef:
     - Apply tool_filter (allow/deny)
     - Apply prefix if prefix_tools=True
     - Construct MCPToolWrapper
     - Register in ToolRegistry at scope="mcp"

5. Ready
   Server connection persists for the harness lifetime.
   All subsequent tool calls go through MCPServerClient.execute_tool().
```

### Pagination handling

MCP tool lists may be paginated. The harness follows the cursor:

```python
async def list_tools_paginated(session: ClientSession) -> list[MCPToolDef]:
    all_tools: list[MCPToolDef] = []
    cursor: str | None = None
    while True:
        response = await session.list_tools(cursor=cursor)
        all_tools.extend(response.tools)
        cursor = response.nextCursor
        if cursor is None:
            break
    return all_tools
```

This replaces the simple `await self._session.list_tools()` call in `MCPServerClient.list_tools()` for production use.

---

## MCP Tool Wrapping: ToolSchema Mapping

MCP `Tool` objects (from the SDK's `mcp.types.Tool`) map to LocalHarness `ToolSchema` as follows:

| MCP field | ToolSchema field | Notes |
|-----------|-----------------|-------|
| `tool.name` | `name` | Modified by prefix_tools. Original stored on wrapper. |
| `tool.description` | `description` | Used as-is. Empty string replaced with `"(no description)"`. |
| `tool.inputSchema` | `parameters` | MCP input schema IS a JSON Schema object. Used directly. |
| (not present) | `scope` | Always `"mcp"` for MCP-discovered tools. |
| (not present) | `destructive` | Conservative default: `True`. MCP tools may have side effects. |
| (not present) | `estimated_tokens` | `None` (unknown without profiling). |
| (not present) | `version` | `"mcp-discovered"`. |

### Context cost awareness

MCP tools with `estimated_tokens=None` are not counted against the agent's context budget by the context manager. The harness emits a warning during `localharness validate` if an agent has more than 5 MCP tools with unknown token cost:

```
WARNING: Agent 'research-agent' has 8 MCP tools with unknown token cost.
Consider setting max_tools in each MCPServerConfig to limit context overhead.
```

Users can annotate expected cost via an optional `tool_cost_hints` map in the agent YAML, which the context manager uses:

```yaml
mcp_servers:
  - name: exa
    transport: stdio
    command: uvx
    args: [mcp-server-exa]
    tool_cost_hints:
      exa_search: 400     # Estimated tokens (input+output) per call
      exa_crawl: 2000
```

---

## Agent YAML Full Reference (MCP section)

```yaml
mcp_servers:
  # Minimal stdio server:
  - name: filesystem
    transport: stdio
    command: npx
    args: [-y, "@modelcontextprotocol/server-filesystem", "/tmp/agent-workspace"]
    prefix_tools: true
    max_tools: 10

  # Stdio with env and filter:
  - name: exa
    transport: stdio
    command: uvx
    args: [mcp-server-exa]
    env:
      EXA_API_KEY: "${EXA_API_KEY}"
    tool_filter:
      allow: [exa_search, exa_crawl]
    prefix_tools: true
    max_tools: 5

  # Streamable-HTTP with auth:
  - name: github
    transport: streamable-http
    url: "http://localhost:3000/mcp"
    headers:
      Authorization: "Bearer ${GITHUB_TOKEN}"
    tool_filter:
      deny: [create_repository, delete_repository]
    prefix_tools: true
    max_tools: 15
    reconnect:
      max_attempts: 5
      backoff_s: 10.0
```

---

## Error Handling Reference

| Situation | Behavior |
|-----------|----------|
| MCP server process fails to start (stdio) | `log.error(...)`, server excluded from registry, harness continues |
| MCP HTTP endpoint unreachable (streamable-http) | `log.error(...)`, server excluded, harness continues |
| MCP initialize handshake fails | `log.error(...)`, server excluded, harness continues |
| MCP list_tools returns empty list | `log.info(... tools=0)`, server connected but contributes no tools |
| MCP list_tools protocol error | `log.error(...)`, server disconnected and excluded |
| Tool name collision at MCP scope | `log.warning(...)`, second registration skipped silently |
| MCP tool name shadows non-MCP tool | `log.warning(...)` (the MCP tool has lower priority; safe but confusing) |
| Tool cap exceeded (`max_tools`) | `log.warning(...)`, excess tools dropped (first N retained in list order) |
| MCP tool call transport failure | Retry up to `reconnect.max_attempts`; on exhaustion, `ToolResult(success=False)` |
| MCP tool call protocol error (4xx/5xx) | No retry; `ToolResult(success=False, error_type="execution_error")` |
| MCP server crashes mid-session | Next tool call detects disconnected session; triggers reconnect |
| `env` var not set for stdio server | `ValueError` at config load time; agent fails to load entirely |
| `url` empty for streamable-http server | `ValueError` at connect time; server excluded |
| MCP response content has no text items | `ToolResult(output="(empty response)", success=True)` |
| MCP `isError=True` in response | `ToolResult(success=False)` with content text as error message |

---

## Security Considerations

### Principle of least privilege for MCP tools

MCP servers execute as a separate process (stdio transport) or network endpoint (streamable-http). They are not sandboxed by LocalHarness itself in v1. Users should apply the following mitigations:

1. **Tool filter `deny` list.** Block destructive tools at the agent config level. Example: deny `delete_*` tools from GitHub MCP.

2. **`max_tools` cap.** Prevents a malicious or misconfigured MCP server from flooding the agent context with hundreds of tools.

3. **`prefix_tools: true` by default.** Prefixing tool names makes it obvious which tools come from which server. It also prevents a rogue MCP server from shadowing a built-in tool by registering a tool named `bash_exec`.

4. **Environment variable isolation.** stdio MCP server processes inherit the environment filtered to `{os.environ, **config.env}`. In v2, the environment should be cleaned to only the declared `env` dict (no implicit inheritance from the harness process).

5. **Local-only for sensitive servers (v1).** The `streamable-http` transport should only point at localhost or LAN endpoints in v1. Remote MCP servers (public internet) may return malicious tool descriptions that attempt prompt injection. TLS verification and origin pinning are v2 features.

6. **Deny-first applies.** Agent `tool_config.deny` in the tool scope section applies to MCP tools identically to built-in tools. An agent can deny an MCP tool by its registered name.

### Prompt injection via tool descriptions

MCP server operators control the `description` field of their tools. A malicious server could return a description designed to manipulate the LLM. Mitigations:

- Only connect to MCP servers the user controls or explicitly trusts (document this in `localharness doctor` output).
- In v2, add a tool description sanitizer that strips markdown formatting from MCP tool descriptions before they reach the LLM context.

### Process isolation (v1 gap, v2 target)

stdio MCP server processes in v1 run as the same user as the harness. A compromised MCP server could access any file the user can. v2 will add optional bubblewrap sandboxing for stdio MCP processes using the same mechanism as the planned bash_exec sandbox.

---

## Startup Integration

```python
# src/localharness/tools/__init__.py (additions for MCP)

async def build_tool_system(
    harness_config: "HarnessConfig",
    agent_configs: list["AgentConfig"],
) -> tuple["ToolRegistry", "HookSystem", "PluginLoader", "MCPClientManager"]:
    """Construct and wire the full tool system including MCP. Called once at startup."""
    from localharness.tools.registry import ToolRegistry
    from localharness.tools.hooks import HookSystem
    from localharness.plugins.loader import PluginLoader
    from localharness.tools.builtin import register_builtin_tools
    from localharness.tools.mcp import MCPClientManager

    registry = ToolRegistry(
        default_timeout_s=harness_config.tools.default_timeout_s,
        result_size_cap_chars=harness_config.tools.result_size_cap_chars,
    )
    hook_system = HookSystem()
    loader = PluginLoader(registry, hook_system)

    # 1. Built-ins
    await register_builtin_tools(registry)

    # 2. Plugins
    await loader.discover_all()

    # 3. MCP — collect all unique server configs across agents + divisions
    all_server_configs = _collect_mcp_configs(agent_configs)
    mcp_manager = MCPClientManager(registry)
    if all_server_configs:
        await mcp_manager.startup(all_server_configs)

    # 4. Wire hooks
    hook_system.wire_to_registry(registry)

    return registry, hook_system, loader, mcp_manager


def _collect_mcp_configs(
    agent_configs: list["AgentConfig"],
) -> list[MCPServerConfig]:
    """Deduplicate MCP server configs across agents. Two configs are the same
    if they share the same server name. Agent-level configs take precedence
    over division-level configs with the same name."""
    seen: dict[str, MCPServerConfig] = {}
    for agent in agent_configs:
        for server_cfg in agent.mcp_servers:
            if server_cfg.name not in seen:
                seen[server_cfg.name] = server_cfg
    return list(seen.values())
```

### Graceful shutdown

```python
# In the harness main shutdown sequence:
async def shutdown(
    mcp_manager: MCPClientManager,
    hook_system: HookSystem,
) -> None:
    """Shutdown order: MCP (closes external processes first), then hooks."""
    await mcp_manager.shutdown()
    # Hook system has no async cleanup in v1 (pluggy is sync)
```

---

## Exception Hierarchy

```python
# src/localharness/tools/mcp.py

class MCPError(Exception):
    """Base class for all MCP integration errors."""

class MCPConnectionError(MCPError):
    """Server is not connected or connection was lost. May be retried."""

class MCPTransportError(MCPError):
    """Transport-level failure (I/O, EOF, broken pipe). Always retried."""

class MCPProtocolError(MCPError):
    """MCP protocol violation or server error response. Not retried."""

class MCPToolNotFound(MCPError):
    """The tool name was not found on the server (server-side)."""
```

---

## Implementation Notes

1. **MCP SDK import.** The official Python MCP SDK package is `mcp`. Import as `from mcp import ClientSession`. The SDK is maintained by Anthropic and tracks the MCP spec. Pin to `mcp>=1.0.0` in `pyproject.toml`. Do not use `mcp-python` (unmaintained fork).

2. **Single session per server.** Each `MCPServerClient` holds one `ClientSession`. Multiple concurrent tool calls from different agents go through the same session. The MCP SDK's `call_tool` is async and the server handles concurrency on its side. If a server cannot handle concurrent requests, wrap `_execute_once` with an asyncio semaphore.

3. **stdio transport process lifetime.** The child process spawned by `stdio_client` lives for the duration of the `MCPServerClient` connection. It is killed when `disconnect()` is called. If the harness crashes without calling `shutdown()`, child processes may become orphans. Register `mcp_manager.shutdown()` in both `atexit` and `signal.SIGTERM` handlers.

4. **Streamable-HTTP session reuse.** The `streamablehttp_client` context manager manages an HTTP session. Keep it open for the harness lifetime — do not re-enter it per tool call. The current implementation does this correctly via `self._transport_ctx.__aenter__()` at connect time.

5. **MCP tool argument coercion.** MCP servers define `inputSchema` as a JSON Schema object. LocalHarness validates arguments against this schema via `ToolRegistry._validate_arguments()` before calling `execute_tool()`. The schema is accessed via `MCPToolWrapper.info().parameters`. This ensures MCP tools get the same Pydantic validation pass as built-in tools.

6. **Tool name prefix separator.** The prefix format is `{server_name}__{tool_name}` (double underscore). This avoids collision with typical snake_case tool names and matches common MCP ecosystem conventions. Example: server `exa`, tool `search` → `exa__search`.

7. **Reconnect does not re-register tools.** When a server reconnects after a crash, it does NOT re-discover and re-register tools. The existing `MCPToolWrapper` instances remain in the registry and will resume working once `_connected=True` again. Re-registering would require unregistering first, which would race with active dispatch calls.

8. **Health check.** `localharness doctor` calls `MCPClientManager.connected_servers` and reports which servers are connected. For each connected server, it sends a `tools/list` ping to verify the session is alive.
