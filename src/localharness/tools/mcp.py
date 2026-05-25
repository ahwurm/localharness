"""MCP client integration: MCPToolWrapper, MCPServerClient, MCPClientManager."""
import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool as MCPToolDef

from localharness.tools.base import Tool, ToolResult, ToolSchema

if TYPE_CHECKING:
    from localharness.config.models import MCPServerConfig
    from localharness.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MCPToolWrapper
# ---------------------------------------------------------------------------

class MCPToolWrapper(Tool):
    """Wraps an MCP tool definition as a LocalHarness Tool.

    Registered in ToolRegistry at scope="mcp". Delegates execution to
    MCPServerClient.execute_tool().
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        session: Any,
        server_name: str,
        prefix: bool = True,
    ) -> None:
        self._raw_name = name
        self._name = f"{server_name}__{name}" if prefix else name
        self._description = description
        self._input_schema = input_schema
        self._session = session
        self._server_name = server_name

    def info(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"[MCP:{self._server_name}] {self._description}",
            parameters=self._input_schema,
            scope="mcp",
            destructive=True,
            version="mcp-discovered",
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        start = time.monotonic()
        try:
            result = await self._session.call_tool(self._raw_name, arguments=kwargs)
            output_parts: list[str] = []
            for content in result.content:
                if hasattr(content, "text"):
                    output_parts.append(content.text)
                elif hasattr(content, "data"):
                    output_parts.append(f"[binary data: {len(content.data)} bytes]")
            output = "\n".join(output_parts) or "(empty response)"
            is_error = getattr(result, "isError", False)
            return ToolResult(
                output=output,
                success=not is_error,
                error=output if is_error else None,
                error_type="execution_error" if is_error else None,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                output="",
                success=False,
                error=f"MCP tool '{self._raw_name}' on server '{self._server_name}' failed: {exc}",
                error_type="execution_error",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

    async def run(self, **kwargs: Any) -> ToolResult:
        """Override Tool.run() — MCPToolWrapper manages its own timeout via _execute."""
        return await self._execute(**kwargs)


# ---------------------------------------------------------------------------
# MCPServerClient
# ---------------------------------------------------------------------------

class MCPServerClient:
    """Manages a single MCP server connection and tool discovery."""

    def __init__(self, config: "MCPServerConfig") -> None:
        self._config = config
        self._session: ClientSession | None = None
        self._ctx: Any = None
        self._session_ctx: Any = None
        self._tools: list[MCPToolWrapper] = []
        self._connected: bool = False
        self._lock = asyncio.Lock()
        self._log = logging.getLogger("localharness.tools.mcp")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> list[MCPToolWrapper]:
        return self._tools

    @property
    def server_name(self) -> str:
        return self._config.name

    async def connect(self) -> None:
        """Connect to the MCP server and discover tools."""
        async with self._lock:
            if self._connected:
                return
            if self._config.transport == "stdio":
                if not self._config.command:
                    raise ValueError(
                        f"MCP server '{self._config.name}': transport=stdio requires 'command'"
                    )
                params = StdioServerParameters(
                    command=self._config.command,
                    args=self._config.args,
                    env={**dict(os.environ), **self._config.env},
                )
                self._ctx = stdio_client(params)
            elif self._config.transport == "streamable_http":
                if not self._config.url:
                    raise ValueError(
                        f"MCP server '{self._config.name}': transport=streamable_http requires 'url'"
                    )
                self._ctx = streamablehttp_client(
                    self._config.url,
                    headers=self._config.headers,
                )
            else:
                raise ValueError(f"Unsupported MCP transport: {self._config.transport!r}")

            read, write, *_ = await self._ctx.__aenter__()
            self._session_ctx = ClientSession(read, write)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()
            self._connected = True

            # Discover tools
            tools_result = await self._session.list_tools()
            for tool in tools_result.tools:
                schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}
                wrapper = MCPToolWrapper(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=schema,
                    session=self._session,
                    server_name=self._config.name,
                    prefix=True,
                )
                self._tools.append(wrapper)

            self._log.info(
                "mcp_server_connected",
                extra={"server": self._config.name, "tools": len(self._tools)},
            )

    async def disconnect(self) -> None:
        """Gracefully disconnect from the MCP server."""
        self._connected = False
        if self._session_ctx:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if self._ctx:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None
        self._tools = []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute an MCP tool by its original (unprefixed) name."""
        if not self._connected or not self._session:
            return ToolResult(
                output="",
                success=False,
                error=f"MCP server '{self._config.name}' is not connected",
                error_type="execution_error",
            )
        start = time.monotonic()
        try:
            response = await self._session.call_tool(tool_name, arguments=arguments)
            text_parts: list[str] = []
            for content in response.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
            combined = "\n".join(text_parts) or "(empty response)"
            is_error = getattr(response, "isError", False)
            return ToolResult(
                output=combined,
                success=not is_error,
                error=combined if is_error else None,
                error_type="execution_error" if is_error else None,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                output="",
                success=False,
                error=f"MCP tool '{tool_name}' on server '{self._config.name}' failed: {exc}",
                error_type="execution_error",
                duration_ms=int((time.monotonic() - start) * 1000),
            )


# ---------------------------------------------------------------------------
# MCPClientManager
# ---------------------------------------------------------------------------

class MCPClientManager:
    """Manages lifecycle of all configured MCP server connections.

    Lifecycle:
        manager = MCPClientManager(registry)
        await manager.startup(server_configs)   # Connect + register all tools
        # ... harness runs ...
        await manager.shutdown()                 # Graceful disconnect all servers
    """

    def __init__(self, registry: "ToolRegistry") -> None:
        self._registry = registry
        self._clients: dict[str, MCPServerClient] = {}
        self._log = logging.getLogger("localharness.tools.mcp")

    async def startup(self, configs: list["MCPServerConfig"]) -> dict[str, int]:
        """Connect to all configured MCP servers and register their tools.

        Returns dict of {server_name: tool_count} for all configured servers.
        Non-fatal: logs errors and continues with remaining servers.
        """
        results: dict[str, int] = {}
        for config in configs:
            client = MCPServerClient(config)
            try:
                await client.connect()
                self._clients[config.name] = client
                for tool in client.tools:
                    await self._registry.register(tool, scope="mcp")
                results[config.name] = len(client.tools)
                self._log.info(
                    f"MCP server '{config.name}' connected — {len(client.tools)} tool(s)"
                )
            except Exception as exc:
                self._log.warning(
                    f"MCP server '{config.name}' failed to connect: {exc}"
                )
                results[config.name] = 0
        return results

    async def shutdown(self) -> None:
        """Disconnect all MCP servers and unregister their tools."""
        for name, client in self._clients.items():
            for tool in client.tools:
                await self._registry.unregister(tool.info().name, scope="mcp")
            await client.disconnect()
            self._log.info(f"MCP server '{name}' disconnected")
        self._clients.clear()

    @property
    def connected_servers(self) -> list[str]:
        """Names of currently-connected MCP servers."""
        return [name for name, c in self._clients.items() if c.is_connected]

    async def reconnect_server(self, server_name: str) -> bool:
        """Manually trigger reconnection for a named server."""
        client = self._clients.get(server_name)
        if client is None:
            self._log.error(f"MCP reconnect: unknown server '{server_name}'")
            return False
        try:
            await client.disconnect()
            await client.connect()
            return client.is_connected
        except Exception as exc:
            self._log.error(f"MCP manual reconnect failed for '{server_name}': {exc}")
            return False
