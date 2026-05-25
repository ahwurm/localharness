"""Tests for MCP client integration (MCPToolWrapper, MCPServerClient, MCPClientManager)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from localharness.tools.mcp import MCPToolWrapper, MCPServerClient, MCPClientManager
from localharness.tools.base import ToolSchema, ToolResult
from localharness.tools.registry import ToolRegistry
from localharness.config.models import MCPServerConfig


class MockTextContent:
    def __init__(self, text: str):
        self.text = text


class MockCallToolResult:
    def __init__(self, text: str, is_error: bool = False):
        self.content = [MockTextContent(text)]
        self.isError = is_error


class MockTool:
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class MockListToolsResult:
    def __init__(self, tools: list):
        self.tools = tools


# ---------------------------------------------------------------------------
# MCPToolWrapper
# ---------------------------------------------------------------------------

def test_mcp_tool_wrapper_info_returns_valid_schema():
    session = AsyncMock()
    wrapper = MCPToolWrapper(
        name="search",
        description="Search web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        session=session,
        server_name="exa",
    )
    schema = wrapper.info()
    assert isinstance(schema, ToolSchema)
    assert schema.name == "exa__search"
    assert schema.scope == "mcp"
    assert "query" in schema.parameters.get("properties", {})


@pytest.mark.asyncio
async def test_mcp_tool_wrapper_run_success():
    session = AsyncMock()
    session.call_tool = AsyncMock(return_value=MockCallToolResult("search results here"))
    wrapper = MCPToolWrapper(
        name="search",
        description="Search",
        input_schema={},
        session=session,
        server_name="exa",
    )
    result = await wrapper.run(query="test")
    assert result.success is True
    assert "search results here" in result.output
    session.call_tool.assert_called_once_with("search", arguments={"query": "test"})


@pytest.mark.asyncio
async def test_mcp_tool_wrapper_run_error():
    session = AsyncMock()
    session.call_tool = AsyncMock(return_value=MockCallToolResult("not found", is_error=True))
    wrapper = MCPToolWrapper(
        name="search",
        description="Search",
        input_schema={},
        session=session,
        server_name="exa",
    )
    result = await wrapper.run(query="bad")
    assert result.success is False
    assert result.error_type == "execution_error"


@pytest.mark.asyncio
async def test_mcp_tool_wrapper_run_exception():
    session = AsyncMock()
    session.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))
    wrapper = MCPToolWrapper(
        name="search",
        description="Search",
        input_schema={},
        session=session,
        server_name="exa",
    )
    result = await wrapper.run(query="test")
    assert result.success is False
    assert "connection lost" in result.error


# ---------------------------------------------------------------------------
# MCPClientManager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_client_manager_registers_tools():
    registry = ToolRegistry()
    manager = MCPClientManager(registry)

    config = MCPServerConfig(name="test-server", transport="stdio", command="echo")

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MockListToolsResult([
        MockTool("tool_a", "Tool A", {"type": "object", "properties": {}}),
        MockTool("tool_b", "Tool B", {"type": "object", "properties": {}}),
    ]))
    mock_session.call_tool = AsyncMock(return_value=MockCallToolResult("ok"))

    # Directly build client and inject state (avoids spawning real process)
    client = MCPServerClient(config)
    client._session = mock_session
    client._connected = True
    client._tools = [
        MCPToolWrapper("tool_a", "Tool A", {"type": "object", "properties": {}}, mock_session, "test-server"),
        MCPToolWrapper("tool_b", "Tool B", {"type": "object", "properties": {}}, mock_session, "test-server"),
    ]
    manager._clients["test-server"] = client
    for tool in client.tools:
        await registry.register(tool, scope="mcp")

    # Verify tools are registered
    tool_config = MagicMock(inherit=["global"], add=[], deny=[])
    schemas = registry.get_tools_for_agent("any-agent", "default", tool_config)
    assert "test-server__tool_a" in schemas
    assert "test-server__tool_b" in schemas


@pytest.mark.asyncio
async def test_mcp_client_manager_shutdown_unregisters():
    registry = ToolRegistry()
    manager = MCPClientManager(registry)

    mock_session = AsyncMock()
    config = MCPServerConfig(name="s1", transport="stdio", command="echo")
    client = MCPServerClient(config)
    client._session = mock_session
    client._connected = True
    wrapper = MCPToolWrapper("mytool", "desc", {}, mock_session, "s1")
    client._tools = [wrapper]
    await registry.register(wrapper, scope="mcp")
    manager._clients["s1"] = client

    await manager.shutdown()

    tool_config = MagicMock(inherit=["global"], add=[], deny=[])
    schemas = registry.get_tools_for_agent("a", "d", tool_config)
    assert "s1__mytool" not in schemas


@pytest.mark.asyncio
async def test_mcp_server_failure_non_fatal():
    """A server that fails to connect should not prevent other servers from connecting."""
    registry = ToolRegistry()
    manager = MCPClientManager(registry)

    config_bad = MCPServerConfig(name="bad", transport="stdio", command="/nonexistent")
    config_good = MCPServerConfig(name="good", transport="stdio", command="echo")

    call_count = [0]

    async def side_effect_connect():
        call_count[0] += 1
        if call_count[0] == 1:
            raise ConnectionError("Failed")

    with patch.object(MCPServerClient, "connect", side_effect=side_effect_connect):
        results = await manager.startup([config_bad, config_good])

    assert results["bad"] == 0  # failed


@pytest.mark.asyncio
async def test_mcp_tool_wrapper_prefix_false():
    """When prefix=False the tool name is bare (no server prefix)."""
    session = AsyncMock()
    wrapper = MCPToolWrapper(
        name="mytool",
        description="desc",
        input_schema={},
        session=session,
        server_name="myserver",
        prefix=False,
    )
    assert wrapper.info().name == "mytool"


@pytest.mark.asyncio
async def test_connected_servers_property():
    registry = ToolRegistry()
    manager = MCPClientManager(registry)

    mock_session = AsyncMock()
    config = MCPServerConfig(name="s1", transport="stdio", command="echo")
    client = MCPServerClient(config)
    client._session = mock_session
    client._connected = True
    client._tools = []
    manager._clients["s1"] = client

    assert "s1" in manager.connected_servers
