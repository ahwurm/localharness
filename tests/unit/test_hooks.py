"""Unit tests for HookSystem pluggy integration with ToolRegistry."""
import asyncio
import pytest
from typing import Any

from localharness.config.models import ToolConfig
from localharness.tools.base import Tool, ToolSchema, ToolResult, ToolVetoed
from localharness.tools.registry import ToolRegistry
from localharness.tools.hooks import HarnesHookSpec, HookSystem, HARNESS_HOOKIMPL


# ---------------------------------------------------------------------------
# Minimal test tool
# ---------------------------------------------------------------------------


class _OkTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="ok_tool",
            description="Always succeeds.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return self.ok("ok")


_TOOL_CONFIG = ToolConfig()


async def _dispatch(registry: ToolRegistry, name: str = "ok_tool") -> ToolResult:
    return await registry.dispatch(
        name=name,
        arguments={},
        agent_id="test-agent",
        division_id="default",
        tool_config=_TOOL_CONFIG,
    )


async def _register_ok(registry: ToolRegistry) -> None:
    await registry.register(_OkTool(), scope="global")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_tool_veto():
    """A hookimpl raising ToolVetoed must block execution and return failure."""

    class VetoPlugin:
        @HARNESS_HOOKIMPL
        def pre_tool(self, name: str, arguments: dict, agent_id: str, division_id: str) -> None:
            raise ToolVetoed("blocked by policy")

    registry = ToolRegistry()
    await _register_ok(registry)

    hs = HookSystem()
    hs.register_plugin(VetoPlugin())
    hs.wire_to_registry(registry)

    result = await _dispatch(registry)
    assert result.success is False
    assert "blocked" in result.error


@pytest.mark.asyncio
async def test_pre_tool_exception_swallowed():
    """Non-ToolVetoed exceptions in pre_tool are swallowed; dispatch succeeds."""

    class BoomPlugin:
        @HARNESS_HOOKIMPL
        def pre_tool(self, name: str, arguments: dict, agent_id: str, division_id: str) -> None:
            raise RuntimeError("unexpected crash")

    registry = ToolRegistry()
    await _register_ok(registry)

    hs = HookSystem()
    hs.register_plugin(BoomPlugin())
    hs.wire_to_registry(registry)

    result = await _dispatch(registry)
    assert result.success is True


@pytest.mark.asyncio
async def test_post_tool_fires_after_execution():
    """post_tool hookimpl fires after dispatch and receives the tool name."""
    observed: list[str] = []

    class ObserverPlugin:
        @HARNESS_HOOKIMPL
        def post_tool(
            self, name: str, arguments: dict, result: Any, agent_id: str, division_id: str
        ) -> None:
            observed.append(name)

    registry = ToolRegistry()
    await _register_ok(registry)

    hs = HookSystem()
    hs.register_plugin(ObserverPlugin())
    hs.wire_to_registry(registry)

    await _dispatch(registry)
    assert len(observed) == 1
    assert observed[0] == "ok_tool"


@pytest.mark.asyncio
async def test_post_tool_exception_swallowed():
    """Exceptions in post_tool are swallowed; dispatch still returns successfully."""

    class BoomPostPlugin:
        @HARNESS_HOOKIMPL
        def post_tool(
            self, name: str, arguments: dict, result: Any, agent_id: str, division_id: str
        ) -> None:
            raise RuntimeError("post boom")

    registry = ToolRegistry()
    await _register_ok(registry)

    hs = HookSystem()
    hs.register_plugin(BoomPostPlugin())
    hs.wire_to_registry(registry)

    result = await _dispatch(registry)
    assert result.success is True


@pytest.mark.asyncio
async def test_register_plugin_dedup():
    """Registering the same plugin object twice must not raise (second is no-op)."""

    class NoopPlugin:
        @HARNESS_HOOKIMPL
        def pre_tool(self, name: str, arguments: dict, agent_id: str, division_id: str) -> None:
            pass

    hs = HookSystem()
    plugin = NoopPlugin()
    hs.register_plugin(plugin)
    hs.register_plugin(plugin)  # Must not raise
