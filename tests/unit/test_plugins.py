"""Unit tests for PluginLoader: entry point and manifest plugin discovery."""
import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from localharness.config.models import ToolConfig
from localharness.tools.base import Tool, ToolSchema, ToolResult, ToolVetoed
from localharness.tools.hooks import HookSystem, HARNESS_HOOKIMPL
from localharness.tools.registry import ToolRegistry
from localharness.plugins.loader import PluginLoader, PluginManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _PingTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="ping",
            description="Ping.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return self.ok("pong")


def _make_registry() -> ToolRegistry:
    return ToolRegistry()


def _make_hook_system() -> HookSystem:
    return HookSystem()


def _make_loader(
    registry: ToolRegistry,
    hook_system: HookSystem,
    plugins_dir: Path | None = None,
) -> PluginLoader:
    return PluginLoader(registry, hook_system, plugins_dir=plugins_dir)


def _fake_ep(name: str, tool_class: type) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = tool_class
    return ep


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entry_point_discovery():
    """Tools registered via localharness.tools entry point appear in ToolRegistry."""
    registry = _make_registry()
    hs = _make_hook_system()
    loader = _make_loader(registry, hs)

    ep = _fake_ep("ping", _PingTool)
    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.side_effect = lambda group="": [ep] if group == "localharness.tools" else []
        await loader.discover_all()

    # ping should now be registered globally
    tool_config = ToolConfig()
    tools = registry.get_tools_for_agent("agent-1", "default", tool_config)
    assert "ping" in tools


@pytest.mark.asyncio
async def test_manifest_discovery(tmp_path: Path):
    """Tool declared in manifest.yaml is registered after _discover_manifest_plugins."""
    # Build a mini plugin directory
    plugin_dir = tmp_path / "my-plugin"
    plugin_dir.mkdir()

    manifest = {
        "name": "my-plugin",
        "version": "0.1.0",
        "tools": [
            {
                "name": "ping",
                "description": "Ping tool.",
                "entrypoint": "localharness.tools.builtin.glob_tool:GlobTool",  # Use real importable class
            }
        ],
    }
    (plugin_dir / "manifest.yaml").write_text(yaml.dump(manifest))

    registry = _make_registry()
    hs = _make_hook_system()
    loader = _make_loader(registry, hs, plugins_dir=tmp_path)

    with patch("importlib.metadata.entry_points", return_value=[]):
        await loader._discover_manifest_plugins()

    tool_config = ToolConfig()
    tools = registry.get_tools_for_agent("agent-1", "default", tool_config)
    # GlobTool registers as "glob"
    assert "glob" in tools
    assert "my-plugin" in loader._loaded


@pytest.mark.asyncio
async def test_manifest_invalid_yaml_skipped(tmp_path: Path):
    """Invalid manifest.yaml is skipped without crash."""
    plugin_dir = tmp_path / "bad-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "manifest.yaml").write_text(":::invalid yaml:::")

    registry = _make_registry()
    hs = _make_hook_system()
    loader = _make_loader(registry, hs, plugins_dir=tmp_path)

    # Must not raise
    await loader._discover_manifest_plugins()
    assert loader._loaded == []


@pytest.mark.asyncio
async def test_entry_point_hook_discovery():
    """Hook plugins via localharness.hooks entry point are registered in HookSystem."""
    observed: list[str] = []

    class _TestHookPlugin:
        @HARNESS_HOOKIMPL
        def pre_tool(self, name: str, arguments: dict, agent_id: str, division_id: str) -> None:
            observed.append(name)

    registry = _make_registry()
    hs = _make_hook_system()
    loader = _make_loader(registry, hs)

    hook_ep = _fake_ep("test-hook", _TestHookPlugin)

    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.side_effect = lambda group="": (
            [] if group == "localharness.tools" else [hook_ep]
        )
        await loader._discover_entry_point_tools()

    # Wire and fire a dispatch to confirm hook is registered
    from localharness.tools.base import Tool as _Tool, ToolSchema as _TS, ToolResult as _TR

    class _OkTool(_Tool):
        def info(self) -> _TS:
            return _TS(
                name="ok_tool",
                description="ok",
                parameters={"type": "object", "properties": {}, "required": []},
            )

        async def _execute(self, **kwargs: Any) -> _TR:
            return self.ok("ok")

    await registry.register(_OkTool(), scope="global")
    hs.wire_to_registry(registry)
    await registry.dispatch(
        name="ok_tool",
        arguments={},
        agent_id="a",
        division_id="d",
        tool_config=ToolConfig(),
    )
    assert "ok_tool" in observed


@pytest.mark.asyncio
async def test_broken_entry_point_skipped():
    """Entry point that raises ImportError on load is skipped; others still load."""
    ep_broken = MagicMock()
    ep_broken.name = "broken"
    ep_broken.load.side_effect = ImportError("no module named fake")

    ep_good = _fake_ep("ping", _PingTool)

    registry = _make_registry()
    hs = _make_hook_system()
    loader = _make_loader(registry, hs)

    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.side_effect = lambda group="": (
            [ep_broken, ep_good] if group == "localharness.tools" else []
        )
        # Must not raise
        await loader._discover_entry_point_tools()

    tool_config = ToolConfig()
    tools = registry.get_tools_for_agent("agent-1", "default", tool_config)
    assert "ping" in tools
