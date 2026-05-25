"""PluginLoader: discovers tools and hook plugins from entry points and manifest.yaml."""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict

from localharness.tools.base import ToolProtocol

if TYPE_CHECKING:
    from localharness.tools.hooks import HookSystem
    from localharness.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class PluginToolDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = {}
    entrypoint: str | None = None  # "package.module:ClassName"


class PluginManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    author: str = ""
    description: str = ""
    tools: list[PluginToolDef] = []
    hooks: list[str] = []  # module paths to hookimpl classes e.g. "mypkg.hooks:MyHooks"
    config_keys: list[str] = []  # required env vars / config keys


class PluginLoader:
    """Discovers tools and hook plugins from entry points and drop-in manifests."""

    def __init__(
        self,
        registry: "ToolRegistry",
        hook_system: "HookSystem",
        plugins_dir: Path | None = None,
    ) -> None:
        self._registry = registry
        self._hook_system = hook_system
        self._plugins_dir = plugins_dir or Path.home() / ".localharness" / "plugins"
        self._loaded: list[str] = []

    async def discover_all(self) -> list[str]:
        """Run full discovery. Returns list of loaded plugin names."""
        await self._discover_entry_point_tools()
        await self._discover_manifest_plugins()
        return list(self._loaded)

    async def _discover_entry_point_tools(self) -> None:
        """Load tools via group='localharness.tools' and hooks via 'localharness.hooks'."""
        # Tools
        eps = importlib.metadata.entry_points(group="localharness.tools")
        for ep in eps:
            try:
                tool_class = ep.load()
                instance = tool_class()
                if isinstance(instance, ToolProtocol):
                    await self._registry.register(instance, scope="global")
                    self._loaded.append(ep.name)
                    log.info("tool_plugin_loaded_via_entrypoint tool=%s", ep.name)
            except Exception:
                log.warning("tool_plugin_entrypoint_failed ep=%s", ep.name, exc_info=True)

        # Hook plugins
        hook_eps = importlib.metadata.entry_points(group="localharness.hooks")
        for ep in hook_eps:
            try:
                hook_class = ep.load()
                instance = hook_class()
                self._hook_system.register_plugin(instance)
                self._loaded.append(ep.name)
                log.info("hook_plugin_loaded_via_entrypoint ep=%s", ep.name)
            except Exception:
                log.warning("hook_plugin_entrypoint_failed ep=%s", ep.name, exc_info=True)

    async def _discover_manifest_plugins(self) -> None:
        """Scan plugins_dir subdirectories for manifest.yaml."""
        if not self._plugins_dir.exists():
            return

        for subdir in self._plugins_dir.iterdir():
            if not subdir.is_dir():
                continue
            manifest_path = subdir / "manifest.yaml"
            if not manifest_path.exists():
                continue
            try:
                data = yaml.safe_load(manifest_path.read_text())
                manifest = PluginManifest(**data)
            except Exception:
                log.warning(
                    "manifest_parse_failed path=%s", manifest_path, exc_info=True
                )
                continue

            # Add subdir to sys.path so plugin modules are importable
            subdir_str = str(subdir)
            if subdir_str not in sys.path:
                sys.path.insert(0, subdir_str)

            # Register tools
            for tool_def in manifest.tools:
                await self._load_manifest_tool(tool_def, subdir, manifest.name)

            # Register hooks
            for hook_path in manifest.hooks:
                await self._load_manifest_hook(hook_path, manifest.name)

            self._loaded.append(manifest.name)
            log.info("manifest_plugin_loaded name=%s version=%s", manifest.name, manifest.version)

    async def _load_manifest_tool(
        self, tool_def: PluginToolDef, subdir: Path, plugin_name: str
    ) -> None:
        try:
            if tool_def.entrypoint:
                module_path, class_name = tool_def.entrypoint.rsplit(":", 1)
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name)
            else:
                # Convention: look for tools.py in the subdir
                module_path = f"{subdir.name}.tools"
                class_name = "".join(p.title() for p in tool_def.name.split("_")) + "Tool"
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name)
            instance = tool_class()
            await self._registry.register(instance, scope="global")
        except Exception:
            log.warning(
                "manifest_tool_load_failed tool=%s plugin=%s",
                tool_def.name,
                plugin_name,
                exc_info=True,
            )

    async def _load_manifest_hook(self, hook_path: str, plugin_name: str) -> None:
        try:
            module_path, class_name = hook_path.rsplit(":", 1)
            module = importlib.import_module(module_path)
            hook_class = getattr(module, class_name)
            instance = hook_class()
            self._hook_system.register_plugin(instance)
        except Exception:
            log.warning(
                "manifest_hook_load_failed path=%s plugin=%s",
                hook_path,
                plugin_name,
                exc_info=True,
            )
