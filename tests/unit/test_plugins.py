"""The plugin substrate's discovery → resolve → lifecycle path, standing in for the deleted legacy loader.

The dormant legacy loader (entry-point groups for bare tools and bare hooks, a manifest-file scan,
sys.path insertion, nothing contained or attributed) was deleted in 44-14. These nine tests are its
nine, rewritten: each docstring names the test it replaces and what it still proves on the one path
every plugin now takes — metadata-only discovery (`localharness.plugins` entry points and
`<global config dir>/plugins/<name>/`), `resolve()` (imports only what is enabled) and
`start_plugins()` (registers tools bare with source_plugin set; every stage contained).

The mocked entry-point idiom (`patch("importlib.metadata.entry_points")`) is kept for these unit tests;
the real installed distribution is proven end to end by the phase's e2e test.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from localharness.config.loader import ConfigLoader
from localharness.config.models import ToolConfig
from localharness.core.bus import EventBus
from localharness.plugins import builtin
from localharness.plugins.api import Plugin, PluginManifest, PluginPaths
from localharness.plugins.lifecycle import start_plugins
from localharness.plugins.resolve import Resolution, resolve
from localharness.tools.base import Tool, ToolResult, ToolSchema
from localharness.tools.hooks import HARNESS_HOOKIMPL, HookSystem
from localharness.tools.registry import ToolRegistry

_DECLARED = {"ingest": "none", "host": "safe", "result_origin": "trusted"}  # a tool co-resides with bash
_NEVER = 'raise AssertionError("a plugin that is not enabled was imported")\n'
OBSERVED: list[str] = []


class _PingTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(name="ping", description="Ping.", parameters={}, **_DECLARED)

    async def _execute(self, **_: Any) -> ToolResult:
        return self.ok("pong")


class _OkTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(name="ok_tool", description="ok", parameters={}, **_DECLARED)

    async def _execute(self, **_: Any) -> ToolResult:
        return self.ok("ok")


class _PingPlugin(Plugin):
    """answers ping"""

    manifest = PluginManifest(name="ping", version="0.1.0", kind="tools")

    async def tools(self, ctx: Any) -> list:
        return [_PingTool()]


class _Hooks:
    @HARNESS_HOOKIMPL
    def pre_tool(self, name: str, arguments: dict, agent_id: str, division_id: str) -> None:
        OBSERVED.append(name)


class _HookPlugin(Plugin):
    """watches every tool call"""

    manifest = PluginManifest(name="hooky", version="0.1.0", kind="dev")

    async def start(self, ctx: Any) -> None:
        ctx.hooks.register_plugin(_Hooks(), name="hooky")


_FOLDER_PLUGIN = textwrap.dedent('''\
    from localharness.plugins.api import Plugin, PluginManifest
    from localharness.tools.base import Tool, ToolSchema

    class _Tool(Tool):
        def info(self):
            return ToolSchema(name="{name}_echo", description="Echo.", parameters={{}},
                              ingest="none", host="safe", result_origin="trusted")

        async def _execute(self, **kw):
            return self.ok("{name} echoes")

    class ThePlugin(Plugin):
        """a folder plugin"""
        manifest = PluginManifest(name="{name}", version="0.1.0", kind="tools")

        async def tools(self, ctx):
            return [_Tool()]

    plugin = ThePlugin
    ''')


@pytest.fixture(autouse=True)
def _clean():
    OBSERVED.clear()
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


def _fake_ep(name: str, target: str) -> MagicMock:
    """An installed package's `localharness.plugins` entry point: metadata, and a load() nothing
    may call — core imports the target itself, and only once the plugin is enabled."""
    ep = MagicMock()
    ep.name, ep.value = name, target
    ep.dist.name, ep.dist.version = f"lh-{name}", "0.1.0"
    return ep


def _entry_points(*eps: MagicMock):
    return patch("importlib.metadata.entry_points",
                 side_effect=lambda group=None: list(eps) if group == "localharness.plugins" else [])


def _config(g: Path, **sections: Any) -> Path:
    """`<g>/config.yaml`: a minimal valid config plus plugin sections (the machine-level layer)."""
    g.mkdir(parents=True, exist_ok=True)
    (g / "config.yaml").write_text(yaml.safe_dump({
        "version": "1", "provider": {"provider_type": "vllm", "base_url": "http://localhost:8000/v1",
                                     "default_model": "m"}, **sections}))
    return g


def _folder(g: Path, name: str, body: str) -> Path:
    folder = g / "plugins" / name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text(body)
    return folder


async def _started(resolution: Resolution, g: Path, hooks: HookSystem | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    await start_plugins(resolution, bus=EventBus(), registry=registry, hooks=hooks, llm=None,
                        paths=PluginPaths(global_config_dir=g, workspace=None, state_dir=g))
    return registry


def _listed(loader: ConfigLoader) -> dict[str, tuple[str, str]]:
    return {e.name: (e.state, e.source) for e in resolve(loader).plan.entries}


async def test_an_entry_point_plugin_is_listed_then_registers_once_enabled(tmp_path):
    """Replaces test_entry_point_discovery (a `localharness.tools` entry point's tool reached the
    registry). Still proves an installed package's tool reaches the registry and dispatches — now
    listed from its `localharness.plugins` metadata alone while it is not enabled (ep.load() is never
    called), and once enabled registered under its bare name at global scope with source_plugin set."""
    ep = _fake_ep("ping", "tests.unit.test_plugins:_PingPlugin")
    g = _config(tmp_path / "g")
    with _entry_points(ep):
        assert resolve(ConfigLoader(config_dir=g)).plan.entry("ping").state == "available"
        _config(g, ping={"enabled": True})
        registry = await _started(resolve(ConfigLoader(config_dir=g)), g)
    ep.load.assert_not_called()
    assert "ping" in {s.name for s in registry.global_schemas()}
    assert registry.schema_of("ping").source_plugin == "ping"
    result = await registry.dispatch("ping", {}, "agent-1", "default", ToolConfig())
    assert result.output == "pong"


async def test_an_enabled_folder_plugin_registers_its_tool(tmp_path):
    """Replaces test_manifest_discovery (a manifest-file plugin dir's tool was registered). Still
    proves a drop-in plugin folder under the machine's config dir contributes a working tool — now
    `<global config dir>/plugins/<name>/__init__.py` binding `plugin`, imported once enabled."""
    g = _config(tmp_path / "g", myplug={"enabled": True})
    _folder(g, "myplug", _FOLDER_PLUGIN.format(name="myplug"))
    with _entry_points():
        registry = await _started(resolve(ConfigLoader(config_dir=g)), g)
    assert registry.schema_of("myplug_echo").source_plugin == "myplug"
    result = await registry.dispatch("myplug_echo", {}, "agent-1", "default", ToolConfig())
    assert result.output == "myplug echoes"


async def test_a_broken_folder_plugin_is_contained_and_the_others_load(tmp_path):
    """Replaces test_manifest_invalid_yaml_skipped (an unparseable manifest was skipped without a
    crash). Still proves a broken drop-in plugin costs only itself — now a folder whose code does not
    even parse is `failed` with the reason named, and the next plugin still loads."""
    g = _config(tmp_path / "g", broken={"enabled": True}, good={"enabled": True})
    _folder(g, "broken", "def (:\n")
    _folder(g, "good", _FOLDER_PLUGIN.format(name="good"))
    with _entry_points():
        resolution = resolve(ConfigLoader(config_dir=g))
        registry = await _started(resolution, g)
    broken = resolution.plan.entry("broken")
    assert broken.state == "failed" and "SyntaxError" in broken.reason
    assert any(p.startswith("plugin broken:") for p in resolution.problems())
    assert registry.schema_of("good_echo").source_plugin == "good"


async def test_an_enabled_plugins_hook_fires_on_a_real_dispatch(tmp_path):
    """Replaces test_entry_point_hook_discovery (a `localharness.hooks` entry point's hook object was
    registered and fired). Still proves a plugin's pre_tool hook fires on a dispatched tool through
    the real registry — now registered by the plugin itself on ctx.hooks in start()."""
    ep = _fake_ep("hooky", "tests.unit.test_plugins:_HookPlugin")
    g = _config(tmp_path / "g", hooky={"enabled": True})
    hooks = HookSystem()
    with _entry_points(ep):
        registry = await _started(resolve(ConfigLoader(config_dir=g)), g, hooks=hooks)
    hooks.wire_to_registry(registry)
    await registry.register(_OkTool(), scope="global")
    await registry.dispatch("ok_tool", {}, "a", "d", ToolConfig())
    assert OBSERVED == ["ok_tool"]


async def test_a_broken_entry_point_is_failed_and_the_others_load(tmp_path):
    """Replaces test_broken_entry_point_skipped (an entry point whose load raised was skipped, the
    others loaded). Still proves one bad package cannot stop the rest — now an enabled entry point
    whose import raises is `failed` with the reason named, and the good one registers."""
    broken = _fake_ep("broken", "lh_no_such_module:Plugin")
    good = _fake_ep("ping", "tests.unit.test_plugins:_PingPlugin")
    g = _config(tmp_path / "g", broken={"enabled": True}, ping={"enabled": True})
    with _entry_points(broken, good):
        resolution = resolve(ConfigLoader(config_dir=g))
        registry = await _started(resolution, g)
    entry = resolution.plan.entry("broken")
    assert entry.state == "failed" and "ModuleNotFoundError" in entry.reason
    assert registry.schema_of("ping").source_plugin == "ping"


def test_an_explicit_config_dir_roots_the_plugin_folder(tmp_path, monkeypatch):
    """Replaces test_plugins_dir_explicit_arg_unchanged (an explicit plugins_dir was honoured exactly;
    the env chain never re-rooted it). Still proves it: plugin folders come from `<config dir>/plugins`
    of the config dir the session names, whatever LOCALHARNESS_DIR says."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())  # discovery mechanics, not the bundled list: swap it out (as test_doctor_plugins does)
    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "ignored"))
    _folder(_config(tmp_path / "ignored"), "decoy", _NEVER)
    explicit = _config(tmp_path / "explicit")
    mine = _folder(explicit, "mine", _NEVER)
    with _entry_points():
        assert _listed(ConfigLoader(config_dir=explicit)) == {"mine": ("available", f"folder: {mine}")}


def test_the_plugin_folder_follows_the_env_chain(tmp_path, monkeypatch):
    """Replaces test_plugins_dir_default_honors_env_chain (a --config-dir / LOCALHARNESS_DIR session
    loads ITS plugins, #150 phase 38). Still proves it: with no dir named, LOCALHARNESS_HOME roots the
    plugin folder, and LOCALHARNESS_DIR wins over it."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())  # discovery mechanics, not the bundled list: swap it out (as test_doctor_plugins does)
    home_env, dir_env = _config(tmp_path / "home"), _config(tmp_path / "dir")
    at_home, at_dir = _folder(home_env, "at_home", _NEVER), _folder(dir_env, "at_dir", _NEVER)
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home_env))
    with _entry_points():
        assert _listed(ConfigLoader()) == {"at_home": ("available", f"folder: {at_home}")}
        monkeypatch.setenv("LOCALHARNESS_DIR", str(dir_env))
        assert _listed(ConfigLoader()) == {"at_dir": ("available", f"folder: {at_dir}")}


def test_with_nothing_set_the_plugin_folder_is_the_default_dir(tmp_path, fake_home, monkeypatch):
    """Replaces test_plugins_dir_default_unchanged_when_nothing_set (no env, no arg: the default
    ~/.localharness/plugins). Still proves it, against a fake home: `~/.localharness/plugins`."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())  # discovery mechanics, not the bundled list: swap it out (as test_doctor_plugins does)
    home = fake_home(tmp_path / "home")  # also clears LOCALHARNESS_DIR / LOCALHARNESS_HOME
    default = _config(home / ".localharness")
    found = _folder(default, "stock", _NEVER)
    loader = ConfigLoader()
    assert loader.global_config_dir == default
    with _entry_points():
        assert _listed(loader) == {"stock": ("available", f"folder: {found}")}


def test_the_plugin_folder_is_read_at_call_time(tmp_path, monkeypatch):
    """Replaces test_plugins_dir_resolved_at_construction_not_import (v013 Risk #3: never capture a
    per-invocation path at import time; two loaders built under different env disagree). Still
    proves it, and more: a folder created after the first resolve is found by the next one, and a
    loader built under a different env sees only its own dir's plugins."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())  # discovery mechanics, not the bundled list: swap it out (as test_doctor_plugins does)
    first, second = _config(tmp_path / "first"), _config(tmp_path / "second")
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(first))
    loader = ConfigLoader()
    with _entry_points():
        assert _listed(loader) == {}
        late = _folder(first, "late", _NEVER)
        assert _listed(loader) == {"late": ("available", f"folder: {late}")}
        other = _folder(second, "other", _NEVER)
        monkeypatch.setenv("LOCALHARNESS_HOME", str(second))
        assert _listed(ConfigLoader()) == {"other": ("available", f"folder: {other}")}
