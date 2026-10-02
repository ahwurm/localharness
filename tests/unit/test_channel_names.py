"""PAPI-09: the names `start --channel` accepts come from static manifests — core channels, every
bundled manifest of kind "channel" (discord is the bundled dispatch plugin's) — resolved before any
plugin loads."""
from __future__ import annotations

import pytest
import typer

from localharness.plugins import builtin
from localharness.plugins.api import Plugin, PluginManifest
from localharness.plugins.channels import channel_names
from localharness.tools.builtin.image_plugin import ImagePlugin


class _FakeChannel(Plugin):
    manifest = PluginManifest(name="fakechan", version="0.1.0", kind="channel")


class _FakeTools(Plugin):
    manifest = PluginManifest(name="faketools", version="0.1.0", kind="tools")


class _Exploding(Plugin):
    manifest = PluginManifest(name="boomchan", version="0.1.0", kind="channel")

    def __init__(self) -> None:
        raise AssertionError("the resolver must never instantiate a plugin")


def test_channel_names_are_core_and_bundled_channel_manifests(monkeypatch):
    from localharness.dispatch.plugin import DispatchPlugin

    assert channel_names() == {"terminal", "acp", "discord", "web"}
    # discord comes only from the bundled dispatch plugin's manifest: without it, no discord
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (ImagePlugin,))
    assert channel_names() == {"terminal", "acp"}
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (ImagePlugin, DispatchPlugin))
    assert channel_names() == {"terminal", "acp", "discord"}
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (ImagePlugin, _FakeChannel, _FakeTools))
    assert channel_names() == {"terminal", "acp", "fakechan"}


def test_resolver_never_instantiates(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (_Exploding,))
    assert "boomchan" in channel_names()


@pytest.mark.asyncio
async def test_a_typo_is_refused_before_any_plugin_loads(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async

    def boom(*a, **k):
        raise AssertionError("resolve() must not run before the channel gate")
    monkeypatch.setattr("localharness.plugins.resolve.resolve", boom)
    with pytest.raises(typer.BadParameter) as exc:
        await _start_async(None, False, False, str(tmp_path), channel_mode="wbe")
    assert "unknown channel 'wbe'; choose one of: acp, discord, terminal, web" in str(exc.value)


@pytest.mark.asyncio
async def test_web_and_acp_without_their_channel_are_sent_to_their_own_command(tmp_path):
    from localharness.cli.start_cmd import WEB_NEEDS_ITS_OWN_COMMAND, _start_async

    assert WEB_NEEDS_ITS_OWN_COMMAND == (
        "the web channel is served by its own command, because the HTTP server has to be reachable "
        "before a session exists. Run `localharness web` instead of `localharness start --channel web`.")
    with pytest.raises(typer.BadParameter) as exc:
        await _start_async(None, False, False, str(tmp_path), channel_mode="acp")
    assert "Run `localharness acp` instead of `localharness start --channel acp`" in str(exc.value)
    assert "plugins enable" not in str(exc.value)  # acp is core, not a plugin
    with pytest.raises(typer.BadParameter) as exc:
        await _start_async(None, False, False, str(tmp_path), channel_mode="web")
    assert WEB_NEEDS_ITS_OWN_COMMAND in str(exc.value)
    assert "(if that command is missing, run `localharness plugins enable web`)" in str(exc.value)


def test_the_channel_help_lists_the_resolved_names():
    from localharness.cli.app import app

    start = typer.main.get_command(app).commands["start"]
    (opt,) = [p for p in start.params if p.name == "channel"]
    assert "acp, discord, terminal, web" in opt.help


# ------------------------------------------------------------------ the needs-extra line at start (46, ruling 5)

def _no_extra(monkeypatch):
    from localharness.plugins import resolve
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: False)


@pytest.mark.asyncio
async def test_start_without_the_web_extra_does_not_warn_about_web(tmp_path, monkeypatch):
    """A terminal-only install never asked for the phone: no `plugin web:` line at start."""
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    _no_extra(monkeypatch)
    await _start_async(None, False, False, str(tmp_path))
    assert any("startup)" in line for line in printed), printed
    assert not any("plugin web:" in line for line in printed), printed


def test_plugins_list_still_shows_the_missing_extra(monkeypatch):
    from typer.testing import CliRunner

    from localharness.cli.app import app

    _no_extra(monkeypatch)
    monkeypatch.setenv("COLUMNS", "400")
    out = CliRunner().invoke(app, ["plugins", "list"])
    assert out.exit_code == 0, out.output
    assert "on (install `localharness[web]` to use it)" in out.output


def test_an_installed_plugins_needs_extra_still_warns_at_start():
    """The filter drops only BUNDLED needs-extra lines: an installed plugin's user opted in."""
    from localharness.cli.start_cmd import _start_problems
    from localharness.plugins.plan import LoadPlan, PlanEntry
    from localharness.plugins.resolve import Resolution

    reason = "install `localharness[web]` to use it"
    entries = (PlanEntry("web", True, "needs-extra", "built in", "", reason),
               PlanEntry("thirdparty", False, "needs-extra", "entry point", "", "install `thirdparty[x]` to use it"),
               PlanEntry("broken", True, "failed", "built in", "", "boom"))
    res = Resolution(plan=LoadPlan(entries=entries, order=(), memory_occupant=None),
                     classes={}, settings={}, enabled={}, warnings=())
    assert _start_problems(res) == ["plugin thirdparty: install `thirdparty[x]` to use it",
                                    "plugin broken: boom"]


def test_the_start_filter_reads_entries_never_rendered_text():
    """Two entries that render to the same line differ only in `bundled`: the filter must keep the
    installed one, which a set-difference over rendered strings cannot do."""
    import inspect

    from localharness.cli.start_cmd import _start_problems
    from localharness.plugins.plan import LoadPlan, PlanEntry
    from localharness.plugins.resolve import Resolution

    entries = (PlanEntry("twin", True, "needs-extra", "built in", "", "install it"),
               PlanEntry("twin", False, "needs-extra", "entry point", "", "install it"))
    res = Resolution(plan=LoadPlan(entries=entries, order=(), memory_occupant=None),
                     classes={}, settings={}, enabled={}, warnings=())
    assert _start_problems(res) == ["plugin twin: install it"]
    assert 'f"plugin {' not in inspect.getsource(_start_problems)


def test_start_and_the_resolver_name_the_same_own_command_channels():
    """start's one own-command table and the resolver's OWN_COMMAND cannot drift: a name in one but
    not the other would be a KeyError at `--channel <name>`."""
    from localharness.cli.start_cmd import _own_command
    from localharness.plugins.channels import OWN_COMMAND

    assert set(_own_command(web_channel=None, acp_channel=None)) == OWN_COMMAND


def test_a_channel_plugin_turned_off_is_not_accepted(tmp_path, monkeypatch):
    """49-02: enablement decides the accepted set — channel_names() still lists it (for --help)."""
    import yaml

    from localharness.config.loader import ConfigLoader
    from localharness.plugins import discovery
    from localharness.plugins.channels import accepted_channels
    from localharness.plugins.resolve import resolve

    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (_FakeChannel,))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    g = tmp_path / "g"
    g.mkdir()
    base = {"version": "1", "provider": {"provider_type": "vllm",
            "base_url": "http://localhost:8000/v1", "default_model": "m"}}
    (g / "config.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    assert "fakechan" in accepted_channels(resolve(ConfigLoader(config_dir=g)))
    (g / "config.yaml").write_text(yaml.safe_dump({**base, "fakechan": {"enabled": False}}),
                                   encoding="utf-8")
    assert accepted_channels(resolve(ConfigLoader(config_dir=g))) == {"terminal", "acp"}
    assert "fakechan" in channel_names()


def test_disabling_dispatch_removes_discord(tmp_path, monkeypatch):
    """The real bundled dispatch plugin, turned off in the global config: `discord` leaves the
    accepted set while channel_names() (the --help menu) still lists it."""
    import yaml

    from localharness.config.loader import ConfigLoader
    from localharness.plugins import discovery, resolve as resolve_mod
    from localharness.plugins.channels import accepted_channels
    from localharness.plugins.resolve import resolve

    monkeypatch.setitem(resolve_mod.resolve.__kwdefaults__, "extra_installed", lambda e: True)
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    g = tmp_path / "g"
    g.mkdir()
    base = {"version": "1", "provider": {"provider_type": "vllm",
            "base_url": "http://localhost:8000/v1", "default_model": "m"}}
    (g / "config.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    assert "discord" in accepted_channels(resolve(ConfigLoader(config_dir=g)))
    (g / "config.yaml").write_text(yaml.safe_dump({**base, "dispatch": {"enabled": False}}),
                                   encoding="utf-8")
    res = resolve(ConfigLoader(config_dir=g))
    assert {e.name: e.state for e in res.plan.entries}["dispatch"] == "off"
    assert "discord" not in accepted_channels(res)
    assert "discord" in channel_names()
