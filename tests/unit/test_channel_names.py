"""PAPI-09: the names `start --channel` accepts come from static manifests — core channels, every
bundled manifest of kind "channel", and the legacy discord entry — resolved before any plugin loads."""
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


def test_channel_names_are_core_legacy_and_bundled_channel_manifests(monkeypatch):
    assert channel_names() == {"terminal", "acp", "discord", "web"}
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (ImagePlugin,))
    assert channel_names() == {"terminal", "acp", "discord"}
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (ImagePlugin, _FakeChannel, _FakeTools))
    assert channel_names() == {"terminal", "acp", "discord", "fakechan"}


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
