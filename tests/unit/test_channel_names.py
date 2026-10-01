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
