"""DispatchPlugin (49-05): manifest, settings, deprecation lines, doctor rows, make_channel — unit
level and through the real lifecycle (start_plugins / doctor_rows). Not yet in BUILTIN_PLUGINS.

Every test runs with HOME and the Discord env sources isolated (tests/dispatch_support.py), so the
box's real bot-token file is never read; a fake `discord` module proves start() opens no client."""
from __future__ import annotations

import pytest

from localharness.core.bus import EventBus
from localharness.dispatch.config import DispatchConfig
from localharness.dispatch.plugin import NOT_CONFIGURED_HINT, DispatchPlugin
from localharness.plugins.api import PluginContext, PluginPaths, plugin_summary
from localharness.plugins.lifecycle import doctor_rows, start_plugins, stop_plugins
from localharness.plugins.plan import build_load_plan
from localharness.plugins.resolve import PluginSettings, Resolution
from localharness.tools.registry import ToolRegistry
from tests.dispatch_support import install_fake_discord, isolate_discord_env

TOKEN = "tok-SECRET-123"


@pytest.fixture(autouse=True)
def fake(monkeypatch, tmp_path):
    isolate_discord_env(monkeypatch, tmp_path)
    return install_fake_discord(monkeypatch)


def _paths(tmp_path) -> PluginPaths:
    return PluginPaths(global_config_dir=tmp_path / "global", workspace=None, state_dir=tmp_path / "state")


def _ctx(tmp_path, **discord) -> PluginContext:
    return PluginContext(bus=EventBus(), tools=ToolRegistry(), hooks=None,
                         config=DispatchConfig(discord=discord), agent_config=None,
                         paths=_paths(tmp_path), llm=None)


def _resolution(**discord) -> Resolution:
    plan = build_load_plan(bundled=(DispatchPlugin,), discovered=[], enabled={"dispatch": True},
                           imported={}, version="0.15.0", core_keys=frozenset({"org"}),
                           extra_installed=lambda extra: True)
    return Resolution(plan, {"dispatch": DispatchPlugin},
                      {"dispatch": PluginSettings(DispatchConfig(discord=discord), None)},
                      {"dispatch": True}, ())


# --- manifest -------------------------------------------------------------------------------------

def test_manifest():
    from localharness.dispatch.adapters import ADAPTERS
    from localharness.dispatch.channel import DispatchChannel

    m = DispatchPlugin.manifest
    assert (m.name, m.version, m.kind, m.enabled_by_default, m.requires_extra) == (
        "dispatch", "0.1.0", "channel", True, "dispatch")
    assert m.channels == ("discord",) == tuple(ADAPTERS) == tuple(DispatchPlugin().channels())
    assert DispatchPlugin().channels() == {"discord": DispatchChannel}
    assert [(f.key, f.secret) for f in m.setup] == [("discord.token", True), ("discord.allow", False)]
    assert "Message Content intent" in m.setup_help and "User ID" in m.setup_help
    assert len(m.setup_help.splitlines()) == 3
    assert "never print the token" in m.agent_prompt and "{" not in m.agent_prompt
    assert m.next_steps == "Then start the Discord session: localharness start --channel discord"
    assert DispatchPlugin.ConfigModel is DispatchConfig and DispatchPlugin.AgentConfigModel is None
    assert DispatchPlugin.wants_artifacts is False
    assert plugin_summary(DispatchPlugin) == "chat: Discord"


def test_channels_reads_the_adapter_registry_at_call_time(monkeypatch):
    from localharness.dispatch import adapters

    monkeypatch.setitem(adapters.ADAPTERS, "fixturechat", "tests.nowhere:Nothing")
    assert set(DispatchPlugin().channels()) == {"discord", "fixturechat"}


def test_registered_fourth_and_once():
    from localharness.plugins.builtin import BUILTIN_PLUGINS

    # dispatch (49) is bundled and on by default (49-06 registered it); autoresearch (50) follows it
    assert BUILTIN_PLUGINS[3] is DispatchPlugin and BUILTIN_PLUGINS.count(DispatchPlugin) == 1


# --- configure / start ----------------------------------------------------------------------------

async def test_configure_is_always_ready(tmp_path):
    assert await DispatchPlugin().configure(_ctx(tmp_path)) == "ready"


async def test_start_warns_once_per_deciding_env_source_and_opens_nothing(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("LOCALHARNESS_DISCORD_ALLOW", "42")
    p = DispatchPlugin()
    await p.start(_ctx(tmp_path))
    assert len(p.startup_warnings) == 1
    assert "LOCALHARNESS_DISCORD_ALLOW" in p.startup_warnings[0]
    assert "dispatch.discord.allow" in p.startup_warnings[0]
    assert fake.client is None and fake.log == []


async def test_start_without_env_has_no_lines(tmp_path):
    p = DispatchPlugin()
    await p.start(_ctx(tmp_path, token=TOKEN, allow=["42"]))
    assert p.startup_warnings == []


async def test_lifecycle_routes_the_line_onto_the_session_warnings(tmp_path, monkeypatch, fake):
    """The real start_plugins: the deprecation line reaches result.warnings (the banner)."""
    monkeypatch.setenv("LOCALHARNESS_DISCORD_TOKEN", TOKEN)
    result = await start_plugins(_resolution(allow=["42"]), bus=EventBus(), registry=ToolRegistry(),
                                 hooks=None, llm=None, paths=_paths(tmp_path))
    try:
        assert [r.name for r in result.running] == ["dispatch"]
        dep = [w for w in result.warnings if "deprecated" in w]
        assert len(dep) == 1 and "LOCALHARNESS_DISCORD_TOKEN" in dep[0] and TOKEN not in dep[0]
        ch = result.running[0].plugin.make_channel("discord", EventBus())
        assert ch._adapter._token == TOKEN and ch._allow == {"42"}
        assert fake.client is None
    finally:
        await stop_plugins(result)


# --- doctor ---------------------------------------------------------------------------------------

def test_doctor_not_configured(tmp_path):
    rows = DispatchPlugin().doctor(_ctx(tmp_path))
    assert [(r.status, r.detail, r.hint) for r in rows] == [
        ("skip", "Discord not configured", NOT_CONFIGURED_HINT)]
    assert NOT_CONFIGURED_HINT == ("localharness plugins enable dispatch --set discord.token=… "
                                   "--set discord.allow=<your user id>")


def test_doctor_half_configured_names_the_missing_key(tmp_path):
    rows = DispatchPlugin().doctor(_ctx(tmp_path, token=TOKEN))
    assert [(r.status, r.detail) for r in rows] == [
        ("skip", "Discord not configured — dispatch.discord.allow is empty")]


def test_doctor_configured(tmp_path):
    rows = DispatchPlugin().doctor(_ctx(tmp_path, token=TOKEN, allow=["42", "43"], channels=["7"]))
    assert [(r.status, r.detail) for r in rows] == [
        ("pass", "Discord configured — 2 allowed user(s); listens in 1 channel(s)")]
    rows = DispatchPlugin().doctor(_ctx(tmp_path, token=TOKEN, allow=["42"]))
    assert rows[0].detail == "Discord configured — 1 allowed user(s); listens in any channel the bot can see"


def test_doctor_warns_per_env_source_without_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_DISCORD_TOKEN", TOKEN)
    monkeypatch.setenv("LOCALHARNESS_DISCORD_ALLOW", "42")
    p = DispatchPlugin()
    rows = p.doctor(_ctx(tmp_path))
    assert [r.status for r in rows] == ["pass", "warn", "warn"]
    assert {r.detail.split()[0] for r in rows[1:]} == {"LOCALHARNESS_DISCORD_TOKEN", "LOCALHARNESS_DISCORD_ALLOW"}
    assert not any(TOKEN in r.detail or TOKEN in r.hint for r in rows)
    assert p.startup_warnings == []  # doctor never starts the plugin


def test_doctor_ignores_the_claude_code_env_file(tmp_path):
    """Claude Code's token file is another program's: doctor reports Discord not configured,
    names the setup command, and neither the file nor the token appears in any row."""
    from pathlib import Path

    env = Path.home() / ".claude" / "channels" / "discord" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text(f"DISCORD_BOT_TOKEN={TOKEN}\n")
    rows = DispatchPlugin().doctor(_ctx(tmp_path, allow=["42"]))
    assert [(r.status, r.detail) for r in rows] == [
        ("skip", "Discord not configured — dispatch.discord.token is empty")]
    assert "localharness plugins enable dispatch" in rows[0].hint
    assert not any(".env" in r.detail or ".env" in r.hint for r in rows)
    assert not any(TOKEN in r.detail or TOKEN in r.hint for r in rows)


async def test_doctor_rows_through_the_lifecycle(tmp_path, monkeypatch):
    """The real doctor_rows: configured, never started, one warn line, no token."""
    monkeypatch.setenv("LOCALHARNESS_DISCORD_ALLOW", "42")
    [row] = await doctor_rows(_resolution(token=TOKEN), paths=_paths(tmp_path))
    assert (row.name, row.state) == ("dispatch", "on")
    assert [c.status for c in row.checks] == ["pass", "warn"]
    assert not any(TOKEN in c.detail for c in row.checks)


# --- make_channel ---------------------------------------------------------------------------------

async def test_make_channel_after_start(tmp_path):
    from localharness.dispatch.adapters.discord import DiscordAdapter
    from localharness.dispatch.channel import DispatchChannel

    p = DispatchPlugin()
    ctx = _ctx(tmp_path, token=TOKEN, allow=["42"], channels=["7", "8"], ack="👀")
    await p.start(ctx)
    bus = EventBus()
    ch = p.make_channel("discord", bus)
    assert isinstance(ch, DispatchChannel) and isinstance(ch._adapter, DiscordAdapter)
    assert ch.bus is bus and ch.channel_id == "discord"
    assert (ch._adapter._token, ch._allow, ch._channels, ch._ack, ch._state_dir) == (
        TOKEN, {"42"}, {"7", "8"}, "👀", ctx.paths.state_dir)
    assert ch.start_banner == "Dispatch mode: Discord — listening for allowlisted messages."


def test_make_channel_before_start_and_unknown_name():
    from localharness.channels.errors import ChannelStartError

    with pytest.raises(ChannelStartError):
        DispatchPlugin().make_channel("discord", EventBus())
    with pytest.raises(ValueError, match="discrod"):
        DispatchPlugin().make_channel("discrod", EventBus())
