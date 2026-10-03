"""49-07: Discord driven by `dispatch.discord.*` settings through the REAL `_start_async`, offline.

Criteria (ROADMAP Phase 49):
  1. Discord runs only through the dispatch plugin, configured by settings.
  2. The env variables fall back per field for one release, each with a deprecation line.
  3. Every way `start --channel discord` cannot run is a named refusal, never the terminal.

Per test:
  - test_settings_only_start                        criterion 1: settings alone start it; the gate holds
  - test_workspace_layer_cannot_set_machine_keys     criterion 1: token/allow/channels are global-only
  - test_workspace_layer_cannot_widen_the_allowlist  criterion 1: a project cannot add a user
  - test_workspace_ack_applies                       criterion 1: ack IS layered
  - test_setting_wins_env_fills_the_rest             criterion 2: per-field precedence, one line each
  - test_disabled_dispatch_refuses_discord           criterion 3: tier 2, plugin off (real CLI)
  - test_missing_extra_refuses_with_install_line     criterion 3: tier 2, extra absent (pinned missing)
  - test_plugin_start_failure_never_falls_back       criterion 3: tier 3, start() raised
  - test_typo_is_still_tier_one                      criterion 3: tier 1, before any config is read

Drive: the REAL `OrchestratorREPL.run`, wrapped only to record `UserMessage` on its own bus and to
start a feeder playing the fake gateway (tests.dispatch_support). Stubbed: LLM probe, tokenizer,
plugin discovery, the provider (discard port), `LLMClient.stream_complete`, the `discord` module.
HOME is a tmp dir (`isolate_discord_env`, autouse): the owner's real token file is unreachable.
"""
from __future__ import annotations

import asyncio

import pytest
import typer
import yaml
from typer.testing import CliRunner

from tests.conftest import FakeLLMResponse
from tests.dispatch_support import install_fake_discord, isolate_discord_env
from tests.integration.test_dispatch_start_e2e import BANNER, REPLY, _wait_for
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

SETTINGS = {"token": "tkn-settings-49-07", "allow": ["42"], "channels": ["7"]}
DROPPED = ": only the global config may set it"
TOKEN_MISSING = ("Discord bot token missing — set dispatch.discord.token "
                 "(LOCALHARNESS_DISCORD_TOKEN / DISCORD_BOT_TOKEN still work until 0.17.0)")
ALLOW_EMPTY = ("Discord allowlist empty — set dispatch.discord.allow to your user id(s) "
               "(LOCALHARNESS_DISCORD_ALLOW still works until 0.17.0); refusing to listen to everyone")


_REAL: dict = {}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Env isolation first; keep the REAL OrchestratorREPL.run before any recipe stubs it."""
    from localharness.cli.repl import OrchestratorREPL
    isolate_discord_env(monkeypatch, tmp_path)
    _REAL["run"] = OrchestratorREPL.run


@pytest.fixture
def printed(monkeypatch) -> list[str]:
    return _capture_start_console(monkeypatch)


def _extra(monkeypatch, present: bool = True) -> None:
    """Pin the dispatch extra installed or missing through resolve()'s `extra_installed` default (the
    plan and plugins_cmd._extra_missing both ask it) — never this venv's answer: CI has discord.py."""
    from localharness.plugins import resolve
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed",
                        lambda e: present or e != "dispatch")


def _add_yaml(path, data: dict) -> None:
    """Merge `data` into the YAML file at `path` (created if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(yaml.safe_dump({**(cur or {}), **data}, allow_unicode=True), encoding="utf-8")


def _global(tmp_path, monkeypatch, discord: dict | None = None):
    """The 49-01 recipe: the explicit config dir `tmp_path` is the GLOBAL layer (no workspace)."""
    _stub_start_boundaries(tmp_path, monkeypatch)
    _offline_provider(tmp_path)
    if discord is not None:
        _add_yaml(tmp_path / "config.yaml", {"dispatch": {"discord": discord}})
    return str(tmp_path)


def _workspace(tmp_path, monkeypatch, fake_home, *, global_discord: dict | None, ws_discord: dict):
    """Fake HOME holding the global layer; cwd in a git project whose `.localharness/config.yaml`
    sets `ws_discord`. Returns the config_dir to pass (None: discovery finds the workspace)."""
    from localharness.config.paths import discover_workspace_dir
    from tests.unit.test_workspace_state_landing import _boom, _hermetic

    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, fake_home, home)
    _global(global_dir, monkeypatch, global_discord)
    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    _add_yaml(proj / ".localharness" / "config.yaml", {"dispatch": {"discord": ws_discord}})
    monkeypatch.chdir(proj)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)
    assert discover_workspace_dir() == (proj / ".localharness").resolve()  # premise guard
    return None


async def _drive(monkeypatch, fake, config_dir, *, before=(), hello=(42, 7), quit_from=(42, 7)):
    """Run the real start. The feeder delivers each `before` (author, channel, text) message, then
    `hello` from `hello`'s author/channel, waits for the reply in that channel, then /quit.
    Returns (UserMessages seen, the hello message, the `before` messages)."""
    from localharness.cli.start_cmd import _start_async
    from localharness.core.events import UserMessage

    real_run = _REAL["run"]
    seen: list = []
    ignored: list = []
    hello_msg = fake.message(hello[0], hello[1], "hello")

    async def feeder():
        await _wait_for(lambda: fake.client is not None and "on_message" in fake.client.events,
                        "the channel to register its gateway handlers")
        for author, chan, text in before:
            ignored.append(m := fake.message(author, chan, text))
            await fake.deliver(m)
        await fake.deliver(hello_msg)
        await _wait_for(lambda: ("send", f"c{hello[1]}", REPLY) in fake.log, "the reply to be posted")
        await fake.deliver(fake.message(quit_from[0], quit_from[1], "/quit"))

    async def run(self):
        async def record(event):
            seen.append(event)
        self._bus.subscribe(UserMessage, record)
        feed = asyncio.ensure_future(feeder())
        try:
            await real_run(self)
        finally:
            feed.cancel()

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", run)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)

    async def model(self, messages, tools=None, on_token=None, **_):
        return FakeLLMResponse(content=REPLY), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)
    await asyncio.wait_for(_start_async(None, False, False, config_dir, channel_mode="discord"), 60)
    return seen, hello_msg, ignored


def _summary(printed: list[str]) -> str:
    return next(p for p in printed if "startup)" in p)  # warnings ride the summary line


def _none_acked(fake, msgs) -> None:
    acked = {r[1] for r in fake.log if r[0] == "react"}
    assert not acked & {f"m{m.id}" for m in msgs}, fake.log


def _sends(fake) -> list:
    return [r for r in fake.log if r[0] == "send"]


# --- criterion 1 -----------------------------------------------------------------------------------


async def test_settings_only_start(tmp_path, monkeypatch, printed):
    fake = install_fake_discord(monkeypatch)
    _extra(monkeypatch)
    cfg = _global(tmp_path, monkeypatch, SETTINGS)

    seen, hello, ignored = await _drive(monkeypatch, fake, cfg,
                                        before=[(43, 7, "not allowed"), (42, 8, "wrong channel")])

    assert [(e.channel, e.content) for e in seen] == [("discord", "hello")], seen
    assert ("react", f"m{hello.id}", "✅") in fake.log, fake.log
    assert _sends(fake) == [("send", "c7", REPLY)], fake.log  # nothing to 8, nothing for 43
    _none_acked(fake, ignored)
    assert any(BANNER in p for p in printed), printed
    assert "deprecated" not in _summary(printed), _summary(printed)
    assert not any(SETTINGS["token"] in p for p in printed)


async def test_workspace_layer_cannot_set_machine_keys(tmp_path, monkeypatch, printed, fake_home):
    """All three keys only in the workspace: each is dropped with a warning naming it. With the
    global allow set, the dropped workspace token leaves start refused naming dispatch.discord.token;
    with nothing global, the (earlier) allowlist refusal names dispatch.discord.allow."""
    from localharness.channels.errors import ChannelStartError

    fake = install_fake_discord(monkeypatch)
    _extra(monkeypatch)
    cfg = _workspace(tmp_path, monkeypatch, fake_home, global_discord={"allow": ["42"]},
                     ws_discord={**SETTINGS, "allow": ["43"]})  # restating a global value is no warning
    with pytest.raises(ChannelStartError) as e:
        await _drive(monkeypatch, fake, cfg)
    assert str(e.value) == TOKEN_MISSING
    summary = _summary(printed)
    for key in ("token", "allow", "channels"):
        assert f"ignoring dispatch.discord.{key} in " in summary and DROPPED in summary, summary
    assert SETTINGS["token"] not in summary and SETTINGS["token"] not in str(e.value)
    assert fake.client is None and fake.log == [], "a refused start reached the client"


async def test_workspace_only_settings_refuse_on_the_allowlist(tmp_path, monkeypatch, fake_home):
    from localharness.channels.errors import ChannelStartError

    fake = install_fake_discord(monkeypatch)
    _extra(monkeypatch)
    cfg = _workspace(tmp_path, monkeypatch, fake_home, global_discord=None, ws_discord=SETTINGS)
    with pytest.raises(ChannelStartError) as e:
        await _drive(monkeypatch, fake, cfg)
    assert str(e.value) == ALLOW_EMPTY  # 49-05's order: the core refuses the allowlist before connect
    assert fake.client is None and fake.log == []


async def test_workspace_layer_cannot_widen_the_allowlist(tmp_path, monkeypatch, printed, fake_home):
    fake = install_fake_discord(monkeypatch)
    _extra(monkeypatch)
    cfg = _workspace(tmp_path, monkeypatch, fake_home, global_discord=SETTINGS,
                     ws_discord={"allow": ["43"], "channels": ["8"]})
    seen, _, ignored = await _drive(monkeypatch, fake, cfg,
                                    before=[(43, 7, "workspace-listed user"), (42, 8, "ws channel")])
    assert [e.content for e in seen] == ["hello"], seen  # the global allow and channels stand
    assert _sends(fake) == [("send", "c7", REPLY)], fake.log
    _none_acked(fake, ignored)
    summary = _summary(printed)
    assert "ignoring dispatch.discord.allow in " in summary and "ignoring dispatch.discord.channels in " \
        in summary, summary


async def test_workspace_ack_applies(tmp_path, monkeypatch, printed, fake_home):
    fake = install_fake_discord(monkeypatch)
    _extra(monkeypatch)
    cfg = _workspace(tmp_path, monkeypatch, fake_home, global_discord=SETTINGS, ws_discord={"ack": "👀"})
    seen, hello, ignored = await _drive(monkeypatch, fake, cfg)
    assert ("react", f"m{hello.id}", "👀") in fake.log, fake.log
    assert not any(r[0] == "react" and r[2] == "✅" for r in fake.log), fake.log
    assert "ignoring dispatch.discord.ack" not in _summary(printed)


# --- criterion 2 -----------------------------------------------------------------------------------


async def test_setting_wins_env_fills_the_rest(tmp_path, monkeypatch, printed):
    fake = install_fake_discord(monkeypatch)
    _extra(monkeypatch)
    monkeypatch.setenv("LOCALHARNESS_DISCORD_ALLOW", "99")
    monkeypatch.setenv("LOCALHARNESS_DISCORD_CHANNELS", "7")
    cfg = _global(tmp_path, monkeypatch, {"token": SETTINGS["token"], "allow": ["42"]})

    seen, _, ignored = await _drive(monkeypatch, fake, cfg,
                                    before=[(99, 7, "env-listed user"), (42, 8, "outside env channels")])

    assert [e.content for e in seen] == ["hello"], seen  # allow stays ["42"]; channels = env ["7"]
    assert _sends(fake) == [("send", "c7", REPLY)], fake.log
    _none_acked(fake, ignored)
    summary = _summary(printed)
    assert summary.count("dispatch: LOCALHARNESS_DISCORD_") == 1, summary
    assert ("dispatch: LOCALHARNESS_DISCORD_CHANNELS is deprecated and stops working in 0.17.0 — set "
            "dispatch.discord.channels (localharness components set dispatch.discord.channels …)") \
        in summary, summary


# --- criterion 3 -----------------------------------------------------------------------------------


def _record_plugin_start(monkeypatch) -> list:
    from localharness.dispatch.plugin import DispatchPlugin
    calls: list = []
    real = DispatchPlugin.start

    async def start(self, ctx):
        calls.append(ctx)
        await real(self, ctx)
    monkeypatch.setattr(DispatchPlugin, "start", start)
    return calls


async def test_disabled_dispatch_refuses_discord(tmp_path, monkeypatch):
    from localharness.cli.app import app

    fake = install_fake_discord(monkeypatch)
    _extra(monkeypatch)
    cfg = _global(tmp_path, monkeypatch, SETTINGS)
    off = CliRunner().invoke(app, ["plugins", "disable", "dispatch", "--config-dir", cfg])
    assert off.exit_code == 0, off.output
    starts = _record_plugin_start(monkeypatch)

    with pytest.raises(typer.BadParameter) as e:
        await _drive(monkeypatch, fake, cfg)
    msg = str(e.value)
    assert "channel 'discord' is provided by the dispatch plugin, which is off" in msg, msg
    assert "localharness plugins enable dispatch" in msg, msg
    assert starts == [] and fake.client is None and fake.log == []


async def test_missing_extra_refuses_with_install_line(tmp_path, monkeypatch):
    fake = install_fake_discord(monkeypatch)  # in sys.modules only; the seam says the extra is missing
    _extra(monkeypatch, present=False)
    cfg = _global(tmp_path, monkeypatch, SETTINGS)
    starts = _record_plugin_start(monkeypatch)
    with pytest.raises(typer.BadParameter) as e:
        await _drive(monkeypatch, fake, cfg)
    msg = str(e.value)
    assert "provided by the dispatch plugin, which is missing its install extra" in msg, msg
    assert "localharness[dispatch]" in msg, msg
    assert starts == [] and fake.client is None and SETTINGS["token"] not in msg


async def test_plugin_start_failure_never_falls_back(tmp_path, monkeypatch):
    from localharness.dispatch.plugin import DispatchPlugin

    fake = install_fake_discord(monkeypatch)
    _extra(monkeypatch)
    cfg = _global(tmp_path, monkeypatch, SETTINGS)

    async def boom(self, ctx):
        raise RuntimeError("gateway config exploded")
    monkeypatch.setattr(DispatchPlugin, "start", boom)
    terminals: list = []
    from localharness.channels.terminal import TerminalChannel
    real_init = TerminalChannel.__init__
    monkeypatch.setattr(TerminalChannel, "__init__",
                        lambda self, *a, **k: terminals.append(1) or real_init(self, *a, **k))

    with pytest.raises(typer.BadParameter) as e:
        await _drive(monkeypatch, fake, cfg)
    msg = str(e.value)
    assert msg.startswith("channel 'discord' is provided by the dispatch plugin, which did not start — "), msg
    assert "gateway config exploded" in msg, msg
    assert terminals == [] and fake.client is None


async def test_typo_is_still_tier_one(tmp_path, monkeypatch):
    """Refused before any config is read: the config dir does not exist and resolve() would raise."""
    from localharness.cli.start_cmd import _start_async

    def never(*a, **k):
        raise AssertionError("resolve() ran for a typo")
    monkeypatch.setattr("localharness.plugins.resolve.resolve", never)
    monkeypatch.setattr("localharness.config.loader.ConfigLoader.__init__", never)
    with pytest.raises(typer.BadParameter) as e:
        await _start_async(None, False, False, str(tmp_path / "missing"), channel_mode="discrod")
    msg = str(e.value)
    assert msg.startswith("unknown channel 'discrod'; choose one of: "), msg
    assert "discord" in msg.split("choose one of: ")[1].split(", "), msg
