"""49-02: the additive plugin-API seams the dispatch plugin needs, landed before it exists.

Manifest-declared channel names, the enablement-aware accepted set, a doctor `warn` level that is
shown but never counted as a failure, `SetupField.secret`, and PLUGIN_API_VERSION still "1".
"""
from __future__ import annotations

import io

import pytest
import yaml
from rich.console import Console

from localharness.cli import doctor_cmd
from localharness.config.loader import ConfigLoader
from localharness.plugins import builtin, discovery
from localharness.plugins.api import PLUGIN_API_VERSION, Check, Plugin, PluginManifest, SetupField
from localharness.plugins.channels import accepted_channels, plugin_channel_names
from localharness.plugins.lifecycle import DoctorRow
from localharness.plugins.resolve import resolve
from localharness.cli.web_plugin import WebPlugin


class _MultiChannel(Plugin):
    manifest = PluginManifest(name="multichat", version="0.1.0", kind="channel",
                              channels=("alpha", "beta"))


def _global(tmp_path, extra: dict | None = None):
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump({"version": "1", "provider": {
        "provider_type": "vllm", "base_url": "http://localhost:8000/v1",
        "default_model": "m"}, **(extra or {})}), encoding="utf-8")
    return g


@pytest.fixture
def multichat(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (WebPlugin, _MultiChannel))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])


def test_plugin_api_version_stays_1():
    assert PLUGIN_API_VERSION == "1"


def test_only_the_dispatch_manifest_declares_channels():
    # dispatch (49) is bundled and on by default; every other bundled manifest declares none
    assert {c.manifest.name: c.manifest.channels for c in builtin.BUILTIN_PLUGINS} == {
        "image": (), "web": (), "memory": (), "dispatch": ("discord",), "autoresearch": ()}


def test_manifest_channels_replace_the_plugin_name(multichat):
    names = plugin_channel_names()
    assert {"alpha", "beta", "web"} == names
    assert "multichat" not in names


def test_web_still_contributes_exactly_web(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (WebPlugin,))
    assert plugin_channel_names() == {"web"}


def test_accepted_channels_follows_the_plan(tmp_path, multichat):
    on = accepted_channels(resolve(ConfigLoader(config_dir=_global(tmp_path))))
    assert {"alpha", "beta", "terminal", "acp"} <= on


def test_accepted_channels_drops_a_channel_plugin_turned_off(tmp_path, multichat):
    g = _global(tmp_path, {"multichat": {"enabled": False}})
    res = resolve(ConfigLoader(config_dir=g))
    assert {e.name: e.state for e in res.plan.entries}["multichat"] == "off"
    acc = accepted_channels(res)
    assert not {"alpha", "beta"} & acc
    assert {"terminal", "acp"} <= acc


def test_a_warn_check_is_shown_and_never_a_failure(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(doctor_cmd, "console", Console(file=buf, width=200, color_system=None))
    failures: list[str] = []
    row = DoctorRow(name="p", state="on", detail="",
                    checks=(Check(name="x", status="warn", detail="d", hint="h"),))
    doctor_cmd.print_plugin_row(row, failures)
    out = buf.getvalue().splitlines()
    assert out[0] == "⚠ x: d"
    assert out[1].strip() == "h"
    assert failures == []


def test_setup_field_secret_is_optional():
    assert SetupField(key="k", prompt="p").secret is False
    assert SetupField(key="k", prompt="p", secret=True).secret is True


# ------------------------------------------------------------------ channel class attributes

def test_bare_mode_command_is_set_by_the_channel_class():
    from localharness.channels.base import ChannelAdapter
    from localharness.dispatch.channel import DispatchChannel
    from localharness.channels.terminal import TerminalChannel

    assert ChannelAdapter.bare_mode_command is False
    assert ChannelAdapter.start_banner == ""
    assert DispatchChannel.bare_mode_command is True
    assert TerminalChannel.bare_mode_command is False


def test_web_and_acp_channels_inherit_no_bare_mode_command():
    from localharness.channels.acp import AcpChannel
    from localharness.channels.web.channel import WebChannel

    assert AcpChannel.bare_mode_command is False and WebChannel.bare_mode_command is False


class _FixtureChat:
    """A channel the REPL has never heard of: only its class attribute makes `mode` a command."""
    channel_id = "fixturechat"
    bare_mode_command = True

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_message(self, text, agent_id=None, metadata=None) -> None:
        self.sent.append(text)


@pytest.mark.asyncio
async def test_a_new_platform_gets_bare_mode_from_its_attribute(tmp_path):
    from tests.unit.test_channel_permission_asks import _gate, _repl

    channel, gate = _FixtureChat(), _gate(tmp_path)
    assert await _repl(channel, gate)._dispatch_input("mode trusted") is None
    assert gate.mode == "trusted"


@pytest.mark.asyncio
async def test_without_the_attribute_the_bare_word_is_a_message(tmp_path):
    import types

    from tests.unit.test_channel_permission_asks import _gate, _repl

    channel, gate = _FixtureChat(), _gate(tmp_path)
    channel.bare_mode_command = False
    repl = _repl(channel, gate)
    started: list[str] = []

    async def _turn(task, on_token=None):
        started.append(task)
        return "done"

    repl._agent = types.SimpleNamespace(
        _config=types.SimpleNamespace(name="a"), current_session_id="s", run_turn=_turn)
    repl._detect_creation_intent = lambda _text: False  # type: ignore[method-assign]
    turn = await repl._dispatch_input("mode trusted")
    if turn is not None:
        await turn
    assert gate.mode == "guarded"
    assert started == ["mode trusted"]
