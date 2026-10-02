"""49-04: `start --channel <name>` builds ANY bundled channel plugin's channel generically, and every
way it cannot run is a named refusal — never a silent terminal session. Proven with a test-only
channel plugin `chatp` (channel `testchat`) swapped into BUILTIN_PLUGINS; Discord is untouched."""
from __future__ import annotations

import pytest
import typer

from localharness.channels.base import ChannelAdapter
from localharness.plugins import builtin
from localharness.plugins.api import Plugin, PluginManifest
from tests.dispatch_support import isolate_discord_env
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

built: list = []


class _RecChannel(ChannelAdapter):
    start_banner = "Testchat mode: listening."

    def __init__(self, bus, config):
        super().__init__(bus=bus, config=config)
        built.append(self)

    async def read_input(self): return None
    async def send_message(self, content, agent_id=None, **kw): pass
    async def send_error(self, error, detail=None, agent_id=None, **kw): pass
    async def send_streaming(self, *a, **kw): pass
    async def send_tool_call(self, *a, **kw): pass
    async def send_tool_result(self, *a, **kw): pass
    async def start(self): pass
    async def stop(self): pass


class _ChatPlain(Plugin):
    manifest = PluginManifest(name="chatp", version="0.1.0", kind="channel", channels=("testchat",))
    stopped: list = []

    def channels(self):
        return {"testchat": _RecChannel}

    async def stop(self, ctx):
        _ChatPlain.stopped.append(self.manifest.name)


class _ChatPlugin(_ChatPlain):
    made: list = []

    def make_channel(self, name, bus):
        _ChatPlugin.made.append((name, bus))
        return _RecChannel(bus=bus, config={"from": "make_channel"})


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    isolate_discord_env(monkeypatch, tmp_path)
    built.clear()
    _ChatPlugin.made.clear()
    _ChatPlain.stopped.clear()


def _bundle(monkeypatch, *plugins):
    from localharness.tools.builtin.image_plugin import ImagePlugin
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (ImagePlugin, *plugins))


def _repl_channel(monkeypatch):
    seen: list = []

    async def run(self):
        seen.append(self._channel)
    return seen, run


async def _start(tmp_path, monkeypatch, channel="testchat", config_extra=""):
    from localharness.cli.start_cmd import _start_async
    seen, run = _repl_channel(monkeypatch)
    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=run)
    if config_extra:
        with open(tmp_path / "config.yaml", "a") as f:
            f.write(config_extra)
    await _start_async(None, False, False, str(tmp_path), channel_mode=channel)
    return seen, printed


@pytest.mark.asyncio
async def test_make_channel_builds_it_and_the_banner_prints(tmp_path, monkeypatch):
    _bundle(monkeypatch, _ChatPlugin)
    seen, printed = await _start(tmp_path, monkeypatch)
    assert [n for n, _ in _ChatPlugin.made] == ["testchat"]
    assert seen == built and len(built) == 1 and built[0].config == {"from": "make_channel"}
    assert _ChatPlugin.made[0][1] is built[0].bus
    assert "[dim]Testchat mode: listening.[/dim]" in printed


@pytest.mark.asyncio
async def test_without_make_channel_the_class_is_built_with_an_empty_config(tmp_path, monkeypatch):
    _bundle(monkeypatch, _ChatPlain)
    seen, printed = await _start(tmp_path, monkeypatch)
    assert seen == built and len(built) == 1 and built[0].config == {}
    assert "[dim]Testchat mode: listening.[/dim]" in printed


@pytest.mark.asyncio
async def test_plugin_off_is_refused_before_start_plugins(tmp_path, monkeypatch):
    _bundle(monkeypatch, _ChatPlugin)
    calls: list = []

    async def spy(*a, **k):
        calls.append(1)
        raise AssertionError("start_plugins must not run")
    monkeypatch.setattr("localharness.plugins.lifecycle.start_plugins", spy)
    with pytest.raises(typer.BadParameter) as exc:
        await _start(tmp_path, monkeypatch, config_extra="chatp:\n  enabled: false\n")
    msg = str(exc.value)
    assert msg == ("channel 'testchat' is provided by the chatp plugin, which is off — "
                   "run `localharness plugins enable chatp`")
    assert calls == [] and built == []


@pytest.mark.asyncio
async def test_missing_extra_is_refused_with_the_install_line(tmp_path, monkeypatch):
    class _NeedsExtra(_ChatPlugin):
        manifest = _ChatPlugin.manifest.model_copy(update={"requires_extra": "nope"})
    _bundle(monkeypatch, _NeedsExtra)
    with pytest.raises(typer.BadParameter) as exc:
        await _start(tmp_path, monkeypatch)
    assert "provided by the chatp plugin, which is missing its install extra" in str(exc.value)
    assert "localharness[nope]" in str(exc.value)
    assert built == []


@pytest.mark.asyncio
async def test_plugin_failing_in_start_is_refused_and_the_others_are_stopped(tmp_path, monkeypatch):
    from localharness.channels.terminal import TerminalChannel

    class _Boom(_ChatPlugin):
        async def start(self, ctx):
            raise RuntimeError("gateway config broken")

    class _Other(_ChatPlain):  # a second, healthy plugin that did start: it must be stopped
        manifest = PluginManifest(name="otherp", version="0.1.0", kind="tools")

        def channels(self):
            return {}
    _bundle(monkeypatch, _Other, _Boom)
    terminals: list = []
    real = TerminalChannel.__init__
    monkeypatch.setattr(TerminalChannel, "__init__",
                        lambda self, *a, **k: (terminals.append(1), real(self, *a, **k))[1])
    with pytest.raises(typer.BadParameter) as exc:
        await _start(tmp_path, monkeypatch)
    msg = str(exc.value)
    assert msg.startswith("channel 'testchat' is provided by the chatp plugin, which did not start — ")
    assert "gateway config broken" in msg
    assert terminals == [] and built == []
    assert _ChatPlain.stopped == ["otherp"]


@pytest.mark.asyncio
async def test_substrate_failure_is_refused_with_its_text(tmp_path, monkeypatch):
    _bundle(monkeypatch, _ChatPlugin)

    async def broken(*a, **k):
        raise RuntimeError("lifecycle exploded")
    monkeypatch.setattr("localharness.plugins.lifecycle.start_plugins", broken)
    with pytest.raises(typer.BadParameter) as exc:
        await _start(tmp_path, monkeypatch)
    assert str(exc.value) == ("channel 'testchat' is provided by the chatp plugin, which did not "
                              "start — plugins: lifecycle exploded")


@pytest.mark.asyncio
async def test_a_typo_is_still_refused_before_any_plugin_code(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    _bundle(monkeypatch, _ChatPlugin)

    def boom(*a, **k):
        raise AssertionError("resolve() must not run before the channel gate")
    monkeypatch.setattr("localharness.plugins.resolve.resolve", boom)
    with pytest.raises(typer.BadParameter) as exc:
        await _start_async(None, False, False, str(tmp_path), channel_mode="testchta")
    assert "unknown channel 'testchta'; choose one of: acp, discord, terminal, testchat" in str(exc.value)


@pytest.mark.asyncio
async def test_terminal_is_unchanged_with_a_channel_plugin_present(tmp_path, monkeypatch):
    from localharness.channels.terminal import TerminalChannel
    _bundle(monkeypatch, _ChatPlugin)
    seen, printed = await _start(tmp_path, monkeypatch, channel="terminal")
    assert len(seen) == 1 and isinstance(seen[0], TerminalChannel)
    assert built == [] and _ChatPlugin.made == []
    assert not any("Testchat" in line for line in printed)


def test_the_generic_branch_names_no_platform():
    import inspect

    from localharness.cli import start_cmd
    src = inspect.getsource(start_cmd._start_async)
    branch = src[src.index("elif channel_mode in plugin_channel_names() - OWN_COMMAND:"):src.index("        else:\n            # #35")]
    assert "discord" not in branch.lower()
