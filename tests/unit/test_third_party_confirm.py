"""SEC-11: turning on a plugin you installed, from inside a session, asks one yes/no line first.

An installed plugin is code that runs with your permissions on this machine in every session, so
`/plugins enable <name>` asks once before it writes — right after you typed the command, on the
plain terminal between the session's two halves, never mid-task. No writes nothing and the
conversation continues. Bundled plugins and the shell command `localharness plugins enable` are
unchanged: neither asks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import plugins_cmd
from localharness.cli.app import app
from localharness.plugins import builtin, discovery
from localharness.plugins.api import Plugin, PluginManifest
from tests.unit.test_plugin_resolve import write_folder_plugin

_REAL_DISCOVER = discovery.discover
_CONFIG = {"version": "1", "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                                        "default_model": "test-model"}}


class _Settings(BaseModel):
    color: str = ""


class Shipped(Plugin):
    """ships with LocalHarness"""

    manifest = PluginManifest(name="shipped", version="0.1.0", kind="tools", enabled_by_default=False)
    ConfigModel = _Settings


@pytest.fixture(autouse=True)
def plugins(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Shipped,))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    marks = tmp_path / "sentinels"
    marks.mkdir()
    monkeypatch.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    write_folder_plugin(g, "swatch")  # a plugin the user installed into the plugins/ folder
    return g


@pytest.fixture
def terminal(monkeypatch):
    """A terminal whose person answers the yes/no question with `answer[0]`; every question asked
    is recorded. Nothing else may be asked."""
    asked: list[tuple[str, object]] = []
    answer = [False]

    def confirm(text, default=None, **_kw):
        asked.append((text, default))
        return answer[0]

    def prompt(*_a, **_kw):
        raise AssertionError("a setup question was asked")

    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "confirm", confirm)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    return asked, answer


def _overrides(g: Path):
    path = g / "overrides.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def test_a_third_party_enable_in_session_asks_once_and_no_writes_nothing(g, terminal) -> None:
    asked, _ = terminal

    outcome = plugins_cmd.session_step(("enable", "swatch"), str(g))

    assert asked == [(plugins_cmd.THIRD_PARTY_CONFIRM.format(name="swatch"), False)]
    assert outcome.stopped is True
    assert _overrides(g) is None, "No wrote something"


def test_a_third_party_enable_in_session_writes_after_yes(g, terminal) -> None:
    asked, answer = terminal
    answer[0] = True

    outcome = plugins_cmd.session_step(("enable", "swatch"), str(g))

    assert len(asked) == 1
    assert outcome.stopped is False
    assert _overrides(g) == {"swatch": {"enabled": True}}


def test_the_question_names_the_plugin_and_what_turning_it_on_means() -> None:
    text = plugins_cmd.THIRD_PARTY_CONFIRM.format(name="swatch")
    assert text.startswith("swatch is a plugin you installed")
    assert "your permissions" in text and "every session" in text and text.endswith("?")


def test_a_bundled_plugin_enabled_in_session_is_never_asked(g, terminal) -> None:
    asked, _ = terminal

    outcome = plugins_cmd.session_step(("enable", "shipped"), str(g))

    assert asked == []
    assert outcome.stopped is False
    assert _overrides(g) == {"shipped": {"enabled": True}}


def test_disabling_an_installed_plugin_in_session_is_never_asked(g, terminal) -> None:
    asked, _ = terminal

    plugins_cmd.session_step(("disable", "swatch"), str(g))

    assert asked == []
    assert _overrides(g) == {"swatch": {"enabled": False}}


def test_the_shell_command_is_unchanged(g, terminal) -> None:
    asked, _ = terminal

    result = CliRunner().invoke(app, ["plugins", "enable", "swatch", "--config-dir", str(g)])

    assert result.exit_code == 0, result.output
    assert asked == []
    assert _overrides(g) == {"swatch": {"enabled": True}}
