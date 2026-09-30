"""`plugins enable NAME` asks a plugin's declared setup questions — on a terminal, and only there.

A plugin declares what to ask as DATA (`PluginManifest.setup`, a tuple of `SetupField`) plus a few
lines of `setup_help`; the harness asks, writes the answers through the same checked, atomic overlay
path `--set` uses, then runs the plugin's own doctor check once and prints it with doctor's own row
printer. Off a terminal (or with --no-input, or --workspace) enable writes the switch only and names
the `--set` spelling as the next step.

`stub` is a bundled plugin swapped into BUILTIN_PLUGINS; every run passes an explicit --config-dir
(no workspace discovery, never the ambient home) except the --workspace case, which uses the fake
home + project layout from test_doctor_layer_report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import plugins_cmd
from localharness.cli.app import app
from localharness.plugins import builtin, discovery
from localharness.plugins.api import GLOBAL_ONLY, Check, Plugin, PluginManifest, SetupField
from tests.unit.test_doctor_layer_report import _layout

runner = CliRunner()
_REAL_DISCOVER = discovery.discover
_CONFIG = {  # port 9 (discard): doctor's endpoint probe fails fast and never reaches a model
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model", "available_models": ["test-model"]},
}


class StubConfig(BaseModel):
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)


class Stub(Plugin):
    """answers stub questions"""

    manifest = PluginManifest(
        name="stub", version="0.1.0", kind="tools", enabled_by_default=False,
        setup=(SetupField(key="url", prompt="Server address", default="http://127.0.0.1:1"),),
        setup_help="HELP-LINE-1\nHELP-LINE-2")
    ConfigModel = StubConfig

    async def configure(self, ctx):
        return "ready" if ctx.config.url else ("unconfigured", "stub.url")

    def doctor(self, ctx):
        url = ctx.config.url
        if url == "http://ok":
            return [Check(name="stub", status="pass", detail=f"answers at {url}")]
        return [Check(name="stub", status="fail", detail=f"no answer at {url}", hint="start it")]


class Plain(Plugin):
    """has nothing to ask"""

    manifest = PluginManifest(name="plain", version="0.1.0", kind="tools", enabled_by_default=False)
    ConfigModel = StubConfig


@pytest.fixture(autouse=True)
def bundled(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Plain))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


@pytest.fixture
def prompts(monkeypatch):
    """A terminal whose person answers `answer[0]`; every prompt is recorded."""
    calls: list[tuple] = []
    answer = ["http://ok"]

    def prompt(text, default=None, **kw):
        calls.append((text, default))
        return answer[0]

    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    return calls, answer


@pytest.fixture
def no_prompt(monkeypatch):
    def prompt(*a, **kw):
        raise AssertionError("prompted")
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)


def _enable(g: Path, *args: str):
    return runner.invoke(app, ["plugins", "enable", *args, "--config-dir", str(g)])


def _overrides(g: Path):
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


def _stub_line(output: str) -> str:
    return next(line for line in output.splitlines() if "stub:" in line)


NEXT = ("next step — give it the Server address: "
        "localharness plugins enable stub --set url=http://127.0.0.1:1")


def test_on_a_terminal_enable_asks_writes_and_checks(g, prompts) -> None:
    calls, _ = prompts
    result = _enable(g, "stub")

    assert result.exit_code == 0, result.output
    assert calls == [("Server address", "http://127.0.0.1:1")]
    assert _overrides(g) == {"stub": {"enabled": True, "url": "http://ok"}}
    assert "Checking it now:" in result.output
    assert "✓ stub: answers at http://ok" in result.output
    assert "HELP-LINE-1" not in result.output


def test_a_failing_check_prints_its_hint_and_the_setup_help_and_still_enables(g, prompts) -> None:
    prompts[1][0] = "http://bad"
    result = _enable(g, "stub")

    assert result.exit_code == 0, result.output
    assert "✗ stub: no answer at http://bad" in result.output
    assert "start it" in result.output
    assert "HELP-LINE-1" in result.output and "HELP-LINE-2" in result.output
    assert _overrides(g) == {"stub": {"enabled": True, "url": "http://bad"}}


def test_the_probe_line_is_the_line_doctor_prints(g, prompts) -> None:
    prompts[1][0] = "http://bad"
    enabled = _enable(g, "stub")
    doctor = runner.invoke(app, ["doctor", "--config-dir", str(g)])

    assert _stub_line(enabled.output) == _stub_line(doctor.output)


def test_without_a_terminal_enable_writes_the_switch_and_names_the_set_spelling(g, no_prompt) -> None:
    result = _enable(g, "stub")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"stub": {"enabled": True}}
    assert NEXT in result.output


def test_no_input_on_a_terminal_is_the_same_as_no_terminal(g, monkeypatch, no_prompt) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "stub", "--no-input")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"stub": {"enabled": True}}
    assert NEXT in result.output


def test_explicit_set_values_win_over_asking(g, monkeypatch, no_prompt) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "stub", "--set", "url=http://ok")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"stub": {"enabled": True, "url": "http://ok"}}
    assert "next step" not in result.output


def test_workspace_never_asks(tmp_path, monkeypatch, fake_home, no_prompt) -> None:
    layout = _layout(tmp_path, monkeypatch, fake_home)
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)

    result = runner.invoke(app, ["plugins", "enable", "stub", "--workspace"])

    assert result.exit_code == 0, result.output
    assert yaml.safe_load((layout.ws_dir / "overrides.yaml").read_text()) == {"stub": {"enabled": True}}


def test_a_plugin_without_setup_fields_is_unchanged(g, monkeypatch, no_prompt) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "plain")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"plain": {"enabled": True}}
    assert "next step" not in result.output and "Checking it now" not in result.output


def test_info_shows_the_set_spelling(g) -> None:
    text = runner.invoke(app, ["plugins", "info", "stub", "--config-dir", str(g)]).output
    assert "set up:" in text
    assert "localharness plugins enable stub --set url=http://127.0.0.1:1" in text

    data = json.loads(runner.invoke(app, ["plugins", "info", "stub", "--json",
                                          "--config-dir", str(g)]).stdout)
    assert data["setup_command"] == "localharness plugins enable stub --set url=http://127.0.0.1:1"
    plain = json.loads(runner.invoke(app, ["plugins", "info", "plain", "--json",
                                           "--config-dir", str(g)]).stdout)
    assert plain["setup_command"] is None


def test_setup_is_optional_and_frozen() -> None:
    m = PluginManifest(name="x", version="1", kind="tools")
    assert m.setup == () and m.setup_help == ""
    field = SetupField(key="url", prompt="Server address")
    assert field.default == ""
    with pytest.raises(ValidationError):
        field.key = "other"
