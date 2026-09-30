"""`localharness plugins list | info | enable [--set k=v …] | disable` (ENAB-03, SAFE-06, PRD §4).

One command shows what is installed, what is on, what is off and why; enabling writes only the
chosen layer's overrides.yaml through the atomic overlay writer — a user's config.yaml is
byte-identical afterwards — and `enable NAME --set k=v` writes the same overlay as `enable NAME`
followed by `components set NAME.k v`. A plugin you installed is turned on only in machine-level
settings (that enable is the operator's trust grant), so --workspace is refused for it.

`p` is a bundled plugin swapped into BUILTIN_PLUGINS. Folder plugins are 44-09's sentinel writers,
so "never imported" is observed. Entry-point discovery is stubbed to none around the REAL folder
scan, so the venv's installed example plugin stays out.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from typer.testing import CliRunner

from localharness import resolved_version
from localharness.cli.app import app
from localharness.plugins import builtin, discovery
from localharness.plugins.api import GLOBAL_ONLY, Plugin, PluginManifest
from tests.unit.test_doctor_layer_report import _layout
from tests.unit.test_plugin_resolve import write_folder_plugin

runner = CliRunner()
_REAL_DISCOVER = discovery.discover
_CONFIG_TEXT = """\
# my machine's settings — this comment and this key order must survive every plugins command
version: '1'
provider:
  provider_type: vllm
  base_url: http://localhost:8000/v1
  default_model: m
"""


class PConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    color: str = Field("#4a90d9", pattern=r"^#[0-9a-f]{6}$")
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)
    retries: int = 3


class PAgentConfig(BaseModel):
    size: int = Field(8, le=64)


class P(Plugin):
    """draws p swatches"""

    manifest = PluginManifest(name="p", version="0.1.0", kind="tools")
    ConfigModel = PConfig
    AgentConfigModel = PAgentConfig


class Q(P):
    """draws q swatches, off until you turn it on"""

    manifest = PluginManifest(name="q", version="0.2.0", kind="tools", enabled_by_default=False)


@pytest.fixture(autouse=True)
def sentinels(tmp_path: Path, monkeypatch):
    """`p` and `q` bundled; discovery = the real folder scan only; folder plugins mark this dir on
    import and leave sys.modules after; wide consoles, so a row is one line."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (P, Q))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setenv("COLUMNS", "400")
    marks = tmp_path / "sentinels"
    marks.mkdir()
    monkeypatch.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    yield marks
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def wide(monkeypatch):
    from localharness.cli import plugins_cmd
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))


def _dir(tmp_path: Path, name: str = "g") -> Path:
    g = tmp_path / name
    g.mkdir()
    (g / "config.yaml").write_text(_CONFIG_TEXT, encoding="utf-8")
    return g


@pytest.fixture
def g(tmp_path: Path) -> Path:
    """Named `[old] g` so every printed path is a markup guard: unescaped, rich deletes `[old]`."""
    return _dir(tmp_path, "[old] g")


def _cli(*args: str):
    return runner.invoke(app, list(args))


def _plugins(g: Path, *args: str):
    return _cli("plugins", *args, "--config-dir", str(g))


def _yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cells(line: str) -> list[str]:
    return re.split(r"\s{2,}", line.strip())


# --------------------------------------------------------------------------- list


def test_list_with_nothing_installed(g, monkeypatch) -> None:
    """QA-13: a PyPI install has no examples/ folder, so the hint says where it looked and gives the
    example's URL."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())

    human, machine = _plugins(g, "list"), _plugins(g, "list", "--json")

    assert human.exit_code == 0, human.output
    assert human.stdout.strip() == (
        f"No plugins found in this Python environment ({sys.prefix}) or in {g / 'plugins'}/. "
        "To write one, copy the example plugin: "
        "https://github.com/ahwurm/localharness/tree/main/examples/plugin-template")
    assert machine.exit_code == 0 and json.loads(machine.stdout) == []


def test_plugins_help_shows_each_summary_as_one_whole_sentence(monkeypatch) -> None:
    """QA-15: typer keeps a docstring's line break inside a command's summary row, so a summary that
    wrapped in the source broke mid-sentence in `plugins --help`. The details are kept, in each
    command's own help."""
    monkeypatch.setenv("COLUMNS", "200")

    lines = _cli("plugins", "--help").stdout.splitlines()
    top = next(i for i, line in enumerate(lines) if "Commands" in line)
    panel = lines[top + 1:next(i for i in range(top, len(lines)) if lines[i].startswith("╰"))]

    assert [_cells(line.strip("│ ")) for line in panel] == [
        ["list", "Every plugin: what it does, whether it is on (and how to turn it on), and where it "
                 "is from."],
        ["info", "One plugin: its state, what it adds, and every setting it owns."],
        ["enable", "Turn a plugin on."],
        ["disable", "Turn a plugin off."],
    ]
    for command, fact in (
            ("info", "The settings are the rows `components list` marks `(plugin: NAME)`."),
            ("enable", "Writes an overrides.yaml, never your config.yaml; takes effect on the next "
                       "`localharness start`."),
            ("disable", "Writes an overrides.yaml, never your config.yaml; takes effect on the next "
                        "`localharness start`.")):
        assert fact in " ".join(_cli("plugins", command, "--help").stdout.split()), command


def test_list_shows_name_what_it_does_state_and_from(g, sentinels) -> None:
    write_folder_plugin(g, "foo")
    write_folder_plugin(g, "future", manifest=', requires_localharness=">=9"')
    (g / "overrides.yaml").write_text("future: {enabled: true}\n", encoding="utf-8")

    result = _plugins(g, "list")

    assert result.exit_code == 0, result.output
    assert [_cells(line) for line in result.stdout.strip().splitlines()] == [
        ["NAME", "WHAT IT DOES", "STATE", "FROM"],
        ["p", "draws p swatches", "on", "built in"],
        ["q", "draws q swatches, off until you turn it on", "off — turn on: localharness plugins enable q",
         "built in"],
        ["foo", "(not loaded)", "available — turn on: localharness plugins enable foo",
         f"folder: {g / 'plugins' / 'foo'}"],
        ["future", "draws future swatches", f"skipped — requires localharness >=9, this is {resolved_version()}",
         f"folder: {g / 'plugins' / 'future'}"],
    ]
    assert not (sentinels / "foo").exists(), "listing an available plugin must not import it"


def test_list_json_rows(g) -> None:
    write_folder_plugin(g, "foo")

    result = _plugins(g, "list", "--json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [
        {"name": "p", "what_it_does": "draws p swatches", "state": "on", "state_kind": "on",
         "from": "built in", "enable_command": None},
        {"name": "q", "what_it_does": "draws q swatches, off until you turn it on",
         "state": "off — turn on: localharness plugins enable q", "state_kind": "off",
         "from": "built in", "enable_command": "localharness plugins enable q"},
        {"name": "foo", "what_it_does": "(not loaded)",
         "state": "available — turn on: localharness plugins enable foo", "state_kind": "available",
         "from": f"folder: {g / 'plugins' / 'foo'}", "enable_command": "localharness plugins enable foo"},
    ]


def test_list_prints_resolver_warnings_on_stderr(g) -> None:
    write_folder_plugin(g, "foo")
    (g / "config.yaml").write_text(_CONFIG_TEXT + "foo: {enabled: 'yes'}\n", encoding="utf-8")

    result = _plugins(g, "list", "--json")

    assert result.exit_code == 0, result.output
    assert "`foo.enabled` must be true or false" in result.stderr
    assert [r["name"] for r in json.loads(result.stdout)] == ["p", "q", "foo"]


# --------------------------------------------------------------------------- info


def test_info_lists_exactly_the_paths_components_list_marks_as_the_plugins(g) -> None:
    info = _plugins(g, "info", "p", "--json")
    rows = json.loads(_cli("components", "list", "--json", "--config-dir", str(g)).stdout)

    assert info.exit_code == 0, info.output
    got = json.loads(info.stdout)
    assert {s["path"] for s in got["settings"]} == {r["path"] for r in rows if r["plugin"] == "p"} == {
        "p.enabled", "p.color", "p.url", "p.retries", "agent.p.size"}
    assert {s["path"] for s in got["settings"] if s["machine_level_only"]} == {"p.url"}
    assert (got["state"], got["what_it_does"], got["from"], got["version"], got["kind"]) == (
        "on", "draws p swatches", "built in", "0.1.0", "tools")


def test_info_prints_type_value_layer_and_machine_level_only(g) -> None:
    (g / "overrides.yaml").write_text("p: {color: '#00ff00'}\n", encoding="utf-8")

    result = _plugins(g, "info", "p")

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: _cells(line) for line in result.stdout.splitlines()
             if line.strip().startswith(("p.", "agent.p."))}
    assert lines == {
        "agent.p.size": ["agent.p.size", "int", "8", "[default]"],
        "p.color": ["p.color", "str", "'#00ff00'", "[global-overrides]"],
        "p.enabled": ["p.enabled", "bool", "True", "[default]"],
        "p.retries": ["p.retries", "int", "3", "[default]"],
        "p.url": ["p.url", "str", "''", "[default] (machine-level only)"],
    }


def test_info_for_a_loaded_plugin_you_installed_marks_enabled_machine_level(g, sentinels) -> None:
    write_folder_plugin(g, "foo")
    (g / "overrides.yaml").write_text("foo: {enabled: true}\n", encoding="utf-8")

    got = json.loads(_plugins(g, "info", "foo", "--json").stdout)

    assert {s["path"]: (s["current_value"], s["layer"], s["machine_level_only"]) for s in got["settings"]} == {
        "foo.enabled": (True, "global-overrides", True),
        "foo.color": ("#4a90d9", "default", False),
        "foo.url": ("http://default.invalid", "default", True),
        "agent.foo.size": (8, "default", False),
    }


def test_info_for_an_available_plugin_shows_only_its_switch_and_imports_nothing(g, sentinels) -> None:
    write_folder_plugin(g, "foo")

    human, machine = _plugins(g, "info", "foo"), _plugins(g, "info", "foo", "--json")

    assert human.exit_code == 0, human.output
    assert ["foo.enabled", "bool", "False", "[default] (machine-level only)"] in [
        _cells(line) for line in human.stdout.splitlines()]
    assert "Its other settings are listed once it is enabled." in human.stdout
    assert f"  from:         folder: {g / 'plugins' / 'foo'}\n" in human.stdout  # `[old] g` escaped
    got = json.loads(machine.stdout)
    assert got["settings"] == [{"path": "foo.enabled", "type": "bool", "current_value": False,
                                "layer": "default", "machine_level_only": True}]
    assert got["state_kind"] == "available" and got["version"] is None
    assert not (sentinels / "foo").exists()


def test_info_names_the_layer_a_switched_off_plugin_you_installed_is_read_from(g) -> None:
    write_folder_plugin(g, "foo")
    assert _plugins(g, "disable", "foo").exit_code == 0

    got = json.loads(_plugins(g, "info", "foo", "--json").stdout)

    assert got["settings"] == [{"path": "foo.enabled", "type": "bool", "current_value": False,
                                "layer": "global-overrides", "machine_level_only": True}]


def test_info_never_credits_a_project_with_turning_on_a_plugin_you_installed(tmp_path, monkeypatch,
                                                                             fake_home) -> None:
    layout = _layout(tmp_path, monkeypatch, fake_home)
    write_folder_plugin(layout.global_dir, "foo")
    (layout.ws_dir / "config.yaml").write_text("foo: {enabled: true}\n", encoding="utf-8")

    got = json.loads(_cli("plugins", "info", "foo", "--json").stdout)

    assert got["state_kind"] == "available"
    assert got["settings"] == [{"path": "foo.enabled", "type": "bool", "current_value": False,
                                "layer": "default", "machine_level_only": True}]


def test_info_for_an_unknown_name_exits_2_naming_plugins_list(g) -> None:
    result = _plugins(g, "info", "nope")

    assert result.exit_code == 2
    assert "nope" in result.stderr and "localharness plugins list" in result.stderr


# --------------------------------------------------------------------------- enable / disable


def test_enable_writes_only_the_global_overrides_and_imports_nothing(g, sentinels) -> None:
    write_folder_plugin(g, "foo")
    before = _sha(g / "config.yaml")

    result = _plugins(g, "enable", "foo")

    assert result.exit_code == 0, result.output
    assert _yaml(g / "overrides.yaml") == {"foo": {"enabled": True}}
    assert _sha(g / "config.yaml") == before
    assert not (sentinels / "foo").exists(), "a plain enable must not import the plugin"
    assert (f"✓ foo enabled in {g / 'overrides.yaml'} — takes effect on the next "
            "`localharness start`") in result.stdout


def test_config_yaml_is_byte_identical_across_enable_set_and_disable(g) -> None:
    write_folder_plugin(g, "foo")
    before = (g / "config.yaml").read_bytes()

    for args in (("enable", "q"), ("enable", "p", "--set", "color=#ff0000"), ("disable", "p"),
                 ("enable", "foo"), ("disable", "foo")):
        result = _plugins(g, *args)
        assert result.exit_code == 0, (args, result.output)
        assert (g / "config.yaml").read_bytes() == before, args
    assert _yaml(g / "overrides.yaml") == {"q": {"enabled": True},
                                           "p": {"enabled": False, "color": "#ff0000"},
                                           "foo": {"enabled": False}}


def test_disable_writes_enabled_false(g) -> None:
    result = _plugins(g, "disable", "p")

    assert result.exit_code == 0, result.output
    assert _yaml(g / "overrides.yaml") == {"p": {"enabled": False}}
    assert f"✓ p disabled in {g / 'overrides.yaml'}" in result.stdout


def test_enable_set_writes_both_keys(g) -> None:
    result = _plugins(g, "enable", "q", "--set", "color=#ff0000")

    assert result.exit_code == 0, result.output
    assert _yaml(g / "overrides.yaml") == {"q": {"enabled": True, "color": "#ff0000"}}


@pytest.mark.parametrize("args, why", [
    (("--set", "color=red"), "String should match pattern"),     # a str: only the model refuses it
    (("--set", "colour=#ff0000"), "colour"),                      # not one of its settings
    (("--set", "color"), "KEY=VALUE"),                            # not a pair
])
def test_enable_set_refuses_and_writes_nothing(g, args, why) -> None:
    (g / "overrides.yaml").write_text("q: {color: '#111111'}\n", encoding="utf-8")
    before = _sha(g / "overrides.yaml")

    result = _plugins(g, "enable", "q", *args)

    assert result.exit_code == 2, result.output
    assert why in result.stderr, result.stderr
    assert _sha(g / "overrides.yaml") == before


def test_enable_set_on_a_plugin_you_installed_imports_it_to_check_the_values(g, sentinels) -> None:
    """The one import a not-yet-enabled plugin gets from this command: --set needs its model, and
    `enable` is the trust grant (SAFE-06)."""
    write_folder_plugin(g, "foo")

    refused = _plugins(g, "enable", "foo", "--set", "color=red")
    assert refused.exit_code == 2 and not (g / "overrides.yaml").exists(), refused.output
    ok = _plugins(g, "enable", "foo", "--set", "color=#00ff00")

    assert ok.exit_code == 0, ok.output
    assert _yaml(g / "overrides.yaml") == {"foo": {"enabled": True, "color": "#00ff00"}}
    assert (sentinels / "foo").exists()


@pytest.mark.parametrize("name, folder", [("q", False), ("foo", True)])
def test_enable_set_is_enable_then_components_set(tmp_path, name, folder) -> None:
    """ENAB-03: `enable NAME --set k=v` is shorthand for `enable NAME` + `components set NAME.k v`."""
    a, b = _dir(tmp_path, "a"), _dir(tmp_path, "b")
    if folder:
        write_folder_plugin(a, name)
        write_folder_plugin(b, name)

    sets = [("color", "#ff0000")] + ([] if folder else [("retries", "5")])  # an int: coercion too

    assert _plugins(a, "enable", name, *[a for k, v in sets for a in ("--set", f"{k}={v}")]).exit_code == 0
    assert _plugins(b, "enable", name).exit_code == 0
    for key, value in sets:
        done = _cli("components", "set", f"{name}.{key}", value, "--config-dir", str(b))
        assert done.exit_code == 0, done.output

    assert _yaml(a / "overrides.yaml") == _yaml(b / "overrides.yaml") == {
        name: {"enabled": True, "color": "#ff0000", **({} if folder else {"retries": 5})}}


def test_enable_set_on_a_plugin_that_cannot_be_imported_writes_nothing(g) -> None:
    write_folder_plugin(g, "boom", prelude="raise RuntimeError('kaput')")

    refused = _plugins(g, "enable", "boom", "--set", "color=#00ff00")

    assert refused.exit_code == 2, refused.output
    assert "boom could not be imported: RuntimeError: kaput" in refused.stderr
    assert not (g / "overrides.yaml").exists()
    # A plain enable runs none of its code, so it cannot fail this way.
    assert _plugins(g, "enable", "boom").exit_code == 0
    assert _yaml(g / "overrides.yaml") == {"boom": {"enabled": True}}


def test_a_name_that_cannot_be_a_plugin_is_refused_and_nothing_is_written(g) -> None:
    """A folder named after a core key: writing `org.enabled` would break the machine's org."""
    write_folder_plugin(g, "org")

    result = _plugins(g, "enable", "org")

    assert result.exit_code == 2, result.output
    assert "core settings key" in result.stderr
    assert not (g / "overrides.yaml").exists()


def test_enable_an_unknown_plugin_exits_2(g) -> None:
    result = _plugins(g, "enable", "nope")

    assert result.exit_code == 2 and "localharness plugins list" in result.stderr
    assert not (g / "overrides.yaml").exists()


# --------------------------------------------------------------------------- --workspace


def test_workspace_writes_the_projects_overrides_for_a_bundled_plugin(tmp_path, monkeypatch,
                                                                      fake_home) -> None:
    layout = _layout(tmp_path, monkeypatch, fake_home)
    config_before = _sha(layout.global_dir / "config.yaml")

    result = _cli("plugins", "enable", "q", "--workspace")

    assert result.exit_code == 0, result.output
    assert _yaml(layout.ws_dir / "overrides.yaml") == {"q": {"enabled": True}}
    assert not (layout.global_dir / "overrides.yaml").exists()
    assert _sha(layout.global_dir / "config.yaml") == config_before
    assert f"✓ q enabled in {layout.ws_dir / 'overrides.yaml'}" in result.stdout


@pytest.mark.parametrize("verb", ["enable", "disable"])
def test_workspace_is_refused_for_a_plugin_you_installed(tmp_path, monkeypatch, fake_home,
                                                        sentinels, verb) -> None:
    layout = _layout(tmp_path, monkeypatch, fake_home)
    write_folder_plugin(layout.global_dir, "foo")

    result = _cli("plugins", verb, "foo", "--workspace")

    assert result.exit_code == 2, result.output
    assert "machine-level" in result.stderr and f"localharness plugins {verb} foo" in result.stderr
    assert not (layout.ws_dir / "overrides.yaml").exists()
    assert not (layout.global_dir / "overrides.yaml").exists()


def test_workspace_refuses_a_machine_level_setting(tmp_path, monkeypatch, fake_home) -> None:
    layout = _layout(tmp_path, monkeypatch, fake_home)

    result = _cli("plugins", "enable", "q", "--workspace", "--set", "url=http://127.0.0.1:1")

    assert result.exit_code == 2, result.output
    assert "q.url" in result.stderr and "machine-level" in result.stderr
    assert not (layout.ws_dir / "overrides.yaml").exists()


def test_a_machine_level_set_is_checked_with_the_machines_layers(tmp_path, monkeypatch,
                                                                  fake_home) -> None:
    """A machine value holds in every project, so this project's own bad value cannot block it."""
    layout = _layout(tmp_path, monkeypatch, fake_home)
    (layout.ws_dir / "config.yaml").write_text("q: {color: not-a-color}\n", encoding="utf-8")

    result = _cli("plugins", "enable", "q", "--set", "color=#00ff00")

    assert result.exit_code == 0, result.output
    assert _yaml(layout.global_dir / "overrides.yaml") == {"q": {"enabled": True, "color": "#00ff00"}}


def test_workspace_with_no_workspace_exits_2(g) -> None:
    result = _plugins(g, "enable", "q", "--workspace")

    assert result.exit_code == 2, result.output
    assert "workspace" in result.stderr
    assert not (g / "overrides.yaml").exists()


def test_a_machine_level_enable_says_when_this_project_still_wins(tmp_path, monkeypatch,
                                                                  fake_home) -> None:
    layout = _layout(tmp_path, monkeypatch, fake_home)
    (layout.ws_dir / "config.yaml").write_text("q: {enabled: false}\n", encoding="utf-8")

    result = _cli("plugins", "enable", "q")

    assert result.exit_code == 0, result.output
    assert _yaml(layout.global_dir / "overrides.yaml") == {"q": {"enabled": True}}
    assert "this project turns q off" in result.stdout
    assert "localharness plugins enable q --workspace" in result.stdout
