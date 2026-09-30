"""Plugin settings are components (ENAB-04): every loaded plugin's `<name>.enabled`, the leaves of
its ConfigModel (`<name>.*`) and of its AgentConfigModel (`agent.<name>.*`), each carrying
`plugin=<name>` and the same layer provenance core rows get — read through the ONE
`layered_catalogue` that `components`, doctor and `config show` share.

`components list/get/set` then treat those rows like core rows: the `(plugin: <name>)` suffix, a
`plugin` JSON field, and `set` writing the GLOBAL overrides.yaml only after the plugin's own model
accepted the value.

`p` is a bundled plugin swapped into BUILTIN_PLUGINS. Folder plugins are 44-09's sentinel writers,
so "never imported" is observed, not assumed. Entry-point discovery is stubbed to none around the
REAL folder scan, so the venv's installed example plugin stays out.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import components_cmd
from localharness.cli.app import app
from localharness.config.models import HarnessConfig
from localharness.plugins import builtin, discovery
from localharness.plugins.api import GLOBAL_ONLY, Plugin, PluginManifest
from localharness.registry.catalogue import PluginRows, build_catalogue
from localharness.registry.provenance import layered_catalogue
from tests.unit.test_plugin_resolve import write_folder_plugin

_REAL_DISCOVER = discovery.discover
_MINIMAL = {
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://localhost:8000/v1",
                 "default_model": "m"},
}


class PConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    color: str = Field("#4a90d9", pattern=r"^#[0-9a-f]{6}$")
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)


class PAgentConfig(BaseModel):
    size: int = Field(8, le=64)


class P(Plugin):
    """draws p swatches"""

    manifest = PluginManifest(name="p", version="0.1.0", kind="tools")
    ConfigModel = PConfig
    AgentConfigModel = PAgentConfig


def _dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def sentinels(tmp_path: Path, monkeypatch):
    """`p` bundled; discovery = the real folder scan only; folder plugins mark this dir on import
    and leave sys.modules after."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (P,))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    marks = tmp_path / "sentinels"
    marks.mkdir()
    monkeypatch.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    yield marks
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    """A global dir holding a minimal valid config.yaml, and an empty workspace `.localharness/`."""
    g, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    _dump(g / "config.yaml", _MINIMAL)
    return g, ws


def _cat(g: Path, ws: Path | None) -> dict:
    return layered_catalogue(g, ws)[0]


def _row(cat: dict, path: str) -> tuple:
    e = cat[path]
    return e.current_value, e.winning_layer


# --------------------------------------------------------------------------- rows


def test_every_leaf_of_a_bundled_plugins_models_is_a_row(layers) -> None:
    cat = _cat(*layers)

    rows = {path: e for path, e in cat.items() if e.plugin is not None}
    assert {path: (e.plugin, e.annotation, e.type_name, e.default_value, e.current_value,
                   e.winning_layer) for path, e in rows.items()} == {
        "p.enabled": ("p", bool, "bool", True, True, "default"),
        "p.color": ("p", str, "str", "#4a90d9", "#4a90d9", "default"),
        "p.url": ("p", str, "str", "", "", "default"),
        "agent.p.size": ("p", int, "int", 8, 8, "default"),
    }


def test_a_bundled_plugin_that_is_off_still_lists_its_settings(layers, monkeypatch) -> None:
    class Q(P):
        manifest = PluginManifest(name="q", version="0.1.0", kind="tools", enabled_by_default=False)

    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (P, Q))
    cat = _cat(*layers)

    assert (cat["q.enabled"].current_value, cat["q.enabled"].default_value) == (False, False)
    assert (cat["q.color"].plugin, cat["q.color"].current_value) == ("q", "#4a90d9")


# --------------------------------------------------------------------------- provenance


def test_a_workspace_value_wins_but_a_global_only_value_is_the_global_one(layers) -> None:
    g, ws = layers
    _dump(g / "config.yaml", {**_MINIMAL, "p": {"url": "http://global.invalid"}})
    _dump(ws / "config.yaml", {"p": {"color": "#00ff00", "url": "http://ws.invalid"}})

    cat = _cat(g, ws)

    assert _row(cat, "p.color") == ("#00ff00", "workspace-config")
    # The workspace url was dropped at load (ENAB-02): crediting the workspace would be a lie.
    assert _row(cat, "p.url") == ("http://global.invalid", "global-config")


def test_build_catalogue_attributes_a_global_only_path_from_the_global_bands_alone() -> None:
    """build_catalogue's own attribution, without layered_catalogue's second pass on top."""
    cfg = HarnessConfig.model_validate(_MINIMAL)
    overlays = {"global-config": {"p": {"url": "g"}},
                "workspace-config": {"p": {"url": "w", "color": "#000000"}}}
    rows = PluginRows(name="p", enabled=True, enabled_default=True, config_model=PConfig,
                      config=PConfig(url="g", color="#000000"), global_only=frozenset({"url"}))

    cat = build_catalogue(cfg, overlays=overlays, plugins=[rows])

    assert _row(cat, "p.url") == ("g", "global-config")
    assert _row(cat, "p.color") == ("#000000", "workspace-config")


def test_turning_on_is_layered_when_bundled_and_machine_level_when_installed(layers, sentinels) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    _dump(g / "overrides.yaml", {"foo": {"enabled": True}})
    _dump(ws / "config.yaml", {"p": {"enabled": False}, "foo": {"enabled": False}})

    cat = _cat(g, ws)

    assert (*_row(cat, "p.enabled"), cat["p.enabled"].default_value) == (False, "workspace-config", True)
    assert (*_row(cat, "foo.enabled"), cat["foo.enabled"].default_value) == (True, "global-overrides", False)
    assert (sentinels / "foo").exists(), "premise: an enabled plugin is imported to list its models"
    assert (cat["foo.color"].plugin, cat["agent.foo.size"].plugin) == ("foo", "foo")


def test_an_agent_level_value_in_the_global_overrides_is_the_current_value(layers) -> None:
    g, ws = layers
    _dump(g / "overrides.yaml", {"agent": {"p": {"size": 3}}})

    assert _row(_cat(g, ws), "agent.p.size") == (3, "global-overrides")


def test_a_plugin_that_is_only_available_has_no_rows_and_is_never_imported(layers, sentinels) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    # Loads only because discovery knows `foo` (an unknown top-level key is refused at load).
    _dump(g / "config.yaml", {**_MINIMAL, "foo": {"color": "#000000"}})

    cat = _cat(g, ws)

    assert [path for path in cat if path.startswith(("foo.", "agent.foo."))] == []
    assert not (sentinels / "foo").exists()


def test_core_rows_are_the_same_with_and_without_a_plugin(layers, monkeypatch) -> None:
    g, ws = layers
    _dump(ws / "config.yaml", {"org": {"log_level": "debug"}, "p": {"color": "#00ff00"}})
    with_p = _cat(g, ws)
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())
    _dump(ws / "config.yaml", {"org": {"log_level": "debug"}})
    without = _cat(g, ws)

    def core(cat: dict) -> list:
        return sorted((path, repr(e.current_value), e.winning_layer, repr(e.default_value),
                       e.type_name) for path, e in cat.items() if e.plugin is None)

    assert core(with_p) == core(without)
    assert _row(with_p, "org.log_level") == ("debug", "workspace-config")
    assert len([e for e in with_p.values() if e.plugin]) == 4
    assert [e for e in without.values() if e.plugin] == []


# --------------------------------------------------------------------------- components list/get/set

runner = CliRunner()


def _cli(*args: str):
    return runner.invoke(app, ["components", *args])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_list_suffixes_each_plugin_rows_layer_and_no_core_row(components_home, monkeypatch) -> None:
    monkeypatch.setattr(components_cmd, "console", Console(width=400))  # no folding: one row, one line

    result = _cli("list")

    assert result.exit_code == 0, result.output
    suffixed = [line for line in result.stdout.splitlines() if "(plugin: " in line]
    assert len(suffixed) == 4 and all("(plugin: p)" in line for line in suffixed)
    assert re.search(r"^│ p\.color +│ str +│ '#4a90d9' +│ default \(plugin: p\) +│$",
                     "\n".join(suffixed), re.MULTILINE)


def test_list_json_carries_the_owning_plugin_and_null_for_core(components_home) -> None:
    result = _cli("list", "--json")

    assert result.exit_code == 0, result.output
    rows = {r["path"]: r for r in json.loads(result.stdout)}
    assert {path: r["plugin"] for path, r in rows.items() if r["plugin"] is not None} == {
        "p.enabled": "p", "p.color": "p", "p.url": "p", "agent.p.size": "p"}
    assert all("plugin" in r for r in rows.values()) and rows["provider.default_model"]["plugin"] is None


def test_get_prints_a_plugin_settings_value_and_layer(components_home) -> None:
    result = _cli("get", "p.color")

    assert result.exit_code == 0, result.output
    assert "p.color = '#4a90d9'" in result.stdout
    assert "layer:   default (plugin: p)" in result.stdout


def test_set_writes_the_global_overrides_and_get_reads_it_back(components_home) -> None:
    config_before = (components_home / "config.yaml").read_bytes()

    result = _cli("set", "p.color", "#00ff00")

    assert result.exit_code == 0, result.output
    assert yaml.safe_load((components_home / "overrides.yaml").read_text()) == {"p": {"color": "#00ff00"}}
    assert (components_home / "config.yaml").read_bytes() == config_before
    got = json.loads(_cli("get", "p.color", "--json").stdout)
    assert (got["value"], got["layer"], got["plugin"]) == ("#00ff00", "global-overrides", "p")


def test_set_refuses_a_value_the_plugins_model_rejects_and_writes_nothing(components_home) -> None:
    overrides = components_home / "overrides.yaml"
    _dump(overrides, {"p": {"color": "#111111"}})
    before, config_before = _sha(overrides), _sha(components_home / "config.yaml")

    result = _cli("set", "p.color", "red")  # a str, so coercion passes: only PConfig's pattern refuses

    assert result.exit_code == 2, result.output
    assert "Validation failed" in result.output and "p.color" in result.output
    assert (_sha(overrides), _sha(components_home / "config.yaml")) == (before, config_before)


def test_set_enabled_false_writes_the_switch(components_home) -> None:
    result = _cli("set", "p.enabled", "false")

    assert result.exit_code == 0, result.output
    assert yaml.safe_load((components_home / "overrides.yaml").read_text()) == {"p": {"enabled": False}}
    got = json.loads(_cli("get", "p.enabled", "--json").stdout)
    assert (got["value"], got["layer"]) == (False, "global-overrides")


def test_set_checks_an_agent_level_setting_with_the_plugins_agent_model(components_home) -> None:
    overrides = components_home / "overrides.yaml"
    assert _cli("set", "agent.p.size", "4").exit_code == 0
    assert yaml.safe_load(overrides.read_text()) == {"agent": {"p": {"size": 4}}}
    before = _sha(overrides)

    result = _cli("set", "agent.p.size", "99")  # an int, so coercion passes: only le=64 refuses

    assert result.exit_code == 2 and "Validation failed" in result.output, result.output
    assert _sha(overrides) == before


def test_a_plugin_validator_that_exits_is_contained_and_nothing_is_written(components_home,
                                                                           monkeypatch) -> None:
    """A plugin's own validator raising SystemExit(0) must not end `set` as a silent success."""
    class ExitsConfig(BaseModel):
        color: str = "#4a90d9"

        @field_validator("color")
        @classmethod
        def _exit(cls, value: str) -> str:
            if value == "#000000":
                raise SystemExit(0)
            return value

    class E(Plugin):
        manifest = PluginManifest(name="e", version="0.1.0", kind="tools")
        ConfigModel = ExitsConfig

    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (E,))

    result = _cli("set", "e.color", "#000000")

    assert result.exit_code == 2, result.output
    assert "Validation failed" in result.output and "SystemExit" in result.output
    assert not (components_home / "overrides.yaml").exists()


def test_set_checks_the_new_value_together_with_the_global_config_section(components_home,
                                                                           monkeypatch) -> None:
    """`host` is required and set only in config.yaml: judged alone, `r: {color: ...}` would fail."""
    class RConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        host: str
        color: str = Field("#4a90d9", pattern=r"^#[0-9a-f]{6}$")

    class R(Plugin):
        manifest = PluginManifest(name="r", version="0.1.0", kind="tools")
        ConfigModel = RConfig

    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (R,))
    config = yaml.safe_load((components_home / "config.yaml").read_text())
    _dump(components_home / "config.yaml", {**config, "r": {"enabled": True, "host": "h"}})

    result = _cli("set", "r.color", "#00ff00")

    assert result.exit_code == 0, result.output
    assert yaml.safe_load((components_home / "overrides.yaml").read_text()) == {"r": {"color": "#00ff00"}}
