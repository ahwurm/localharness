"""50-02: `PluginManifest.sections` — a BUNDLED plugin owns pre-existing top-level core settings.

ON, every row under a claimed section is the plugin's (`plugin=<name>`); OFF, those rows leave the
catalogue while the plugin's `<name>.enabled` switch stays (G1) and the loader still validates the
section. A plugin you install that declares `sections` is refused. `sectp` is a bundled-shaped test
plugin swapped into BUILTIN_PLUGINS; every run passes --config-dir."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import components_cmd, plugins_cmd
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.config.models import HarnessConfig
from localharness.plugins import builtin, discovery
from localharness.plugins.api import Plugin, PluginManifest
from localharness.registry.catalogue import PluginRows, build_catalogue
from tests.unit.test_plugin_plan import build, found, make
from tests.unit.test_plugins_enable_setup import _CONFIG

runner = CliRunner()
# Sectp stands in for the real owner of proposer:/sentinel: (autoresearch, 50), so it is left out
# here: two bundled owners of one section is not what these tests are about.
_REAL_BUILTINS = tuple(p for p in builtin.BUILTIN_PLUGINS if p.manifest.name != "autoresearch")
_CLAIMED = ("proposer.", "sentinel.")
_PROPOSER = {"base_url": "http://127.0.0.1:9/v1", "model": "fake-proposer"}


class Sectp(Plugin):
    """owns two core sections"""

    manifest = PluginManifest(name="sectp", version="0.1.0", kind="dev", sections=("proposer", "sentinel"))


class Onep(Plugin):
    """owns one core section"""

    manifest = PluginManifest(name="onep", version="0.1.0", kind="dev", enabled_by_default=False,
                              sections=("sentinel",))


@pytest.fixture(autouse=True)
def bundled(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Sectp, *_REAL_BUILTINS))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(components_cmd, "console", Console(width=400))


def _home(tmp_path: Path, **extra) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    cfg = dict(_CONFIG, org={"audit_log_path": str(g / "audit.jsonl")}, proposer=_PROPOSER, **extra)
    (g / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return g


def _rows(g: Path) -> dict[str, dict]:
    result = runner.invoke(app, ["components", "list", "--json", "--config-dir", str(g)])
    assert result.exit_code == 0, result.output
    return {r["path"]: r for r in json.loads(result.output)}


def test_on_every_claimed_row_is_the_plugins_and_no_other_core_row_is(tmp_path) -> None:
    rows = _rows(_home(tmp_path))
    claimed = {p for p in rows if p.startswith(_CLAIMED)}
    assert {"proposer.model", "proposer.api_key", "sentinel.saturation_k"} <= claimed
    assert {rows[p]["plugin"] for p in claimed} == {"sectp"}
    assert {p for p, r in rows.items() if r["plugin"] == "sectp"} == claimed | {"sectp.enabled"}
    assert rows["provider.base_url"]["plugin"] is None


def test_off_the_claimed_rows_leave_and_the_switch_stays(tmp_path) -> None:
    g = _home(tmp_path, sectp={"enabled": False})
    rows = _rows(g)
    assert not [p for p in rows if p.startswith(_CLAIMED)]
    assert rows["sectp.enabled"]["current_value"] is False
    # the loader is unchanged: the section still validates while its owner is off
    assert ConfigLoader(config_dir=g).load_harness().proposer.model == "fake-proposer"
    # ...and `components set` refuses a path the catalogue no longer lists (AUTO-02)
    result = runner.invoke(app, ["components", "set", "proposer.model", "x", "--config-dir", str(g)])
    assert (result.exit_code, "Unknown path: 'proposer.model'" in result.output) == (2, True), result.output


def test_no_plugin_rows_means_the_core_catalogue_is_unchanged() -> None:
    cfg = HarnessConfig.model_validate(dict(_CONFIG, proposer=_PROPOSER))
    entries = build_catalogue(cfg)  # the autoresearch callers; test_registry_catalogue.py pins the count
    assert any(p.startswith(_CLAIMED) for p in entries)
    assert all(e.plugin is None for e in entries.values())
    # a plugin that claims nothing adds its own switch and changes no core row
    plain = build_catalogue(cfg, plugins=(PluginRows("plain", True, True),))
    assert {p: e for p, e in plain.items() if p != "plain.enabled"} == entries


def test_a_plugin_you_installed_that_declares_sections_is_refused() -> None:
    plan = build(discovered=(found("grab"),), enabled={"grab": True},
                 imported={"grab": make("grab", sections=("provider",))})
    entry = plan.entry("grab")
    assert entry.state == "refused"
    assert entry.reason == "declares sections (claiming core settings), which only bundled plugins may do"
    assert plan.order == ()
    # the same manifest bundled is fine
    assert build(bundled=(make("own", sections=("sentinel",)),), enabled={"own": True}).entry("own").state == "on"


@pytest.mark.parametrize("cls", (*builtin.BUILTIN_PLUGINS, Sectp, Onep))
def test_every_declared_section_is_a_real_core_setting(cls) -> None:
    assert set(cls.manifest.sections) <= set(HarnessConfig.model_fields)


@pytest.mark.parametrize("name", ["image", "mobile", "memory", "dispatch"])
def test_the_existing_bundled_manifests_claim_nothing(name) -> None:
    assert next(c for c in _REAL_BUILTINS if c.manifest.name == name).manifest.sections == ()


# --------------------------------------------------------------------------- plugins info


def _info(g: Path, name: str, *flags: str):
    result = runner.invoke(app, ["plugins", "info", name, *flags, "--config-dir", str(g)])
    assert result.exit_code == 0, result.output
    return result.output


def test_info_lists_the_claimed_rows_and_discloses_their_names(tmp_path) -> None:
    g = _home(tmp_path)
    out = _info(g, "sectp")
    assert "proposer.model" in out and "sentinel.saturation_k" in out
    assert out.count("proposer: and sentinel: keep their pre-plugin names") == 1
    got = json.loads(_info(g, "sectp", "--json"))
    assert got["sections"] == ["proposer", "sentinel"]
    assert {s["path"] for s in got["settings"]} >= {"sectp.enabled", "proposer.model", "sentinel.saturation_k"}


def test_one_claimed_section_reads_singular_even_while_off(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Onep, *_REAL_BUILTINS))
    out = _info(_home(tmp_path), "onep")
    assert "  sentinel: keeps its pre-plugin name" in out
    assert "sentinel.saturation_k" not in out  # off: the claimed rows are not its listed settings


_INFO_KEYS = {"name", "state", "state_kind", "what_it_does", "from", "enable_command", "version", "kind",
              "requires", "uses", "cli", "slash", "settings", "setup_command", "note"}


@pytest.mark.parametrize("name", ["mobile", "memory"])
def test_a_plugin_that_claims_nothing_gains_only_an_empty_sections_key(tmp_path, name) -> None:
    g = _home(tmp_path)
    assert "pre-plugin name" not in _info(g, name)
    got = json.loads(_info(g, name, "--json"))
    assert set(got) == _INFO_KEYS | {"sections"} and got["sections"] == []
