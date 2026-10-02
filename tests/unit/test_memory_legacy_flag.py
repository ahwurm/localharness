"""MEMP-06: the deprecated `org.memory_enabled` still decides `memory.enabled`, per layer, in the one
resolver every surface shares — with ONE deprecation line naming each file whose legacy key decides
(or would decide) the outcome. A bare `true` that changes nothing is silent: older `init` wrote
`org.memory_enabled: true` into every config.yaml it created (RESEARCH P6)."""
from __future__ import annotations

from pathlib import Path

from localharness.config.loader import ConfigLoader
from localharness.plugins.resolve import fold_legacy_memory_flag, legacy_memory_warning, resolve

FILES = ("/g/config.yaml", "/g/overrides.yaml", "/ws/config.yaml", "/ws/overrides.yaml")
NONE4 = (None, None, None, None)
LINE = "org.memory_enabled is deprecated — use memory.enabled (read from {})"


def test_fold_a_lone_legacy_false_becomes_that_layers_enabled():
    assert fold_legacy_memory_flag(NONE4, (None, None, False, None)) == (
        None, None, {"enabled": False}, None)


def test_fold_memory_enabled_wins_in_the_same_layer():
    layers = (None, None, {"enabled": True}, None)
    assert fold_legacy_memory_flag(layers, (None, None, False, None)) == layers


def test_fold_keeps_each_layer_so_the_highest_wins():
    assert fold_legacy_memory_flag((None, None, {"enabled": True}, None), (False, None, None, None)) == (
        {"enabled": False}, None, {"enabled": True}, None)


def test_fold_leaves_a_non_mapping_layer_for_flag_to_report():
    assert fold_legacy_memory_flag(("on", None, None, None), (False, None, None, None)) == (
        "on", None, None, None)


def test_fold_adds_enabled_beside_other_keys():
    assert fold_legacy_memory_flag((None, {"x": 1}, None, None), (None, False, None, None)) == (
        None, {"x": 1, "enabled": False}, None, None)


def test_warning_names_the_one_deciding_file():
    assert legacy_memory_warning(NONE4, (None, None, False, None), FILES) == LINE.format(FILES[2])


def test_warning_is_one_line_naming_every_false_file_in_layer_order():
    assert legacy_memory_warning(NONE4, (False, None, False, None), FILES) == LINE.format(
        f"{FILES[0]}, {FILES[2]}")


def test_a_bare_true_that_changes_nothing_is_silent():
    assert legacy_memory_warning(NONE4, (True, None, None, None), FILES) is None


def test_a_true_overriding_a_false_decides_so_it_is_named():
    assert legacy_memory_warning(NONE4, (False, None, True, None), FILES) == LINE.format(
        f"{FILES[0]}, {FILES[2]}")


def test_a_legacy_key_shadowed_by_memory_enabled_in_its_layer_is_not_named():
    assert legacy_memory_warning((None, None, {"enabled": True}, None), (None, None, False, None),
                                 FILES) is None


def test_loader_reads_the_four_raw_legacy_values(tmp_path: Path):
    g, ws = tmp_path / "g", tmp_path / "ws" / ".localharness"
    g.mkdir()
    ws.mkdir(parents=True)
    (g / "config.yaml").write_text(
        "version: '1'\nprovider:\n  provider_type: ollama\n  base_url: http://localhost:11434/v1\n"
        "  default_model: m\n  api_key: none\norg:\n  memory_enabled: false\n")
    (ws / "config.yaml").write_text("version: '1'\n")
    assert ConfigLoader(config_dir=g, local_config_dir=ws).legacy_org_flags() == (
        False, None, None, None)


def test_resolve_folds_the_legacy_flag_now_memory_is_a_plugin(tmp_path: Path, monkeypatch):
    """Post-cut (47-06): `memory` is a bundled plugin, so its layer exists and the fold is live —
    the legacy key turns the real plugin off, with the deprecation line exactly once. (Pre-cut this
    asserted the fold was inert.) The composed session proof is 47-08's."""
    monkeypatch.setattr("localharness.plugins.discovery.discover", lambda _d: [])
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(
        "version: '1'\nprovider:\n  provider_type: ollama\n  base_url: http://localhost:11434/v1\n"
        "  default_model: m\n  api_key: none\norg:\n  memory_enabled: false\n")
    result = resolve(ConfigLoader(config_dir=g))
    assert len([w for w in result.warnings if "org.memory_enabled" in w]) == 1
    assert result.enabled["memory"] is False


def test_resolve_wires_the_fold_and_the_line_once_a_memory_layer_exists(tmp_path: Path, monkeypatch):
    """The branch inside resolve() bites as soon as plugin_layers() lists `memory` (47-06): the
    line reaches Resolution.warnings exactly once."""
    monkeypatch.setattr("localharness.plugins.discovery.discover", lambda _d: [])
    real = ConfigLoader.plugin_layers
    monkeypatch.setattr(ConfigLoader, "plugin_layers",
                        lambda self: {**real(self), "memory": (None, None, None, None)})
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(
        "version: '1'\nprovider:\n  provider_type: ollama\n  base_url: http://localhost:11434/v1\n"
        "  default_model: m\n  api_key: none\norg:\n  memory_enabled: false\n")
    lines = [w for w in resolve(ConfigLoader(config_dir=g)).warnings if "org.memory_enabled" in w]
    assert lines == [LINE.format(g / "config.yaml")]
