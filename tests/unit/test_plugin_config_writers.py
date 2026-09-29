"""Every WRITE-time and provenance validator accepts plugin sections (ENAB-01).

Once `plugins enable example` has written `example: {enabled: true}` into overrides.yaml (or
`components set agent.example.size 4` an `agent.example` section), every command that validates
"current config + my change" before writing must still work: `components set`, the security-defaults
migration, `/model` persistence, and autoresearch adoption and provenance. Each check validates the
CORE view; the plugin validates its own section. Unknown keys are still refused where the typo guard
lives — when the config LOADS — and each writer here can only write known paths (catalogue paths or
fixed core keys), so none of them can introduce one.

Each behaviour is driven through the real function, with the existing fixtures reused by import.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from localharness.autoresearch.adoption import AdoptionRefused, _validate_merged
from localharness.autoresearch.experiment import (
    _provenance_agent_cfg as experiment_provenance,
    _resolve_worktree_agent_cfg,
)
from localharness.autoresearch.proposer import _provenance_agent_cfg as proposer_provenance
from localharness.cli.components_cmd import _validate_overlay
from localharness.cli.model_ops import persist_active_endpoint, persist_default_model
from localharness.config import migrate
from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.config.models import AgentConfig, HarnessConfig
from localharness.config.overlay import _resolve_user_overlay_path
from localharness.plugins import discovery
from localharness.plugins.discovery import DiscoveredPlugin
from tests.unit.test_config_migrate import _write_config

_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
        "available_models": ["global-model"],
    },
}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def _example_is_a_known_plugin(monkeypatch):
    """Discovery lists exactly `example`, whatever the venv has installed."""
    monkeypatch.setattr(
        discovery, "discover",
        lambda global_config_dir: [DiscoveredPlugin("example", "entry_point", "ex:P", "ex", "1")],
    )


@pytest.fixture
def global_dir(tmp_path: Path) -> Path:
    g = tmp_path / "global"
    _write_yaml(g / "config.yaml", {**_MINIMAL, "example": {"enabled": True, "color": "#ff0000"}})
    return g


# --------------------------------------------------------------------------- components set


def test_components_validation_accepts_plugin_sections(global_dir) -> None:
    loader = ConfigLoader(config_dir=global_dir)

    _validate_overlay(loader, "example.color", {"example": {"color": "#00ff00"}})
    _validate_overlay(loader, "agent.example.size", {"agent": {"example": {"size": 4}}})


def test_components_validation_still_refuses_an_invalid_core_value(global_dir) -> None:
    loader = ConfigLoader(config_dir=global_dir)

    with pytest.raises(ValidationError):
        _validate_overlay(loader, "org.default_temperature",
                          {"example": {"enabled": True}, "org": {"default_temperature": 9.0}})
    with pytest.raises(ValidationError):
        _validate_overlay(loader, "agent.temperature",
                          {"agent": {"example": {"size": 4}, "temperature": 9.0}})


# --------------------------------------------------------------------------- migrate


def test_the_security_defaults_migration_keeps_a_plugin_section(tmp_path) -> None:
    config_file = _write_config(tmp_path / "cfg", deny=[])
    data = yaml.safe_load(config_file.read_text())
    config_file.write_text(yaml.safe_dump({**data, "example": {"enabled": True}}, sort_keys=False))
    original, migration = migrate.load_plan(config_file)
    assert migration is not None and not migration.config_unchanged, "premise: a real migration"

    migrate.apply(config_file, original, migration)

    written = yaml.safe_load(config_file.read_text())
    assert written["example"] == {"enabled": True}
    assert written["org"]["permissions"]["deny_patterns"], "the migration itself happened"


# --------------------------------------------------------------------------- /model persistence


@pytest.mark.asyncio
async def test_persisting_a_default_model_keeps_a_plugin_section(global_dir) -> None:
    _write_yaml(global_dir / "overrides.yaml", {"example": {"enabled": True}})
    harness = ConfigLoader(config_dir=global_dir).load_harness()

    await persist_default_model(harness, "next-model", config_dir=global_dir)

    overlay = yaml.safe_load((global_dir / "overrides.yaml").read_text())
    assert overlay["example"] == {"enabled": True}
    assert overlay["provider"]["default_model"] == "next-model"


@pytest.mark.asyncio
async def test_persisting_an_active_endpoint_keeps_a_plugin_section(global_dir) -> None:
    _write_yaml(global_dir / "overrides.yaml", {"example": {"enabled": True}})
    harness = ConfigLoader(config_dir=global_dir).load_harness()
    peer = SimpleNamespace(name="peer", base_url="http://peer:8000/v1", provider_type="vllm",
                           api_key="none")

    await persist_active_endpoint(harness, peer, "peer-model", config_dir=global_dir)

    overlay = yaml.safe_load((global_dir / "overrides.yaml").read_text())
    assert overlay["example"] == {"enabled": True}
    assert overlay["active_endpoint"]["model"] == "peer-model"


# --------------------------------------------------------------------------- autoresearch


def test_adoption_validation_accepts_plugin_sections_and_refuses_bad_core_values(global_dir) -> None:
    cfg = ConfigLoader(config_dir=global_dir).load_harness()

    _validate_merged(cfg, "provider.default_model",
                     {"example": {"enabled": True}, "provider": {"default_model": "m2"}})
    _validate_merged(cfg, "agent.model", {"agent": {"example": {"size": 4}, "model": "m2"}})
    with pytest.raises(AdoptionRefused):
        _validate_merged(cfg, "org.default_temperature",
                         {"example": {"enabled": True}, "org": {"default_temperature": 9.0}})


@pytest.mark.parametrize("provenance", [experiment_provenance, proposer_provenance],
                         ids=["experiment", "proposer"])
def test_provenance_reads_an_agent_overlay_that_carries_a_plugin_section(
    components_home, provenance
) -> None:
    assert _resolve_user_overlay_path() == components_home / "overrides.yaml", "premise"
    _write_yaml(components_home / "overrides.yaml",
                {"agent": {"example": {"size": 4}, "temperature": 0.3}})

    cfg = provenance()

    assert isinstance(cfg, AgentConfig) and cfg.temperature == 0.3


def test_a_gate_arm_config_builds_with_a_plugin_section_in_the_agent_overlay(tmp_path) -> None:
    _write_yaml(tmp_path / ".localharness" / "overrides.yaml",
                {"agent": {"example": {"size": 4}, "temperature": 0.3}})
    scenario = SimpleNamespace(
        name="single_read",
        budget=SimpleNamespace(max_actions=5, max_duration_minutes=5.0),
        limits=SimpleNamespace(max_tool_calls=5),
    )

    cfg = _resolve_worktree_agent_cfg(tmp_path, scenario, include_experiment_overlay=False)

    assert isinstance(cfg, AgentConfig) and cfg.temperature == 0.3


# --------------------------------------------------------------------------- the typo guard


def test_the_typo_guard_still_lives_at_load(global_dir) -> None:
    _write_yaml(global_dir / "config.yaml", {**_MINIMAL, "provder": {"base_url": "x"}})

    with pytest.raises(ConfigValidationError, match="provder"):
        ConfigLoader(config_dir=global_dir).load_harness()


def test_the_core_views_keep_exactly_the_models_own_keys() -> None:
    from localharness.config.plugin_sections import core_agent_view, core_harness_view

    data = {"version": "1", "org": {}, "agent": {"model": "m"}, "example": {}, "provder": 1}
    assert core_harness_view(data) == {"version": "1", "org": {}}
    assert set(core_harness_view(data)) <= set(HarnessConfig.model_fields)
    assert core_agent_view({"name": "a", "role": "r", "example": {"size": 4}}) == {
        "name": "a", "role": "r"}
