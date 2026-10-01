"""ENAB-01 / MEMP-06 (gap P8): a BUNDLED plugin's invalid `agent.<name>` section fails `load_agent`
with the same ConfigValidationError shape a core agent key gives (field `<name>.<field>`, its yaml
line) — not a soft plugin failure, which `validate` (it never resolves plugins) would not see. A
discovered plugin's section is never imported or validated here (ENAB-06)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.plugins import builtin
from localharness.plugins.api import Plugin, PluginManifest
from localharness.plugins.discovery import DiscoveredPlugin

CONFIG = ("version: '1'\nprovider:\n  provider_type: ollama\n  base_url: http://localhost:11434/v1\n"
          "  default_model: test-model\n  api_key: none\n")


class _FakeMemAgent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(default=0, ge=0)


class _FakeMem(Plugin):
    """a bundled test plugin with agent-level settings"""

    manifest = PluginManifest(name="fakemem", version="1", kind="tools")
    AgentConfigModel = _FakeMemAgent


@pytest.fixture
def d(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr("localharness.plugins.builtin.BUILTIN_PLUGINS",
                        (*builtin.BUILTIN_PLUGINS, _FakeMem))
    monkeypatch.setattr("localharness.plugins.discovery.discover", lambda _d: [])
    root = tmp_path / "lh"
    (root / "agents").mkdir(parents=True)
    (root / "config.yaml").write_text(CONFIG)
    monkeypatch.setenv("LOCALHARNESS_DIR", str(root))
    return root


def _agent(d: Path, body: str) -> None:
    (d / "agents" / "probe.yaml").write_text("name: probe\nrole: Probe\nmodel: inherit\n" + body)


def test_bundled_agent_section_invalid_value_fails_load(d):
    _agent(d, "fakemem:\n  x: -1\n")  # `x:` is line 5
    with pytest.raises(ConfigValidationError) as ei:
        ConfigLoader(config_dir=d).load_agent("probe")
    [err] = ei.value.errors
    assert (err.field_path, err.yaml_line, err.value) == ("fakemem.x", 5, -1)
    assert ei.value.path == str(d / "agents" / "probe.yaml")


def test_bundled_agent_section_unknown_key_fails_load(d):
    _agent(d, "fakemem:\n  y: 1\n")
    with pytest.raises(ConfigValidationError) as ei:
        ConfigLoader(config_dir=d).load_agent("probe")
    [err] = ei.value.errors
    assert err.field_path == "fakemem.y"


def test_a_plugin_error_joins_a_core_error_in_one_report(d):
    _agent(d, "max_iterations: nope\nfakemem:\n  x: -1\n")
    with pytest.raises(ConfigValidationError) as ei:
        ConfigLoader(config_dir=d).load_agent("probe")
    assert {e.field_path for e in ei.value.errors} == {"max_iterations", "fakemem.x"}


def test_valid_section_still_reaches_agent_plugin_sections(d):
    _agent(d, "fakemem:\n  x: 3\n")
    loader = ConfigLoader(config_dir=d)
    loader.load_agent("probe")
    assert loader.agent_plugin_sections("probe") == {"fakemem": {"x": 3}}


def test_discovered_plugin_sections_are_not_validated_at_load(d, monkeypatch):
    example = DiscoveredPlugin(name="example", source="entry_point",
                               target="localharness_plugin_example.plugin:ExamplePlugin")
    monkeypatch.setattr("localharness.plugins.discovery.discover", lambda _d: [example])
    before = set(sys.modules)
    _agent(d, "example:\n  size: -1\n")  # ExampleAgentConfig would refuse it (ge=1)
    loader = ConfigLoader(config_dir=d)
    loader.load_agent("probe")
    assert loader.agent_plugin_sections("probe") == {"example": {"size": -1}}
    assert not [m for m in set(sys.modules) - before if m.startswith("localharness_plugin_example")]
