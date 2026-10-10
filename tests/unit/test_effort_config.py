"""EFFORT-02: the level enum on agent and division files, loader-resolved agent > division."""
from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError

from localharness.config.loader import ConfigLoader
from localharness.config.models import (
    LEVELS,
    ActiveSelection,
    AgentConfig,
    DivisionConfig,
    EffortLevel,
)


def _agent(**kw) -> AgentConfig:
    return AgentConfig(name="a", role="r", **kw)


def test_levels_are_the_four_effort_levels_in_order() -> None:
    assert LEVELS == ("off", "low", "high", "max")
    assert set(LEVELS) == set(get_args(EffortLevel))


def test_agent_effort_defaults_to_inherit_and_accepts_the_levels() -> None:
    assert _agent().effort == "inherit"
    for level in (*LEVELS, "inherit"):
        assert _agent(effort=level).effort == level


def test_agent_effort_rejects_a_level_that_is_not_one() -> None:
    with pytest.raises(ValidationError):
        _agent(effort="medium")


def test_division_effort_is_optional() -> None:
    assert DivisionConfig(name="d").effort is None
    assert DivisionConfig(name="d", effort="max").effort == "max"
    with pytest.raises(ValidationError):
        DivisionConfig(name="d", effort="inherit")


def test_active_selection_effort_defaults_to_none() -> None:
    assert ActiveSelection(base_url="http://x/v1", model="m").effort is None
    assert ActiveSelection(base_url="http://x/v1", model="m", effort="low").effort == "low"


# --- the loader: agent > division, never the org ---------------------------------------------

def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def _load(tmp_path: Path, *, agent: dict, division: dict | None = None, org: dict | None = None):
    config = {
        "version": "1",
        "provider": {"provider_type": "vllm", "base_url": "http://localhost:8000/v1",
                     "default_model": "m"},
    }
    _write(tmp_path / "config.yaml", config)
    if org is not None:
        _write(tmp_path / "org.yaml", {"name": "acme", **org})
    body = {"name": "worker", "role": "Does work", **agent}
    if division is not None:
        _write(tmp_path / "divisions" / "ops.yaml", {"name": "ops", **division})
        body["division"] = "ops"
    _write(tmp_path / "agents" / "worker.yaml", body)
    return ConfigLoader(config_dir=tmp_path).load_agent("worker")


def test_the_agent_file_effort_is_kept(tmp_path: Path) -> None:
    assert _load(tmp_path, agent={"effort": "low"}).effort == "low"


def test_a_division_effort_fills_in_when_the_agent_says_nothing(tmp_path: Path) -> None:
    assert _load(tmp_path, agent={}, division={"effort": "max"}).effort == "max"


def test_no_file_naming_an_effort_leaves_inherit(tmp_path: Path) -> None:
    assert _load(tmp_path, agent={}).effort == "inherit"
    assert _load(tmp_path, agent={}, division={"description": "x"}).effort == "inherit"


def test_inherit_in_the_agent_file_takes_the_division_level(tmp_path: Path) -> None:
    assert _load(tmp_path, agent={"effort": "inherit"}, division={"effort": "off"}).effort == "off"


def test_the_agent_beats_the_division(tmp_path: Path) -> None:
    assert _load(tmp_path, agent={"effort": "low"}, division={"effort": "max"}).effort == "low"


def test_the_loader_never_consults_the_org_default(tmp_path: Path) -> None:
    loaded = _load(tmp_path, agent={}, org={"default_effort": "low"})
    assert loaded.effort == "inherit"


# --- a hand-typed bare `effort: off` (WR-01) ---------------------------------------------------
# YAML 1.1 reads an unquoted off/no as False and on/yes/true as True. `yaml.dump` quotes 'off', so
# the helpers above never feed the loader what a person types; these write the YAML text by hand.

_BARE_OFF_HARNESS = """\
version: "1"
provider:
  provider_type: llamacpp
  base_url: http://localhost:8081/v1
  default_model: m
server:
  runtime: llamacpp
  binary: /x/llama-server
  model: /m/a.gguf
  effort: {value}
extra_endpoints:
  - name: peer
    base_url: http://localhost:9000/v1
    provider_type: llamacpp
    gpu: true
    lifecycle:
      gpu: true
      runtime: llamacpp
      binary: /x/llama-server
      model: /m/b.gguf
      port: 9000
      effort: {value}
active_endpoint:
  name: peer
  base_url: http://localhost:9000/v1
  provider_type: llamacpp
  model: b
  effort: {value}
"""


def _raw(tmp_path: Path, relpath: str, text: str) -> Path:
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_bare_off_in_an_agent_file_is_the_off_level(tmp_path: Path) -> None:
    _raw(tmp_path, "config.yaml", _BARE_OFF_HARNESS.format(value="'off'"))
    _raw(tmp_path, "agents/worker.yaml", "name: worker\nrole: Does work\neffort: off\n")
    assert ConfigLoader(config_dir=tmp_path).load_agent("worker").effort == "off"


def test_a_bare_off_in_a_division_file_is_the_off_level(tmp_path: Path) -> None:
    _raw(tmp_path, "config.yaml", _BARE_OFF_HARNESS.format(value="'off'"))
    _raw(tmp_path, "divisions/ops.yaml", "name: ops\neffort: off\n")
    _raw(tmp_path, "agents/worker.yaml", "name: worker\nrole: Does work\ndivision: ops\n")
    loader = ConfigLoader(config_dir=tmp_path)
    assert loader.load_division("ops").effort == "off"
    assert loader.load_agent("worker").effort == "off"


def test_a_bare_off_in_the_org_default_effort_is_the_off_level(tmp_path: Path) -> None:
    _raw(tmp_path, "config.yaml", _BARE_OFF_HARNESS.format(value="'off'"))
    _raw(tmp_path, "org.yaml", "name: acme\ndefault_effort: off\n")
    assert ConfigLoader(config_dir=tmp_path).load_org().default_effort == "off"


def test_a_bare_off_in_server_lifecycle_and_active_endpoint_is_the_off_level(tmp_path: Path) -> None:
    _raw(tmp_path, "config.yaml", _BARE_OFF_HARNESS.format(value="off"))
    h = ConfigLoader(config_dir=tmp_path).load_harness()
    assert h.server.effort == "off"
    assert h.extra_endpoints[0].lifecycle.effort == "off"
    assert h.active_endpoint.effort == "off"


def test_a_bare_off_in_a_builtin_subagent_overlay_loads(tmp_path: Path) -> None:
    from localharness.agent.subagent import build_web_researcher_config

    _raw(tmp_path, "agents/web-researcher.yaml", "effort: off\n")
    base = build_web_researcher_config("web-researcher")
    out = ConfigLoader(config_dir=tmp_path).overlay_builtin_config("web-researcher", base)
    assert out.effort == "off"


@pytest.mark.parametrize("bare", ["on", "yes", "true"])
def test_a_bare_on_is_refused_and_the_message_names_the_quoting(tmp_path: Path, bare: str) -> None:
    from localharness.config.loader import ConfigValidationError

    _raw(tmp_path, "agents/worker.yaml", f"name: worker\nrole: Does work\neffort: {bare}\n")
    with pytest.raises(ConfigValidationError) as caught:
        ConfigLoader(config_dir=tmp_path).load_agent("worker")
    assert "YAML boolean" in str(caught.value)
    assert "effort: 'off'" in str(caught.value)


def test_a_boolean_effort_is_refused_by_every_effort_field() -> None:
    from localharness.config.models import ManagedServerConfig, OrgConfig

    for build in (
        lambda v: _agent(effort=v),
        lambda v: DivisionConfig(name="d", effort=v),
        lambda v: OrgConfig(name="o", default_effort=v),
        lambda v: ActiveSelection(base_url="http://x/v1", model="m", effort=v),
        lambda v: ManagedServerConfig(binary="/x/vllm", model="m", effort=v),
    ):
        assert build(False).__dict__  # False is the level `off`
        with pytest.raises(ValidationError, match="YAML boolean"):
            build(True)


def test_a_typo_still_gets_the_literal_message() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        _agent(effort="turbo")


def test_the_effort_level_alias_keeps_its_four_arguments() -> None:
    assert get_args(EffortLevel) == ("off", "low", "high", "max")
