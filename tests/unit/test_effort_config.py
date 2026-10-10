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
