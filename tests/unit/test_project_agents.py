"""Project agents: one loader, one roster, one writer, and what the model and the person are told."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.config.loader import (
    AgentFile,
    ConfigLoader,
    ConfigNotFoundError,
    ConfigParseError,
    shadow_label,
)


def _write(agents_dir: Path, stem: str, data: dict | str) -> Path:
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{stem}.yaml"
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data), encoding="utf-8")
    return path


def _layers(tmp_path, *, bound: bool = True, trusted: bool = True) -> tuple[ConfigLoader, Path, Path]:
    layer = tmp_path / "proj" / ".localharness"
    loader = ConfigLoader(config_dir=tmp_path / "machine",
                          local_config_dir=layer if bound else None, project_trusted=trusted)
    machine, project = tmp_path / "machine" / "agents", layer / "agents"
    machine.mkdir(parents=True)
    project.mkdir(parents=True)
    return loader, machine, project


def _agent(name: str) -> dict:
    return {"name": name, "role": "r", "model": "inherit"}


def test_a_project_file_is_labelled_project_and_wins_its_stem(tmp_path):
    loader, m, p = _layers(tmp_path)
    for stem in ("alpha", "beta"):
        _write(m, stem, _agent(stem))
    for stem in ("alpha", "gamma"):
        _write(p, stem, _agent(stem))
    assert loader.discover_agent_files() == [
        AgentFile(m / "alpha.yaml", "machine"), AgentFile(m / "beta.yaml", "machine"),
        AgentFile(p / "alpha.yaml", "project"), AgentFile(p / "gamma.yaml", "project")]
    winners = loader.agent_winners()
    assert {s: f.origin for s, f in winners.items()} == {
        "alpha": "project", "beta": "machine", "gamma": "project"}
    assert winners["alpha"].path == p / "alpha.yaml"
    assert loader.agent_file("alpha").origin == "project"
    assert loader.agent_file("beta").origin == "machine"
    assert loader.agent_file("nope") is None


def test_search_paths_and_dirs_come_from_one_place(tmp_path):
    loader, m, p = _layers(tmp_path)
    layer = tmp_path / "proj" / ".localharness"
    assert loader.local_dir == layer
    assert loader.agents_dir("project") == layer / "agents"
    assert loader.agents_dir("machine") == tmp_path / "machine" / "agents"
    paths = loader.agent_search_paths("x")
    assert paths == [str(layer / "agents" / "x.yaml"), str(tmp_path / "machine" / "agents" / "x.yaml")]
    with pytest.raises(ConfigNotFoundError) as e:
        loader.load_agent("x")
    assert e.value.searched_paths == paths

    unbound = ConfigLoader(config_dir=tmp_path / "machine")
    assert unbound.local_dir is None and unbound.agents_dir("project") is None
    assert unbound.agent_search_paths("x") == [str(tmp_path / "machine" / "agents" / "x.yaml")]
    with pytest.raises(ConfigNotFoundError) as e:
        unbound.load_agent("x")
    assert e.value.searched_paths == unbound.agent_search_paths("x")

    custom = ConfigLoader(config_dir=tmp_path / "custom")
    label = shadow_label(custom.agents_dir("machine") / "x.yaml")
    assert label == f"shadows {tmp_path / 'custom' / 'agents' / 'x.yaml'}" and "~" not in label


def test_read_agent_file_is_the_one_read_discover_agents_makes(tmp_path, monkeypatch):
    loader, m, p = _layers(tmp_path)
    nameless = _write(m, "nameless", {"role": "r"})
    assert ConfigLoader.read_agent_file(nameless)["name"] == "nameless"
    broken = _write(m, "broken", "a: [unclosed")
    with pytest.raises(ConfigParseError) as e:
        ConfigLoader.read_agent_file(broken)
    assert e.value.__cause__ is None and "[unclosed" not in str(e.value)

    _write(p, "nameless", _agent("nameless"))
    reads = []
    real = ConfigLoader.read_agent_file
    monkeypatch.setattr(ConfigLoader, "read_agent_file", staticmethod(lambda f: reads.append(f) or real(f)))
    errors = []
    found = loader.discover_agents(on_error=lambda f, exc: errors.append((f, exc)))
    assert reads == loader.agent_yaml_paths()
    assert found == [{"name": "nameless", "role": "r", "model": "inherit"}]
    assert [f for f, _ in errors] == [broken] and isinstance(errors[0][1], ConfigParseError)


def test_a_project_link_leading_outside_is_never_listed(tmp_path):
    loader, m, p = _layers(tmp_path)
    outside = _write(tmp_path, "outside", _agent("x"))
    (p / "x.yaml").symlink_to(outside)
    _write(m, "x", _agent("x"))
    assert all(f.origin == "machine" for f in loader.discover_agent_files())
    assert loader.agent_winners()["x"].origin == "machine"
    assert loader.agent_file("x").origin == "machine"


def test_an_empty_world_enumerates_nothing(tmp_path):
    loader = ConfigLoader(config_dir=tmp_path / "machine", local_config_dir=tmp_path / "proj" / ".localharness")
    assert loader.discover_agent_files() == [] and loader.agent_winners() == {}
    assert loader.agent_file("x") is None
