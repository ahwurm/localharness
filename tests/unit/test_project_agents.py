"""Project agents: one loader, one roster, one writer, and what the model and the person are told."""
from __future__ import annotations

import io
from pathlib import Path
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from localharness.channels.base import one_line
from localharness.config.loader import (
    PROBLEM_LINE_CAP,
    AgentFile,
    AgentProblem,
    ConfigLoader,
    ConfigNotFoundError,
    ConfigParseError,
    Origin,
    ProblemKind,
    problem_lines,
    shadow_label,
)
from localharness.config.models import AgentConfig
from localharness.orchestrator.workflow import AgentCreationWorkflow


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


def test_one_line_flattens_controls_and_whitespace():
    assert one_line("a\n\tb\r\x1b[2Jc") == "a b c"
    assert one_line("x\x9by") == "xy"


def test_a_problem_is_one_flattened_line():
    p = AgentProblem(Path("/p/x\n\x1b[2J.yaml"), "project", "invalid-name", "bad", "fix it")
    assert p.line == "/p/x .yaml: bad; fix it"
    assert "\n" not in AgentProblem(Path("/x.yaml"), "machine", "parse", "a\nb", "f").line


def test_loads_is_true_only_for_a_mismatch():
    assert get_args(ProblemKind) == ("parse", "validation", "invalid-name", "mismatch", "load")
    assert get_args(Origin) == ("project", "machine")
    for kind in get_args(ProblemKind):
        assert AgentProblem(Path("/x.yaml"), "machine", kind, "r", "f").loads == (kind == "mismatch")


def test_problem_lines_caps_with_the_pointer_on_the_overflow_line_only():
    def many(n):
        return [AgentProblem(Path(f"/{i}.yaml"), "project", "load", "r", "f") for i in range(n)]

    assert PROBLEM_LINE_CAP == 20
    exact = problem_lines(many(20), 20)
    assert len(exact) == 20 and not any(line.startswith("...and") for line in exact)
    over = problem_lines(many(21), 20)
    assert len(over) == 21 and over[-1] == "...and 1 more; run localharness agent list"
    assert len(problem_lines(many(25))) == 25
    assert len(problem_lines(many(1), 20)) == 1
    assert problem_lines([]) == [] and problem_lines([], PROBLEM_LINE_CAP) == []


def test_a_markup_lookalike_in_a_problem_prints_literally():
    out = io.StringIO()
    line = AgentProblem(Path("/p/[\x1b/red]x.yaml"), "project", "parse", "1:1: bad", "edit the file").line
    Console(file=out, width=300).print(escape(line))
    assert "[/red]x.yaml" in out.getvalue()


def test_the_name_check_is_a_full_match():
    for ok in ("ab", "a", "a" * 64):
        assert AgentConfig.validate_name_format(ok) == ok
    for bad in ("ab\n", "a" * 65, "", "Ab", "a_b", "é"):
        with pytest.raises(ValueError):
            AgentConfig.validate_name_format(bad)
    with pytest.raises(ValidationError):
        AgentConfig(name="ab\n", role="r")


def test_set_generated_yaml_stores_a_valid_name_or_none(tmp_path):
    flow = AgentCreationWorkflow(config_dir=tmp_path)
    flow.set_generated_yaml("name: ok-name\nrole: r")
    assert flow.gathered["name"] == "ok-name"
    for text in ('name: "ab\\n"\nrole: r', "this is: [[[not: valid", "- a\n- b", "role: r",
                 "name: 123\nrole: r", ""):
        flow.set_generated_yaml("name: ok-name\nrole: r")
        flow.set_generated_yaml(text)
        assert flow.gathered["name"] is None and flow.generated_yaml == text
