"""Project agents: one loader, one roster, one writer, and what the model and the person are told."""
from __future__ import annotations

import io
import os
import stat
from pathlib import Path
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

import localharness.agent.subagent as subagent
from localharness.agent.gate_types import DEFAULT_MODE
from localharness.agent.subagent import _not_wired_text, make_explore_agent_runner
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
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.workflow import AgentCreationWorkflow
from localharness.tools.builtin.agent_tool import BUILTIN_AGENT_NAMES, INVALID_AGENT_NAME, AgentTool
from tests.unit.test_start_cmd import _capture_err_console, _stub_start_boundaries, _write_agent
from tests.unit.test_start_floor import _capture_agent
from tests.unit.test_workspace_state_landing import _drive, _install_recorders, _one, _workspace_start


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


def _text(name: str = "x", **extra) -> str:
    return yaml.safe_dump({"name": name, "role": "r", **extra})


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_write_agent_lands_in_the_bound_project_layer(tmp_path):
    loader, m, p = _layers(tmp_path)
    dest = loader.write_agent("x", _text(), layer="project")
    assert dest == p / "x.yaml" and dest.read_text() == _text() and _mode(dest) == 0o600
    assert not list(p.glob("*.bak")) and list(m.iterdir()) == []


def test_write_agent_lands_in_the_machine_layer_through_a_linked_agents_folder(tmp_path):
    loader = ConfigLoader(config_dir=tmp_path / "machine")
    (tmp_path / "machine").mkdir()
    (tmp_path / "dotfiles").mkdir()
    (tmp_path / "machine" / "agents").symlink_to(tmp_path / "dotfiles")
    dest = loader.write_agent("x", _text(), layer="machine")
    landed = tmp_path / "dotfiles" / "x.yaml"
    assert landed.read_text() == _text() and _mode(landed) == 0o600 and not dest.is_symlink()


def test_write_agent_without_a_layer_creates_nothing(tmp_path):
    loader = ConfigLoader(config_dir=tmp_path / "machine")
    with pytest.raises(ValueError, match="^no project layer in this session"):
        loader.write_agent("x", _text(), layer="project")
    assert list(tmp_path.iterdir()) == []

    layer = tmp_path / "proj" / ".localharness"
    loader = ConfigLoader(config_dir=tmp_path / "machine", local_config_dir=layer)
    with pytest.raises(FileNotFoundError):
        loader.write_agent("x", _text(), layer="project")
    assert not layer.exists()

    layer.mkdir(parents=True)
    loader.write_agent("x", _text(), layer="project")
    assert (layer / "agents" / "x.yaml").is_file()


def test_write_agent_refuses_a_bad_name_or_a_mismatched_name(tmp_path):
    loader, m, p = _layers(tmp_path)
    for bad in ("ab\n", "Ab", "", "a" * 65, "é"):
        with pytest.raises(ValueError):
            loader.write_agent(bad, _text(bad), layer="project")
    for ok in ("a", "a" * 64):
        assert loader.write_agent(ok, _text(ok), layer="project").is_file()
    with pytest.raises(ValueError, match="is not 'x'"):
        loader.write_agent("x", _text("other"), layer="project")
    with pytest.raises(ValueError):
        loader.write_agent("x", "role: r", layer="project")
    assert sorted(f.stem for f in p.iterdir()) == ["a", "a" * 64]


def test_write_agent_refuses_to_overwrite_without_overwrite(tmp_path):
    loader, m, p = _layers(tmp_path)
    loader.write_agent("x", _text(), layer="project")
    with pytest.raises(FileExistsError, match=r"already exists at .*Choose a different name or remove the existing config\.$"):
        loader.write_agent("x", _text(role="new"), layer="project")
    loader.write_agent("x", _text(role="new"), layer="project", overwrite=True)
    assert yaml.safe_load((p / "x.yaml").read_text())["role"] == "new"
    assert not list(p.glob("*.bak"))


@pytest.mark.parametrize("layer", ["project", "machine"])
@pytest.mark.parametrize("dangling", [True, False])
def test_a_planted_tmp_link_is_never_followed(tmp_path, layer, dangling):
    loader, m, p = _layers(tmp_path)
    victim = tmp_path / "victim"
    if not dangling:
        victim.write_text("keep")
    link = (p if layer == "project" else m) / "x.yaml.tmp"
    link.symlink_to(victim)
    loader.write_agent("x", _text(), layer=layer)
    assert link.is_symlink() and (victim.read_text() == "keep" if not dangling else not victim.exists())


@pytest.mark.parametrize("layer", ["project", "machine"])
@pytest.mark.parametrize("dangling", [True, False])
def test_a_symlinked_agent_file_is_refused_in_both_layers(tmp_path, layer, dangling):
    loader, m, p = _layers(tmp_path)
    target = tmp_path / "target.yaml"
    if not dangling:
        target.write_text("keep")
    link = (p if layer == "project" else m) / "x.yaml"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="x.yaml is a symlink; replace it with a file"):
        loader.write_agent("x", _text(), layer=layer, overwrite=True)
    assert link.is_symlink() and (target.read_text() == "keep" if not dangling else not target.exists())


def test_a_symlinked_project_agents_folder_is_refused(tmp_path):
    loader, m, p = _layers(tmp_path)
    p.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    p.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="is a symlink; replace it with a folder"):
        loader.write_agent("x", _text(), layer="project")
    assert list(elsewhere.iterdir()) == []


def test_a_project_link_leading_outside_is_never_written_over(tmp_path):
    loader, m, p = _layers(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_text("keep")
    (p / "x.yaml").symlink_to(outside)
    with pytest.raises(ValueError):
        loader.write_agent("x", _text(), layer="project", overwrite=True)
    assert outside.read_text() == "keep"


def _runner(registry, monkeypatch, seen: list | None = None):
    async def stub(task, *, agent_config, **kw):
        if seen is not None:
            seen.append(agent_config)
        return "done"

    monkeypatch.setattr(subagent, "dispatch_config_subagent", stub)
    return make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(), permission_evaluator=object(),
        get_parent_session_id=lambda: "sid", load_agent=registry.load, available_agents=registry)


async def test_a_project_only_agent_dispatched_through_load_joins_the_description(tmp_path, monkeypatch):
    loader, m, p = _layers(tmp_path)
    _write(p, "proj-only", _agent("proj-only"))
    registry = AgentCardRegistry(loader=loader)
    assert "proj-only" not in list(registry)
    seen: list = []
    runner = _runner(registry, monkeypatch, seen)
    await runner("proj-only", "do it")
    assert [c.name for c in seen] == ["proj-only"]
    assert list(registry) == [*BUILTIN_AGENT_NAMES, "proj-only"]
    assert registry.get("proj-only").origin == "project"
    tool = AgentTool(agent_runner=runner, available_agents=registry,
                     machine_agents_dir=loader.agents_dir("machine"), project_agents_dir=loader.agents_dir("project"))
    assert "proj-only" in tool.info().description


def test_iteration_is_builtins_then_project_then_machine(tmp_path):
    loader, m, p = _layers(tmp_path)
    for stem in ("zeta", "alpha"):
        _write(m, stem, _agent(stem))
    for stem in ("mid", "beta", "explore", "orchestrator"):
        _write(p, stem, _agent(stem))
    registry = AgentCardRegistry(loader=loader)
    for stem in ("zeta", "alpha", "mid", "beta", "explore", "orchestrator"):
        registry.load(stem)
    expected = [*BUILTIN_AGENT_NAMES, "beta", "mid", "alpha", "zeta"]
    assert list(registry) == expected and len(registry) == 8
    assert ", ".join(registry) == ", ".join(expected) and {*registry} == set(expected)
    assert registry.get("explore") and registry.get("orchestrator")
    registry.register_from_config(AgentConfig(name="aaa", role="r"))
    assert list(registry) == [*BUILTIN_AGENT_NAMES, "beta", "mid", "aaa", "alpha", "zeta"]


def test_a_bare_registry_lists_the_builtins_and_loads_nothing():
    registry = AgentCardRegistry()
    assert list(registry) == list(BUILTIN_AGENT_NAMES) and len(registry) == 4
    with pytest.raises(ConfigNotFoundError) as e:
        registry.load("x")
    assert e.value.searched_paths == []


def test_load_checks_the_name_before_any_path(tmp_path, monkeypatch):
    loader, m, p = _layers(tmp_path)
    calls: list = []
    monkeypatch.setattr(loader, "load_agent", lambda *a, **k: calls.append(a))
    registry = AgentCardRegistry(loader=loader)
    for name in ("../x", "ab\n", "", "My_Agent"):
        with pytest.raises(ConfigNotFoundError) as e:
            registry.load(name)
        assert e.value.searched_paths == loader.agent_search_paths(name)
        assert str(e.value.__cause__) == INVALID_AGENT_NAME
    assert calls == []
    monkeypatch.undo()
    with pytest.raises(ConfigNotFoundError) as e:
        registry.load("nope")
    assert e.value.searched_paths == loader.agent_search_paths("nope")


def test_load_keys_by_the_stem_and_always_reregisters(tmp_path):
    loader, m, p = _layers(tmp_path)
    _write(m, "foo", {"name": "bar", "role": "r"})
    registry = AgentCardRegistry(loader=loader)
    assert registry.load("foo").name == "bar"
    assert registry.get("foo").agent_id == "foo" and registry.get("bar") is None
    _write(m, "x", {"name": "x", "role": "m"})
    registry.load("x")
    assert registry.get("x").origin == "machine"
    _write(p, "x", {"name": "x", "role": "p"})
    registry.load("x")
    assert registry.get("x").origin == "project" and registry.get("x").description == "p"


def test_refresh_registers_a_file_written_after_start_and_adds_nothing_twice(tmp_path):
    loader, m, p = _layers(tmp_path)
    _write(m, "a", _agent("a"))
    registry = AgentCardRegistry(loader=loader)
    assert registry.refresh() == [] and "a" in registry
    _write(p, "b", _agent("b"))
    assert registry.refresh() == []
    listed = list(registry)
    assert listed == [*BUILTIN_AGENT_NAMES, "b", "a"]
    assert registry.refresh() == [] and list(registry) == listed
    assert AgentCardRegistry().refresh() == []


def test_a_broken_file_returns_the_same_problem_until_fixed(tmp_path):
    loader, m, p = _layers(tmp_path)
    path = _write(p, "w", {"name": "w", "role": "r", "capabilities": "lots"})
    registry = AgentCardRegistry(loader=loader)
    first, second = registry.refresh(), registry.refresh()
    assert first == second and len(first) == 1
    problem = first[0]
    assert (problem.path, problem.origin, problem.kind, problem.fix) == (path, "project", "validation", "edit the file")
    assert problem.reason.startswith("capabilities: ")
    assert "got:" not in problem.reason and "lots" not in problem.reason
    _write(p, "w", _agent("w"))
    assert registry.refresh() == [] and "w" in list(registry)


def test_a_broken_winner_shadowing_a_machine_file_is_loaded_once(tmp_path, monkeypatch):
    loader, m, p = _layers(tmp_path)
    bad = _write(p, "x", {"name": "x", "role": "r", "capabilities": "lots"})
    _write(m, "x", _agent("x"))
    registry = AgentCardRegistry(loader=loader)
    calls: list[str] = []
    real = registry.load
    monkeypatch.setattr(registry, "load", lambda name: (calls.append(name), real(name))[1])
    problems = registry.refresh()
    assert calls == ["x"]
    assert [pr.path for pr in problems] == [bad]
    assert "x" not in list(registry)


def test_a_name_mismatch_registers_the_stem_and_returns_the_pinned_problem(tmp_path):
    loader, m, p = _layers(tmp_path)
    path = _write(m, "foo", {"name": "bar", "role": "r"})
    registry = AgentCardRegistry(loader=loader)
    assert registry.refresh() == [AgentProblem(
        path, "machine", "mismatch", "name: 'bar' ignored; foo.yaml is agent 'foo'",
        "rename the file or the name: to match")]
    assert "foo" in list(registry) and "bar" not in list(registry)
    assert registry.refresh() == []
    evil = _write(p, "evil", {"name": "../../x", "role": "r"})
    problems = registry.refresh()
    assert [pr.path for pr in problems] == [evil]
    assert registry.get("../../x") is None and registry.get("x") is None and "evil" not in list(registry)
    raw = _write(p, "raw", {"name": "[red]x", "role": "r"})
    (problem,) = [pr for pr in registry.refresh() if pr.path == raw]
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=500).print(escape(problem.line))
    assert "[red]x" in out.getvalue()


def test_problem_kinds_come_from_the_exception(tmp_path, monkeypatch):
    loader, m, p = _layers(tmp_path)
    _write(p, "p", "name: [unclosed\nrole: SECRETTEXT\n")
    _write(p, "design_reviewer", _agent("design-reviewer"))
    _write(p, "refdiv", {"name": "refdiv", "role": "r", "division": "nope"})
    _write(p, "y", _agent("y"))
    _write(p, "boom", _agent("boom"))
    real = loader.load_agent

    def fake(name, **kw):
        if name == "y":
            raise ConfigNotFoundError("y", [])
        if name == "boom":
            raise RuntimeError("boom")
        return real(name, **kw)

    monkeypatch.setattr(loader, "load_agent", fake)
    by_stem = {pr.path.stem: pr for pr in AgentCardRegistry(loader=loader).refresh()}
    assert set(by_stem) == {"p", "design_reviewer", "refdiv", "y"}
    parse = by_stem["p"]
    assert parse.kind == "parse" and "SECRETTEXT" not in parse.reason
    assert parse.reason.split(": ")[0].count(":") == 1
    inv = by_stem["design_reviewer"]
    assert (inv.kind, inv.reason, inv.fix) == ("invalid-name", INVALID_AGENT_NAME, "rename the file")
    ref = by_stem["refdiv"]
    assert (ref.kind, ref.reason, ref.fix) == ("load", "does not load", "run localharness validate")
    assert "nope" not in ref.line
    assert by_stem["y"].kind == "load"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_file_is_no_problem(tmp_path):
    loader, m, p = _layers(tmp_path)
    locked = _write(p, "locked", _agent("locked"))
    locked.chmod(0)
    try:
        registry = AgentCardRegistry(loader=loader)
        assert registry.refresh() == [] and "locked" not in list(registry)
    finally:
        locked.chmod(0o600)


def test_a_multi_line_validation_reason_is_one_line_of_at_most_200(tmp_path):
    loader, m, p = _layers(tmp_path)
    bad = {"name": "many", "role": "r", **{f"field{i}": i for i in range(40)}}
    _write(p, "many", bad)
    (problem,) = AgentCardRegistry(loader=loader).refresh()
    assert problem.kind == "validation"
    assert "\n" not in problem.reason and 0 < len(problem.reason) <= 200


def test_a_partial_builtin_overlay_is_not_a_problem(tmp_path):
    loader, m, p = _layers(tmp_path)
    _write(p, "explore", {"permissions": {"budget": {"max_actions": 50}}})
    registry = AgentCardRegistry(loader=loader)
    assert registry.refresh() == []
    assert list(registry).count("explore") == 1


def test_a_link_leading_outside_is_never_refreshed_in(tmp_path):
    loader, m, p = _layers(tmp_path)
    outside = _write(tmp_path, "outside", _agent("x"))
    (p / "x.yaml").symlink_to(outside)
    registry = AgentCardRegistry(loader=loader)
    assert registry.refresh() == [] and "x" not in list(registry)
    _write(m, "x", _agent("x"))
    assert registry.refresh() == []
    assert registry.get("x").origin == "machine"


def test_not_wired_line_two_says_when_there_is_no_project_layer(tmp_path):
    m, p = tmp_path / "m", tmp_path / "p"
    assert _not_wired_text("zzzz", "zzzz", [str(m / "zzzz.yaml")]).split("\n")[1] == (
        f"searched: {str(m)!r}; no project layer in this session")
    assert _not_wired_text("zzzz", "zzzz", [str(p / "zzzz.yaml"), str(m / "zzzz.yaml")]).split("\n")[1] == (
        f"searched: {str(p)!r}, {str(m)!r}")
    assert _not_wired_text("z", "z", []).split("\n")[1:] == ["searched: none"]


def test_an_unbound_my_agent_gets_the_three_clauses_in_order(tmp_path):
    m = tmp_path / "m"
    lines = _not_wired_text("my_agent", "my-agent", [str(m / "my-agent.yaml")]).split("\n")
    assert len(lines) == 2
    assert lines[1] == (f"searched: {str(m)!r}; no project layer in this session; "
                        "looked for 'my-agent.yaml' (_ becomes -)")
    assert _not_wired_text("my_agent", "my-agent", [], reason="r").split("\n")[2:] == ["r"]


async def test_dispatching_an_invalid_name_prints_the_not_wired_text_and_the_reason(tmp_path, monkeypatch):
    loader, m, p = _layers(tmp_path)
    runner = _runner(AgentCardRegistry(loader=loader), monkeypatch)
    with pytest.raises(ValueError) as e:
        await runner("../x", "t")
    lines = str(e.value).split("\n")
    assert "the closest available to '../x', which is not wired." in lines[0]
    assert lines[-1] == INVALID_AGENT_NAME and "could not be loaded" not in str(e.value)
    with pytest.raises(ValueError) as e:
        await runner("ab\n", "t")
    lines = str(e.value).split("\n")
    assert len(lines) == 3 and "'ab\\n'" in lines[0] and lines[2] == INVALID_AGENT_NAME


async def test_a_sanitized_miss_prints_the_rename_notice_and_no_third_line(tmp_path, monkeypatch):
    loader, m, p = _layers(tmp_path)
    runner = _runner(AgentCardRegistry(loader=loader), monkeypatch)
    with pytest.raises(ValueError) as e:
        await runner("my_agent", "t")
    lines = str(e.value).split("\n")
    assert len(lines) == 2 and lines[1].endswith("; looked for 'my-agent.yaml' (_ becomes -)")
    assert "no project layer" not in lines[1]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
async def test_an_unreadable_project_file_prints_two_lines_and_no_errno(tmp_path, monkeypatch):
    loader, m, p = _layers(tmp_path)
    locked = _write(p, "locked", _agent("locked"))
    locked.chmod(0)
    try:
        runner = _runner(AgentCardRegistry(loader=loader), monkeypatch)
        with pytest.raises(ValueError) as e:
            await runner("locked", "t")
    finally:
        locked.chmod(0o600)
    text = str(e.value)
    assert len(text.split("\n")) == 2
    assert "Errno" not in text and "Permission denied" not in text and str(locked) not in text


@pytest.mark.parametrize("trusted", [True, False])
async def test_a_live_dispatched_project_agent_keeps_the_narrowing(tmp_path, monkeypatch, trusted):
    loader, m, p = _layers(tmp_path, trusted=trusted)
    server = {"name": "evil", "transport": "stdio", "command": "/bin/echo", "args": ["pwned"]}
    _write(p, "worker", {"name": "worker", "role": "r", "tools": {"mcp_servers": [server]},
                         "permissions": {"mode": "unattended"}})
    seen: list = []
    await _runner(AgentCardRegistry(loader=loader), monkeypatch, seen)("worker", "t")
    (cfg,) = seen
    if trusted:
        assert [s.name for s in cfg.tools.mcp_servers] == ["evil"]
    else:
        assert cfg.tools.mcp_servers == [] and cfg.permissions.mode == DEFAULT_MODE


async def test_a_bound_start_advertises_project_and_machine_agents_from_the_registry(tmp_path, monkeypatch, fake_home):
    home, global_dir, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    rec = _install_recorders(monkeypatch)
    agents = _capture_agent(monkeypatch)
    _write_agent(ws / "agents", "orchestrator")
    _write_agent(ws / "agents", "proj-helper")
    _write_agent(global_dir / "agents", "mach-helper")

    await _drive()

    kw = _one(rec, "runner")
    registry = kw["available_agents"]
    assert isinstance(registry, AgentCardRegistry) and registry.loader is not None
    assert kw["load_agent"] == registry.load
    assert list(registry) == [*BUILTIN_AGENT_NAMES, "proj-helper", "solo", "mach-helper"]
    assert _one(rec, "repl")["on_agent_deployed"] == registry.load
    (loop,) = agents
    tool = loop._tools._tools["global"]["agent"]
    assert tool._available_agents is registry
    assert Path(tool._project_agents_dir).resolve() == (ws / "agents").resolve()
    assert Path(tool._machine_agents_dir).resolve() == (global_dir / "agents").resolve()
    text = tool.info().description
    assert (f"A project specialist goes at {tool._project_agents_dir}/<name>.yaml and is read by "
            "every session started in this project") in text
    assert f"a machine specialist goes at {tool._machine_agents_dir}/<name>.yaml and serves every project" in text


async def test_an_explicit_config_dir_start_renders_that_dir_and_no_project_layer(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async

    _stub_start_boundaries(tmp_path, monkeypatch)
    agents = _capture_agent(monkeypatch)
    _write_agent(tmp_path / "agents", "orchestrator")
    _write_agent(tmp_path / "agents", "helper")

    await _start_async(None, False, False, str(tmp_path))

    (loop,) = agents
    tool = loop._tools._tools["global"]["agent"]
    assert Path(tool._machine_agents_dir).resolve() == (tmp_path / "agents").resolve()
    text = tool.info().description
    assert f"There is no project layer in this session, so write it at {tool._machine_agents_dir}/<name>.yaml" in text
    assert "Available agents: explore, web-researcher, cruncher, search-verifier, helper." in text
    assert "~/.localharness" not in text
