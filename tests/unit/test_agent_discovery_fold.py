"""Phase 38 (#150): ConfigLoader owns the ONE agent-discovery implementation.

Three independent implementations existed (ConfigLoader.list_agents, agent_cmd._discover_agents,
start_cmd._discover_agents_for_start) with THREE different malformed-YAML behaviors. These tests
pin the folded API: global-dir-first ordering (so the local layer wins by stem), warn-and-skip as
the one error behavior, and element-for-element parity with the start_cmd helper the swap in plan
38-05 will delete.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader

BROKEN_YAML = "a: [unclosed"


def _write_agent(agents_dir: Path, name: str, role: str = "Test role") -> Path:
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{name}.yaml"
    path.write_text(yaml.dump({"name": name, "role": role, "model": "inherit"}), encoding="utf-8")
    return path


@pytest.fixture
def layered(tmp_path, monkeypatch):
    """A global config dir + a CWD-local `.localharness/` project dir.

    `ConfigLoader._local_dir` defaults to the RELATIVE `.localharness`, so the local layer is
    whatever the process CWD says it is — chdir is part of the contract under test.
    """
    global_dir = tmp_path / "global"
    project = tmp_path / "project"
    (project / ".localharness" / "agents").mkdir(parents=True)
    monkeypatch.chdir(project)
    return global_dir, project


def test_agent_yaml_paths_orders_global_then_local(layered):
    """Global dir first, local second, each sorted — the ordering that makes local win by stem."""
    global_dir, project = layered
    _write_agent(global_dir / "agents", "beta")
    _write_agent(global_dir / "agents", "alpha")
    _write_agent(project / ".localharness" / "agents", "alpha", role="local role")

    paths = ConfigLoader(config_dir=global_dir).agent_yaml_paths()

    assert [str(p) for p in paths] == [
        str(global_dir / "agents" / "alpha.yaml"),
        str(global_dir / "agents" / "beta.yaml"),
        str(Path(".localharness") / "agents" / "alpha.yaml"),
    ]


def test_agent_yaml_paths_skips_missing_dirs(tmp_path, monkeypatch):
    """Neither layer existing is an empty list, never an exception."""
    monkeypatch.chdir(tmp_path)
    assert ConfigLoader(config_dir=tmp_path / "nope").agent_yaml_paths() == []


def test_discover_agents_parity_with_start_cmd_helper(layered):
    """The folded API returns EXACTLY what start_cmd's helper returns today.

    This is the whole point of the wave: plan 38-05 swaps the call sites onto ConfigLoader, and
    this assertion is the proof that the swap is behavior-preserving. The old helper still exists
    while this test runs.
    """
    from localharness.cli.start_cmd import _discover_agents_for_start

    global_dir, project = layered
    _write_agent(global_dir / "agents", "alpha")
    _write_agent(global_dir / "agents", "beta", role="global beta")
    _write_agent(project / ".localharness" / "agents", "beta", role="local beta")

    assert ConfigLoader(config_dir=global_dir).discover_agents() == _discover_agents_for_start(global_dir)


def test_discover_agents_local_wins_wholesale(layered):
    """Same stem in both layers: the LOCAL file replaces the global one whole-file (no merge)."""
    global_dir, project = layered
    _write_agent(global_dir / "agents", "shared", role="global role")
    _write_agent(project / ".localharness" / "agents", "shared", role="local role")

    agents = ConfigLoader(config_dir=global_dir).discover_agents()

    assert len(agents) == 1
    assert agents[0]["role"] == "local role"


def test_discover_agents_backfills_name_from_stem(layered):
    """A yaml with no `name:` key still lands in the roster, named by its file stem."""
    global_dir, _ = layered
    (global_dir / "agents").mkdir(parents=True)
    (global_dir / "agents" / "nameless.yaml").write_text("role: has no name key\n", encoding="utf-8")

    agents = ConfigLoader(config_dir=global_dir).discover_agents()

    assert [a["name"] for a in agents] == ["nameless"]


def test_discover_agents_warns_and_skips_unreadable_file(layered, caplog):
    """Warn-and-skip is the ONE malformed-YAML behavior (it retires agent_cmd's silent `pass`).

    Swallowing the parse error made a typo'd agents/orchestrator.yaml indistinguishable from a
    fresh install, and start's mint branch then overwrote it with the default template.
    """
    global_dir, _ = layered
    _write_agent(global_dir / "agents", "good")
    (global_dir / "agents" / "broken.yaml").write_text(BROKEN_YAML, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="localharness.config.loader"):
        agents = ConfigLoader(config_dir=global_dir).discover_agents()

    assert [a["name"] for a in agents] == ["good"]
    warnings = [r for r in caplog.records if "broken.yaml" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]


def test_discover_agents_on_error_called_once_per_bad_file(layered):
    """`on_error` is the CLI's channel for the user-visible warning — config/ never imports a
    rich console from cli/, so the caller supplies the printer."""
    global_dir, _ = layered
    _write_agent(global_dir / "agents", "good")
    (global_dir / "agents" / "broken.yaml").write_text(BROKEN_YAML, encoding="utf-8")
    (global_dir / "agents" / "also-broken.yaml").write_text(BROKEN_YAML, encoding="utf-8")

    seen: list[tuple[str, str]] = []
    agents = ConfigLoader(config_dir=global_dir).discover_agents(
        on_error=lambda path, exc: seen.append((path.name, type(exc).__name__))
    )

    assert [a["name"] for a in agents] == ["good"]
    assert sorted(name for name, _ in seen) == ["also-broken.yaml", "broken.yaml"]


def test_list_agents_returns_sorted_stem_union(layered):
    """list_agents() delegates to the same paths but keeps its today-shape: sorted stems."""
    global_dir, project = layered
    _write_agent(global_dir / "agents", "beta")
    _write_agent(global_dir / "agents", "alpha")
    _write_agent(project / ".localharness" / "agents", "alpha")
    _write_agent(project / ".localharness" / "agents", "gamma")

    assert ConfigLoader(config_dir=global_dir).list_agents() == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Plan 38-05 — ROADMAP Phase 38 criterion 4, asserted across the COMMANDS
# ---------------------------------------------------------------------------

def test_agent_list_and_loader_report_one_roster(layered, monkeypatch):
    """`agent list`, the start roster and ConfigLoader name the SAME agents.

    Criterion 4 says the three "report the same roster". After plan 38-05 that is true by
    construction — `agent list` (agent_cmd:166) and start (start_cmd:417) both call
    `ConfigLoader.discover_agents()` — but a construction claim is only worth the test that
    drives the real CLI. This invokes `agent list --json` through Typer and compares its names to
    the loader's two roster surfaces on the same layered tree (local `beta` shadowing global).
    """
    import json
    from typer.testing import CliRunner
    from localharness.cli.agent_cmd import agent_app

    monkeypatch.setenv("COLUMNS", "400")  # keep Rich from wrapping the JSON line
    global_dir, project = layered
    _write_agent(global_dir / "agents", "alpha")
    _write_agent(global_dir / "agents", "beta", role="global beta")
    _write_agent(project / ".localharness" / "agents", "beta", role="local beta")
    _write_agent(project / ".localharness" / "agents", "gamma")

    result = CliRunner().invoke(agent_app, ["list", "--json", "--config-dir", str(global_dir)])
    assert result.exit_code == 0, result.output
    cli_agents = json.loads(result.output)

    loader = ConfigLoader(config_dir=global_dir)
    assert sorted(a["name"] for a in cli_agents) == sorted(
        a["name"] for a in loader.discover_agents()
    )
    assert sorted(a["name"] for a in cli_agents) == loader.list_agents() == ["alpha", "beta", "gamma"]
    # ...and the same shadowing decision, not just the same names
    assert [a["role"] for a in cli_agents if a["name"] == "beta"] == ["local beta"]
