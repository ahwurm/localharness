"""D1 — a project with a workspace layer and NO machine config is not an invisible project.

`localharness init --workspace` creates a project layer without ever touching the machine-global
dir, and `agent create --project` writes into it. That is a supported, documented starting point:
clone a repo that ships a `.localharness/`, or scaffold one before you have configured the
machine. Every reader then gated on the GLOBAL `config.yaml` existing and exited BEFORE reaching
its own already-wired workspace-aware loader, so all three commands reported an empty machine and
said nothing at all about the layer sitting in the current directory.

The whole file is one flow — scaffold, create, read — because that is the shape of the bug: each
command in isolation looked reasonable, and the composition was what lied.

What is deliberately NOT changed: the machine still is not configured, and the commands still say
so. `doctor` still fails, `config show` still exits 2. The fix is that they say it in ONE line and
then go on to answer the question that was asked.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.paths import WORKSPACE_DIR_NAME

runner = CliRunner()


@pytest.fixture
def unconfigured_project(tmp_path, monkeypatch) -> Path:
    """CWD inside a project, with a $HOME that has NO `.localharness` at all."""
    for var in ("LOCALHARNESS_DIR", "LOCALHARNESS_HOME", "LOCALHARNESS_ENDPOINT",
                "LOCALHARNESS_MODEL"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


@pytest.fixture
def scaffolded(unconfigured_project) -> Path:
    """The badmood-D repro, exactly: scaffold a workspace, create a project agent in it."""
    assert runner.invoke(app, ["init", "--workspace"]).exit_code == 0
    assert runner.invoke(app, ["agent", "create", "deployer", "--project"]).exit_code == 0
    workspace = unconfigured_project / WORKSPACE_DIR_NAME
    assert (workspace / "agents" / "deployer.yaml").exists()
    assert not (Path.home() / WORKSPACE_DIR_NAME).exists(), "the scaffold touched the machine"
    return workspace


def test_doctor_shows_the_workspace_layer_it_found(scaffolded):
    """One failure line for the unconfigured machine — and then the layer report anyway."""
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1, result.output
    assert str(scaffolded) in result.output, "doctor never mentioned the workspace"
    # ONE failure, not two: a missing machine config must not also be reported as an invalid one.
    assert "1 issue(s) found." in result.output
    assert "localharness init" in result.output


def test_doctor_reports_the_workspace_agents_directory(scaffolded):
    result = runner.invoke(app, ["doctor"])

    assert str(scaffolded / "agents") in result.output


def test_config_show_renders_the_layers_with_the_global_ones_missing(scaffolded):
    """The command whose entire job is saying where config comes from must not go silent."""
    result = runner.invoke(app, ["config", "show"])

    assert str(scaffolded / "config.yaml") in result.output, "the workspace layer is not shown"
    assert "missing" in result.output, "the absent global layer is not marked"
    assert str(Path.home() / WORKSPACE_DIR_NAME / "config.yaml") in result.output


def test_config_show_json_is_valid_json_with_no_machine_config(scaffolded):
    """A machine-output run must emit a payload, not prose on stderr and nothing on stdout."""
    result = runner.invoke(app, ["config", "show", "--json"])

    payload = json.loads(result.stdout)
    layers = {row["layer"]: row for row in payload["layers"]}
    assert any(not row["exists"] for row in layers.values()), "no layer is marked absent"
    assert any(row["exists"] for row in layers.values()), "the workspace layer is not present"


def test_config_show_names_the_keys_the_workspace_sets(scaffolded):
    """A workspace that actually sets something has that shown, attributed to its own file."""
    (scaffolded / "config.yaml").write_text(
        yaml.dump({"org": {"log_level": "debug"}}), encoding="utf-8"
    )

    result = runner.invoke(app, ["config", "show"])

    assert "org.log_level" in result.output
    assert "debug" in result.output


def test_agent_list_lists_the_workspace_agent(scaffolded):
    result = runner.invoke(app, ["agent", "list"])

    assert result.exit_code == 0, result.output
    assert "deployer" in result.output


def test_agent_list_json_emits_an_array_on_an_empty_roster(unconfigured_project):
    """39-06's deferred item: `--json` is a machine contract. An empty roster is `[]`, and prose
    on stdout is a parse error in whatever is reading it."""
    runner.invoke(app, ["init", "--workspace"])

    result = runner.invoke(app, ["agent", "list", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def test_agent_list_json_lists_the_workspace_agent(scaffolded):
    result = runner.invoke(app, ["agent", "list", "--json"])

    names = [a["name"] for a in json.loads(result.stdout)]
    assert names == ["deployer"]


# --------------------------------------------------------------------------- controls


def test_no_workspace_and_no_config_is_unchanged(unconfigured_project):
    """LAYR-03: with nothing to discover, doctor stops exactly where it always did — one failure,
    and none of the checks that need a config."""
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "1 issue(s) found." in result.output
    assert "Web search" not in result.output, "doctor ran past the missing-config stop"
