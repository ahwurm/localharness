"""Tests for localharness agent create and agent list commands."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.agent_cmd import agent_app

runner = CliRunner()


# ---------------------------------------------------------------------------
# agent create — name validation
# ---------------------------------------------------------------------------

def test_agent_create_invalid_name_uppercase(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "MyAgent",
        "--global",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 1
    assert "invalid" in result.output.lower() or "Invalid" in result.output


def test_agent_create_invalid_name_spaces(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "my agent",
        "--global",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 1


def test_agent_create_invalid_name_starts_with_digit(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "1agent",
        "--global",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# agent create — mutual exclusion of --global / --project
# ---------------------------------------------------------------------------

def test_agent_create_both_flags_exits_error(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "test-agent",
        "--global",
        "--project",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 1
    assert "Cannot use both" in result.output


# ---------------------------------------------------------------------------
# agent create — --global flag (no prompt)
# ---------------------------------------------------------------------------

def test_agent_create_global_writes_to_config_dir(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "test-agent",
        "--global",
        "--role", "Test role",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    expected = tmp_path / "agents" / "test-agent.yaml"
    assert expected.exists()


def test_agent_create_global_generates_valid_yaml(tmp_path):
    runner.invoke(agent_app, [
        "create", "test-agent",
        "--global",
        "--role", "Test role",
        "--config-dir", str(tmp_path),
    ])
    path = tmp_path / "agents" / "test-agent.yaml"
    data = yaml.safe_load(path.read_text())
    assert data["name"] == "test-agent"
    assert "role" in data
    assert data.get("model", "inherit") == "inherit"


# ---------------------------------------------------------------------------
# agent create — --project flag (no prompt)
# ---------------------------------------------------------------------------

def test_agent_create_project_writes_to_local_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(agent_app, [
        "create", "local-agent",
        "--project",
        "--role", "Local role",
        "--config-dir", str(tmp_path / "global"),
    ])
    assert result.exit_code == 0
    expected = tmp_path / ".localharness" / "agents" / "local-agent.yaml"
    assert expected.exists()


# ---------------------------------------------------------------------------
# agent create — interactive prompt (neither flag)
# ---------------------------------------------------------------------------

def test_agent_create_prompt_global_answer(tmp_path):
    result = runner.invoke(
        agent_app,
        [
            "create", "prompted-agent",
            "--role", "Prompted",
            "--config-dir", str(tmp_path),
        ],
        input="global\n",
    )
    assert result.exit_code == 0
    expected = tmp_path / "agents" / "prompted-agent.yaml"
    assert expected.exists()


def test_agent_create_prompt_project_answer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        agent_app,
        [
            "create", "prompted-proj",
            "--role", "Prompted",
            "--config-dir", str(tmp_path / "global"),
        ],
        input="project\n",
    )
    assert result.exit_code == 0
    expected = tmp_path / ".localharness" / "agents" / "prompted-proj.yaml"
    assert expected.exists()


# ---------------------------------------------------------------------------
# agent create — dry-run
# ---------------------------------------------------------------------------

def test_agent_create_dry_run_prints_yaml_no_file(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "dry-agent",
        "--global",
        "--role", "Dry role",
        "--dry-run",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "dry-agent" in result.output
    # No file written
    assert not (tmp_path / "agents" / "dry-agent.yaml").exists()


# ---------------------------------------------------------------------------
# agent list — empty
# ---------------------------------------------------------------------------

def test_agent_list_no_agents(tmp_path):
    result = runner.invoke(agent_app, [
        "list",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "No agents" in result.output


# ---------------------------------------------------------------------------
# agent list — with agents (Rich table)
# ---------------------------------------------------------------------------

def test_agent_list_shows_table(tmp_path):
    # Create an agent first
    runner.invoke(agent_app, [
        "create", "listed-agent",
        "--global",
        "--role", "Listed role",
        "--config-dir", str(tmp_path),
    ])
    result = runner.invoke(agent_app, [
        "list",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "listed-agent" in result.output
    assert "Name" in result.output


# ---------------------------------------------------------------------------
# agent list — JSON output
# ---------------------------------------------------------------------------

def test_agent_list_json_output(tmp_path):
    runner.invoke(agent_app, [
        "create", "json-agent",
        "--global",
        "--role", "JSON role",
        "--config-dir", str(tmp_path),
    ])
    result = runner.invoke(agent_app, [
        "list",
        "--json",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    names = [a["name"] for a in data]
    assert "json-agent" in names
