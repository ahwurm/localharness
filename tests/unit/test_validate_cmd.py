"""Tests for localharness validate command."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()

_VALID_HARNESS = """\
version: "1"
provider:
  provider_type: ollama
  base_url: http://localhost:11434
  default_model: test-model:7b
  available_models:
    - test-model:7b
  supports_function_calling: true
  timeout_seconds: 300.0
"""

_VALID_AGENT = """\
name: test-agent
role: A test agent for validation
"""

_INVALID_AGENT = """\
name: BadName
role: Agent with bad name
"""


def _setup_config_dir(tmp_path: Path) -> Path:
    """Write a valid harness config.yaml."""
    (tmp_path / "config.yaml").write_text(_VALID_HARNESS)
    return tmp_path


def test_validate_valid_config(tmp_path):
    """Valid agent YAML -> exit code 0, output contains 'valid'."""
    _setup_config_dir(tmp_path)
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "test-agent.yaml").write_text(_VALID_AGENT)

    result = runner.invoke(app, ["validate", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_validate_invalid_config(tmp_path):
    """Agent YAML with bad name -> exit code 1, output contains error details."""
    _setup_config_dir(tmp_path)
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "bad.yaml").write_text(_INVALID_AGENT)

    result = runner.invoke(app, ["validate", "--config-dir", str(tmp_path)])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "invalid" in combined.lower() or "error" in combined.lower()


def test_validate_no_configs(tmp_path):
    """Empty config dir (no config.yaml, no agents) -> exit code 2."""
    result = runner.invoke(app, ["validate", "--config-dir", str(tmp_path)])
    assert result.exit_code == 2


def test_validate_path_flag(tmp_path):
    """--path specific file -> only that file validated."""
    _setup_config_dir(tmp_path)
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "good-agent.yaml").write_text(_VALID_AGENT)
    (agents_dir / "bad.yaml").write_text(_INVALID_AGENT)

    # When validating only the good file directly, should exit 0
    good_path = str(agents_dir / "good-agent.yaml")
    result = runner.invoke(app, ["validate", good_path, "--config-dir", str(tmp_path)])
    # Should validate only that one file and it should be valid
    assert result.exit_code == 0, result.output
