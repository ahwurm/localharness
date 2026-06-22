"""Tests for localharness doctor command."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()

_VALID_CONFIG = """\
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


def _write_valid_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(_VALID_CONFIG)


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_all_pass(mock_httpx, tmp_path):
    """Valid config, reachable LLM -> exit code 0, output contains checkmarks."""
    _write_valid_config(tmp_path)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"id": "test-model:7b", "max_model_len": 131072}]
    }
    mock_httpx.get.return_value = mock_response
    # /tokenize reachability check (FIX 3): return a valid 200 count response.
    mock_tok = MagicMock()
    mock_tok.status_code = 200
    mock_tok.json.return_value = {"count": 1}
    mock_httpx.post.return_value = mock_tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    # Should contain some pass indicators
    assert result.output  # has output


def test_doctor_no_config(tmp_path):
    """No config.yaml -> exit code 1, output contains 'config' failure."""
    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 1
    combined = (result.output or "").lower() + (result.stderr or "").lower()
    assert "config" in combined


def test_doctor_python_version(tmp_path):
    """Always passes on 3.12+ -> output contains 'Python'."""
    _write_valid_config(tmp_path)
    with patch("localharness.cli.doctor_cmd.httpx") as mock_httpx:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "test-model:7b"}]}
        mock_httpx.get.return_value = mock_response
        result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert "Python" in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_llm_unreachable(mock_httpx, tmp_path):
    """httpx raises connection error -> exit code 1, LLM endpoint check fails."""
    _write_valid_config(tmp_path)
    import httpx as real_httpx
    mock_httpx.get.side_effect = real_httpx.ConnectError("connection refused")
    mock_httpx.ConnectError = real_httpx.ConnectError
    mock_httpx.TimeoutException = real_httpx.TimeoutException

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "LLM" in combined or "endpoint" in combined.lower()


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_shows_tool_mode(mock_httpx, tmp_path):
    """Config with supports_function_calling -> output contains 'Tool calling'."""
    _write_valid_config(tmp_path)
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"id": "test-model:7b"}]}
    mock_httpx.get.return_value = mock_response

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert "Tool calling" in result.output
