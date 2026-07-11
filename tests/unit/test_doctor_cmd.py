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


# --- #9: /tokenize check branches by provider_type (not an unconditional vLLM probe) ---


def _write_config(tmp_path: Path, provider_type: str, base_url: str, model: str = "m") -> None:
    (tmp_path / "config.yaml").write_text(
        'version: "1"\n'
        "provider:\n"
        f"  provider_type: {provider_type}\n"
        f"  base_url: {base_url}\n"
        f"  default_model: {model}\n"
        "  available_models:\n"
        f"    - {model}\n"
        "  supports_function_calling: true\n"
        "  timeout_seconds: 600.0\n"
    )


def _models_resp(payload: dict) -> MagicMock:
    r = MagicMock()
    r.json.return_value = payload
    return r


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_ollama_tokenize_is_info_not_failure(mock_httpx, tmp_path):
    """#9: Ollama serves no /tokenize — doctor must NOT probe it or count a failure; an
    INFO line explains approximate counting. Exit 0."""
    _write_config(tmp_path, "ollama", "http://localhost:11434", model="m")
    mock_httpx.get.return_value = _models_resp({"models": [{"name": "m"}]})

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "approximate" in result.output.lower()
    assert "✗ /tokenize" not in result.output and "tokenize unreachable" not in result.output.lower()
    mock_httpx.post.assert_not_called()  # no /tokenize probe on a runtime that has none


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_lmstudio_tokenize_is_info_not_failure(mock_httpx, tmp_path):
    """#9: LM Studio has no /tokenize — INFO, not a failure. Exit 0, no probe."""
    _write_config(tmp_path, "lmstudio", "http://localhost:1234/v1", model="m")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m"}]})

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "approximate" in result.output.lower()
    mock_httpx.post.assert_not_called()


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_llamacpp_tokenize_exact(mock_httpx, tmp_path):
    """#9: llama.cpp serves /tokenize with a {tokens:[...]} shape — doctor checks it with the
    llama.cpp contract (POST {content}) and reports EXACT counts. Exit 0."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m"}]})
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"tokens": [1, 2]}
    mock_httpx.post.return_value = tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "exact" in result.output.lower()
    # probed with llama.cpp's {content} shape, not vLLM's {model,prompt}
    _, kwargs = mock_httpx.post.call_args
    assert "content" in kwargs.get("json", {})


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_vllm_tokenize_absent_still_fails(mock_httpx, tmp_path):
    """#9: vLLM SHOULD serve /tokenize — a 404 there stays a real FAILURE (exit 1)."""
    _write_config(tmp_path, "vllm", "http://localhost:8000/v1", model="m")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m"}]})
    tok = MagicMock()
    tok.status_code = 404
    tok.json.return_value = {}
    mock_httpx.post.return_value = tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "tokenize" in result.output.lower()


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_llamacpp_meta_nctx_reconciles(mock_httpx, tmp_path):
    """#9: llama.cpp reports its window as /v1/models meta.n_ctx — 5b must read it so it does
    NOT print 'Served max_model_len not reported'."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m", "meta": {"n_ctx": 32768}}]})
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"tokens": [1]}
    mock_httpx.post.return_value = tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert "not reported" not in result.output.lower()
