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


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_lmstudio_reconciles_served_window(mock_httpx, tmp_path):
    """#13: LM Studio reports its window at /api/v0/models (loaded_context_length /
    max_context_length), NOT /v1/models. Doctor must query it to reconcile the budget instead
    of reporting 'max_model_len not reported'."""
    _write_config(tmp_path, "lmstudio", "http://localhost:1234/v1", model="m")
    v1 = _models_resp({"data": [{"id": "m"}]})  # /v1/models exposes no window
    apiv0 = _models_resp(
        {"data": [{"id": "m", "state": "loaded", "loaded_context_length": 8192,
                   "max_context_length": 32768}]}
    )
    mock_httpx.get.side_effect = [v1, apiv0]

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    urls = [c.args[0] for c in mock_httpx.get.call_args_list]
    assert any("/api/v0/models" in u for u in urls), urls  # discovered the served window
    assert "not reported" not in result.output.lower()


# --- #16: doctor must build the model-probe URL from the STRIPPED root (base_url always
# carries a /v1 suffix), hit Ollama's native /api/tags, and FAIL the model check on a
# non-2xx probe instead of green-lighting an empty 404 body. ---


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_v1_base_probes_single_v1_models(mock_httpx, tmp_path):
    """#16: a realistic base_url already ending in /v1 must probe exactly <root>/v1/models —
    NOT /v1/v1/models (init always writes base_url WITH the /v1 suffix)."""
    _write_config(tmp_path, "vllm", "http://localhost:8000/v1", model="m")
    resp = _models_resp({"data": [{"id": "m"}]})
    resp.status_code = 200
    mock_httpx.get.return_value = resp
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"count": 1}
    mock_httpx.post.return_value = tok

    runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    get_urls = [c.args[0] for c in mock_httpx.get.call_args_list]
    assert "http://localhost:8000/v1/models" in get_urls, get_urls
    assert not any("/v1/v1" in u for u in get_urls), get_urls


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_ollama_v1_base_probes_native_api_tags(mock_httpx, tmp_path):
    """#16: an Ollama base_url (…:11434/v1) must probe the native /api/tags at the server
    root — NOT /v1/api/tags (Ollama's tags endpoint lives at the root, not under /v1)."""
    _write_config(tmp_path, "ollama", "http://localhost:11434/v1", model="m")
    resp = _models_resp({"models": [{"name": "m"}]})
    resp.status_code = 200
    mock_httpx.get.return_value = resp

    runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    get_urls = [c.args[0] for c in mock_httpx.get.call_args_list]
    assert "http://localhost:11434/api/tags" in get_urls, get_urls
    assert not any("/v1/api/tags" in u for u in get_urls), get_urls


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_non_2xx_probe_fails_model_check(mock_httpx, tmp_path):
    """#16: a non-2xx model-probe response (404, no model list) must FAIL the model check —
    an empty error body must not sail through the benefit-of-doubt pass as 'Model available'."""
    _write_config(tmp_path, "vllm", "http://localhost:8000/v1", model="m")
    bad = MagicMock()
    bad.status_code = 404
    bad.json.return_value = {"error": "not found"}  # no data / models keys
    mock_httpx.get.return_value = bad
    tok = MagicMock()  # make /tokenize pass so the model check is the only failure
    tok.status_code = 200
    tok.json.return_value = {"count": 1}
    mock_httpx.post.return_value = tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert "✓ Model available" not in result.output, result.output
    assert result.exit_code == 1, result.output
