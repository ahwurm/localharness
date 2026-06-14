"""Tests for localharness init command."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.provider.client import CapabilityResult
from localharness.provider.detector import DetectorResult

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_window_probe(monkeypatch):
    """Keep init tests hermetic — the live-window probe makes a real HTTP call."""
    import localharness.cli.init_cmd as init_cmd
    monkeypatch.setattr(init_cmd, "_detect_max_model_len", lambda *_: None)


def test_fit_context_tokens_reserves_output():
    from localharness.cli.init_cmd import _fit_context_tokens
    assert _fit_context_tokens(65_536) == 61_440    # reference 64K window
    assert _fit_context_tokens(131_072) == 126_976  # 128K window
    assert _fit_context_tokens(8_000) == 8_192      # floor for tiny windows


def test_detect_llamacpp_nctx_parses_props(monkeypatch):
    """llama.cpp /props.default_generation_settings.n_ctx is read as the served window."""
    import httpx
    from localharness.cli import init_cmd

    class _Resp:
        def json(self):
            return {"default_generation_settings": {"n_ctx": 65536}}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    # base_url carries the /v1 suffix; /props lives at the server root
    assert init_cmd._detect_llamacpp_nctx("http://localhost:8080/v1") == 65536


def test_detect_llamacpp_nctx_returns_none_on_error(monkeypatch):
    """A llama.cpp probe failure falls back to None (→ safe context default)."""
    import httpx
    from localharness.cli import init_cmd

    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert init_cmd._detect_llamacpp_nctx("http://localhost:8080/v1") is None


def _make_detector_result(found: bool = True, models: list[str] | None = None) -> DetectorResult:
    models = models or ["test-model:7b"]
    return DetectorResult(
        found=found,
        provider_type="ollama",
        base_url="http://localhost:11434",
        models=models,
        suggested_model=models[0] if models else "",
        probe_duration_ms=42.0,
    )


def _make_capability_result(mode: str = "native") -> CapabilityResult:
    return CapabilityResult(
        tool_call_mode=mode,
        context_window=128_000,
        supports_streaming=True,
        probe_duration_ms=10.0,
        probe_error=None,
    )


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_writes_config(mock_client_cls, mock_detect, tmp_path):
    """detect_provider returning found=True -> config.yaml written."""
    mock_detect.return_value = _make_detector_result()
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    config_file = tmp_path / "config.yaml"
    assert config_file.exists(), "config.yaml should be written"
    content = config_file.read_text()
    assert "test-model" in content
    assert "base_url" in content


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_writes_local_decode_timeout(mock_client_cls, mock_detect, tmp_path):
    """Written config uses the 600s local-decode timeout, not the old too-tight 300s
    (a 4096-token completion at ~10 tok/s is ~410s)."""
    mock_detect.return_value = _make_detector_result()
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    content = (tmp_path / "config.yaml").read_text()
    assert "timeout_seconds: 600" in content


@patch("localharness.cli.init_cmd.detect_provider")
def test_init_no_server(mock_detect, tmp_path):
    """detect_provider returning found=False -> exit code 1, error message."""
    mock_detect.return_value = _make_detector_result(found=False, models=[])

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 1
    combined = (result.output or "") + (result.stderr or "")
    assert "No local LLM detected" in combined


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_existing_config_no_force(mock_client_cls, mock_detect, tmp_path):
    """config.yaml exists, --force not set -> prompts with 'n' -> config not overwritten."""
    # Create existing config
    config_file = tmp_path / "config.yaml"
    original_content = "version: '1'\nprovider:\n  base_url: http://original\n  provider_type: ollama\n  default_model: old-model\n"
    config_file.write_text(original_content)

    mock_detect.return_value = _make_detector_result()
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    # User answers "n" to the overwrite prompt
    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path)], input="n\n")
    # Should not overwrite
    assert config_file.read_text() == original_content


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_force_flag(mock_client_cls, mock_detect, tmp_path):
    """config.yaml exists, --force set -> config overwritten without prompt."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("version: '1'\nold: true\n")

    mock_detect.return_value = _make_detector_result()
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    content = config_file.read_text()
    assert "old: true" not in content
    assert "test-model" in content


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_shows_tool_call_mode(mock_client_cls, mock_detect, tmp_path):
    """detect_capabilities returning native -> output contains 'Tool calling: native'."""
    mock_detect.return_value = _make_detector_result()
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result("native"))
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    assert "Tool calling: native" in result.output


@patch("localharness.cli.init_cmd.LLMClient")
def test_init_endpoint_override(mock_client_cls, tmp_path):
    """--endpoint and --model set -> skips probe, writes endpoint directly."""
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path), "--force",
         "--endpoint", "http://localhost:9999/v1", "--model", "custom-model"],
    )
    assert result.exit_code == 0, result.output
    config_file = tmp_path / "config.yaml"
    content = config_file.read_text()
    assert "9999" in content
    assert "custom-model" in content
