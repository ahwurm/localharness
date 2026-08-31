"""Tests for localharness init command."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
# Bound at import time, BEFORE the autouse fixture stubs the module attribute — the direct
# identification tests below exercise the real function, everything else gets the hermetic stub.
from localharness.cli.init_cmd import _identify_endpoint_provider as _real_identify_provider
from localharness.provider.client import CapabilityResult
from localharness.provider.detector import DetectorResult

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_window_probe(monkeypatch):
    """Keep init tests hermetic — the live-window and runtime-identification probes both make
    real HTTP calls. Tests that exercise identification patch httpx or override this stub."""
    import localharness.cli.init_cmd as init_cmd
    monkeypatch.setattr(init_cmd, "_detect_max_model_len", lambda *_: None)
    monkeypatch.setattr(init_cmd, "_identify_endpoint_provider", lambda *_: "unknown")


def test_init_help_probe_order_matches_detector():
    """#52: `init --help` must list the real probe order derived from the detector's
    DEFAULT_PORTS — so it can never again omit :8000 (vLLM's stock port) or drift."""
    from localharness.provider.detector import DEFAULT_PORTS

    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0, result.output
    for port in DEFAULT_PORTS:
        assert f":{port}" in result.output, (port, result.output)
    assert ":8000" in result.output  # the specific omission #52 flags


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_writes_the_full_served_window(mock_client_cls, mock_detect, tmp_path, monkeypatch):
    """#145: init writes the served window VERBATIM. It used to write window − 4,096 while the
    harness ALSO reserved 4,096 at runtime — the same room reserved twice, which pushed the
    emergency floor below the 0.95 full-compact trigger on small windows.

    The budget the harness actually spends is still window − response_reserve(window); that
    subtraction now happens in exactly one place, and it is not this one."""
    import httpx

    from localharness.agent.context import response_reserve

    mock_detect.return_value = DetectorResult(
        found=True, provider_type="llamacpp", base_url="http://localhost:8080/v1",
        models=["qwen"], suggested_model="qwen", probe_duration_ms=1.0,
    )
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    class _Resp:
        def json(self):
            return {"default_generation_settings": {"n_ctx": 32_768}}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    assert "max_context_tokens: 32768" in (tmp_path / "config.yaml").read_text()
    assert "4,096 output reservation" not in result.output, "the double-reserve claim is gone"
    assert response_reserve(32_768) == 4_096  # …and the reserve still happens, once, at runtime


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


def test_detect_lmstudio_ctx_prefers_loaded(monkeypatch):
    """Issue #13a: the loaded model's loaded_context_length is the served window."""
    import httpx
    from localharness.cli import init_cmd

    class _Resp:
        def json(self):
            return {"object": "list", "data": [
                {"id": "a", "state": "not-loaded", "max_context_length": 131072},
                {"id": "b", "state": "loaded", "max_context_length": 32768, "loaded_context_length": 16384},
            ]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    # base_url carries /v1; /api/v0 lives at the server root
    assert init_cmd._detect_lmstudio_ctx("http://localhost:1234/v1") == 16384


def test_detect_lmstudio_ctx_falls_back_to_max(monkeypatch):
    """Issue #13a: with no loaded model reporting loaded_context_length, fall back to
    the largest max_context_length."""
    import httpx
    from localharness.cli import init_cmd

    class _Resp:
        def json(self):
            return {"object": "list", "data": [
                {"id": "a", "state": "not-loaded", "max_context_length": 32768},
                {"id": "b", "state": "not-loaded", "max_context_length": 131072},
            ]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    assert init_cmd._detect_lmstudio_ctx("http://localhost:1234/v1") == 131072


def test_detect_lmstudio_ctx_returns_none_on_error(monkeypatch):
    """An LM Studio probe failure falls back to None (→ safe context default)."""
    import httpx
    from localharness.cli import init_cmd

    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert init_cmd._detect_lmstudio_ctx("http://localhost:1234/v1") is None


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


def _make_capability_result(
    mode: str = "native", *, server_reached: bool = True, probe_error: str | None = None
) -> CapabilityResult:
    # server_reached defaults True: these fixtures stand in for a probe the server ANSWERED.
    # A probe that never reached the endpoint is an explicit opt-out (init must refuse to write).
    return CapabilityResult(
        tool_call_mode=mode,
        context_window=128_000,
        supports_streaming=True,
        probe_duration_ms=10.0,
        probe_error=probe_error,
        server_reached=server_reached,
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
def test_init_creates_agents_directory(mock_client_cls, mock_detect, tmp_path):
    """#53: a fresh init must create the agents directory. doctor points a user at `init`
    as the remedy for a missing agents dir, so init has to actually create one."""
    mock_detect.return_value = _make_detector_result()
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    agents_dir = tmp_path / "agents"
    assert agents_dir.exists() and agents_dir.is_dir(), "init should create the agents directory"


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_stamps_current_defaults_revision(mock_client_cls, mock_detect, tmp_path):
    """A freshly-init'd config is born stamped at the current defaults revision, so the first
    `start` never spuriously migrates AND a later deliberate removal of a default is respected
    (removal-respect only holds for configs stamped current at birth)."""
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
    from localharness.config.migrate import plan

    mock_detect.return_value = _make_detector_result()
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    import yaml

    data = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert data["org"]["permissions"]["defaults_revision"] == CURRENT_DEFAULTS_REVISION
    # already current → auto-migration is a no-op on a fresh install
    assert plan(data) is None


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


# ---------------------------------------------------------------------------
# --endpoint: an unreachable endpoint must NOT report success or write a config
# ---------------------------------------------------------------------------

@patch("localharness.cli.init_cmd.LLMClient")
def test_init_endpoint_unreachable_errors_and_writes_nothing(mock_client_cls, tmp_path):
    """A probe that never reached the server proves nothing about tool calling: init used to print
    '⚠ XML fallback' + '✓ LocalHarness configured' and persist supports_function_calling: false for
    a dead port. It must fail non-zero, name the cause, and leave no config behind."""
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result(
        "xml",
        server_reached=False,
        probe_error="inference endpoint 127.0.0.1:1 unreachable (TCP connect failed)",
    ))
    mock_client_cls.return_value = mock_client

    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path), "--force",
         "--endpoint", "http://127.0.0.1:1/v1", "--model", "fake-model"],
    )
    assert result.exit_code != 0, result.output
    assert "LocalHarness configured" not in result.output
    assert "TCP connect failed" in result.output, "the message must name the concrete cause"
    assert not (tmp_path / "config.yaml").exists(), "no config may be written for a dead endpoint"


@patch("localharness.cli.init_cmd.LLMClient")
def test_init_endpoint_persists_detected_runtime(mock_client_cls, tmp_path, monkeypatch):
    """--endpoint used to hardcode provider_type: unknown, which skips exact GGUF token counting
    for Ollama/LM Studio. The identified runtime must reach config.yaml."""
    import localharness.cli.init_cmd as init_cmd
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client
    monkeypatch.setattr(init_cmd, "_identify_endpoint_provider", lambda *_: "ollama")

    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path), "--force",
         "--endpoint", "http://ollama-host:11434/v1", "--model", "llama3"],
    )
    assert result.exit_code == 0, result.output
    assert "provider_type: ollama" in (tmp_path / "config.yaml").read_text()


@patch("localharness.cli.init_cmd.LLMClient")
def test_init_endpoint_says_so_when_runtime_unidentified(mock_client_cls, tmp_path):
    """"unknown" is a real degradation (approximate token counting) — init must say it out loud
    instead of recording it silently. The autouse fixture stubs identification to 'unknown'."""
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path), "--force",
         "--endpoint", "http://somewhere:9000/v1", "--model", "m"],
    )
    assert result.exit_code == 0, result.output
    assert "Could not identify the runtime" in result.output
    assert "provider_type: unknown" in (tmp_path / "config.yaml").read_text()


# --- _identify_endpoint_provider: the detector's shape rules, reused ------------------------

class _IdResp:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_identify_endpoint_provider_ollama_native_api(monkeypatch):
    """Ollama's /v1/models is a plain OpenAI list (indistinguishable from vLLM) — its native
    /api/tags is what identifies it, so a non-standard port still classifies correctly."""
    import httpx

    def _get(url, **kw):
        if url.endswith("/api/tags"):
            return _IdResp({"models": [{"name": "llama3"}]})
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _get)
    assert _real_identify_provider("http://ollama-host:12345/v1") == "ollama"


def test_identify_endpoint_provider_llamacpp_owned_by(monkeypatch):
    """llama.cpp self-identifies via owned_by in /v1/models (detector shape rule, reused)."""
    import httpx

    def _get(url, **kw):
        if url.endswith("/models"):
            return _IdResp({"data": [{"id": "q", "owned_by": "llamacpp"}]})
        raise httpx.ConnectError("no native api here")

    monkeypatch.setattr(httpx, "get", _get)
    assert _real_identify_provider("http://host:9999/v1") == "llamacpp"


def test_identify_endpoint_provider_unknown_when_probes_fail(monkeypatch):
    """A silent/unreachable endpoint falls back to 'unknown' — never a guess."""
    import httpx

    def _boom(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert _real_identify_provider("http://host:9999/v1") == "unknown"


# ---------------------------------------------------------------------------
# #118: --endpoint without --model asks the server before refusing
# ---------------------------------------------------------------------------

def _models_payload(*ids: str):
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


def _flat(result) -> str:
    """Rich wraps at 80 cols — normalize whitespace so a phrase assertion survives the wrap."""
    return " ".join(((result.output or "") + (result.stderr or "")).split())


def test_list_endpoint_models_parses_ids(monkeypatch):
    """#118: the /v1/models listing is the source for auto-selection — parse its ids."""
    import httpx
    from localharness.cli import init_cmd

    class _Resp:
        def json(self):
            return _models_payload("a/b-awq", "c/d-fp8")

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    assert init_cmd._list_endpoint_models("http://host:9999/v1") == ["a/b-awq", "c/d-fp8"]


def test_list_endpoint_models_none_when_unreachable(monkeypatch):
    """An unreadable listing is None ("couldn't ask"), NOT [] ("serves nothing")."""
    import httpx
    from localharness.cli import init_cmd

    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert init_cmd._list_endpoint_models("http://host:9999/v1") is None


@patch("localharness.cli.init_cmd.LLMClient")
def test_init_endpoint_auto_selects_sole_served_model(mock_client_cls, tmp_path, monkeypatch):
    """#118: exactly one model served -> nothing to disambiguate. init selects it, says so,
    and writes it to config instead of erroring out with '--model is required'."""
    import httpx
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    class _Resp:
        def json(self):
            return _models_payload("solo-model")

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path), "--force", "--endpoint", "http://h:9/v1"],
    )
    assert result.exit_code == 0, result.output
    assert "the only model served" in _flat(result)
    content = (tmp_path / "config.yaml").read_text()
    assert "solo-model" in content, "the auto-selected model must reach config.yaml"


@patch("localharness.cli.init_cmd.LLMClient")
def test_init_endpoint_lists_served_ids_when_ambiguous(mock_client_cls, tmp_path, monkeypatch):
    """#118: several models -> keep the error, but name the ids and where they came from,
    so the user knows what to pass to --model."""
    import httpx
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    class _Resp:
        def json(self):
            return _models_payload("model-a", "model-b")

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path), "--force", "--endpoint", "http://h:9/v1"],
    )
    assert result.exit_code == 1, result.output
    flat = _flat(result)
    assert "--model is required" in flat
    assert "model-a" in flat and "model-b" in flat, "the served ids must be listed"
    assert "/models" in flat, "the message must say where the list came from"
    assert not (tmp_path / "config.yaml").exists()


@patch("localharness.cli.init_cmd.LLMClient")
def test_init_endpoint_no_models_served_says_so(mock_client_cls, tmp_path, monkeypatch):
    """#118: a reachable endpoint serving NOTHING is a different failure from an unreadable
    one — say the listing is empty rather than claiming we couldn't read it."""
    import httpx
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    class _Resp:
        def json(self):
            return _models_payload()

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path), "--force", "--endpoint", "http://h:9/v1"],
    )
    assert result.exit_code == 1, result.output
    assert "lists no models" in _flat(result)
    assert not (tmp_path / "config.yaml").exists()


@patch("localharness.cli.init_cmd.LLMClient")
def test_init_endpoint_no_model_unreachable_keeps_honest_failure(mock_client_cls, tmp_path, monkeypatch):
    """#118 must not paper over a dead endpoint: an unreadable listing still errors,
    says it couldn't be read, and writes no config."""
    import httpx
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)

    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path), "--force", "--endpoint", "http://127.0.0.1:1/v1"],
    )
    assert result.exit_code == 1, result.output
    flat = _flat(result)
    assert "--model is required" in flat and "could not be read" in flat
    assert not (tmp_path / "config.yaml").exists()


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_lmstudio_fits_loaded_context(mock_client_cls, mock_detect, tmp_path, monkeypatch):
    """Issue #13a: LM Studio init fits the budget to the loaded model's loaded_context_length
    (16384), not its max_context_length (32768). #145: the window is written verbatim — the
    reply reserve is taken at runtime, so subtracting it here too would reserve it twice."""
    import httpx
    mock_detect.return_value = DetectorResult(
        found=True, provider_type="lmstudio", base_url="http://localhost:1234/v1",
        models=["qwen"], suggested_model="qwen", probe_duration_ms=1.0,
    )
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    class _Resp:
        def json(self):
            return {"object": "list", "data": [
                {"id": "qwen", "state": "loaded", "max_context_length": 32768, "loaded_context_length": 16384},
            ]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    content = (tmp_path / "config.yaml").read_text()
    assert "max_context_tokens: 16384" in content


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_ollama_prints_window_guidance(mock_client_cls, mock_detect, tmp_path):
    """Issue #13b: Ollama's served window isn't discoverable — init surfaces
    OLLAMA_CONTEXT_LENGTH guidance instead of silently keeping the default budget."""
    mock_detect.return_value = _make_detector_result()  # provider_type="ollama"
    mock_client = MagicMock()
    mock_client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    assert "OLLAMA_CONTEXT_LENGTH" in result.output
