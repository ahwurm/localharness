"""Tests for localharness doctor command."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()


class _StubCounter:
    """Stand-in for TokenCounter in doctor tests.

    doctor no longer probes /tokenize itself — it asks the counter for its RESOLVED mode, so
    doctor tests must stub the counter rather than the transport. (These tests patch `httpx`;
    TokenCounter talks urllib, so an httpx mock never covered it — which is why every doctor
    test broke the moment doctor started delegating.) Default mirrors the configured
    provider_type, i.e. "the config is accurate and the server is healthy"; individual tests
    override `mode` to exercise drift and degradation.
    """

    mode_override: str | None = None

    def __init__(self, base_url=None, model=None, provider_type=None, **_kw):
        if self.mode_override:
            self.mode = self.mode_override
        elif provider_type in ("vllm", "llamacpp"):
            self.mode = provider_type          # server-side /tokenize
        elif provider_type in ("ollama", "lmstudio"):
            self.mode = "exact_local"          # no /tokenize; counts from the model's GGUF vocab
        else:
            self.mode = "vllm"
        self.approximate = self.mode == "approximate"


@pytest.fixture(autouse=True)
def _stub_token_counter(monkeypatch):
    """Autouse: doctor asks the counter on every run, so every test needs it stubbed."""
    import localharness.agent.context as _ctx

    _StubCounter.mode_override = None
    monkeypatch.setattr(_ctx, "TokenCounter", _StubCounter)
    yield _StubCounter
    _StubCounter.mode_override = None

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


def test_doctor_fix_help_states_real_scope(tmp_path):
    """#53: `--fix` only creates a missing agents directory today — its help must name that
    real, narrow scope, not overpromise a generic 'Attempt to auto-fix detected issues'."""
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0, result.output
    assert "agents" in result.output.lower()  # states its actual scope


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


@patch("localharness.agent.gguf_tokenizer.resolve_gguf_path", return_value=None)
@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_ollama_tokenize_is_info_not_failure(mock_httpx, mock_resolve, tmp_path):
    """#9: Ollama serves no /tokenize — doctor must NOT probe it or count a failure; with no local
    GGUF for exact counting an INFO line explains the approximate fallback. Exit 0."""
    _StubCounter.mode_override = "approximate"  # no /tokenize, no usable GGUF
    _write_config(tmp_path, "ollama", "http://localhost:11434", model="m")
    mock_httpx.get.return_value = _models_resp({"models": [{"name": "m"}]})

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "approximate" in result.output.lower()
    assert "✗ /tokenize" not in result.output and "tokenize unreachable" not in result.output.lower()
    mock_httpx.post.assert_not_called()  # no /tokenize probe on a runtime that has none


@patch("localharness.agent.gguf_tokenizer.resolve_gguf_path", return_value=None)
@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_lmstudio_tokenize_is_info_not_failure(mock_httpx, mock_resolve, tmp_path):
    """#9: LM Studio has no /tokenize — with no local GGUF, INFO (approximate), not a failure. Exit 0."""
    _StubCounter.mode_override = "approximate"  # no /tokenize, no usable GGUF
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
    at = MagicMock()
    at.status_code = 200
    at.json.return_value = {"prompt": "<|im_start|>user\nx<|im_end|>\n"}
    mock_httpx.post.side_effect = (
        lambda url, **kw: at if url.endswith("/apply-template") else tok
    )

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "exact" in result.output.lower()
    assert "message-level" in result.output  # /apply-template capability reported
    # The CONTRACT-SHAPE guarantee ({content} for llama.cpp, not vLLM's {model,prompt}) moved
    # into TokenCounter when doctor stopped running its own probe, and is covered there —
    # tests/unit/test_context.py rejects a body without "model" and distinguishes both shapes,
    # incl. test_token_counter_llamacpp_router_mode_names_the_model for #141. Asserting it here
    # would only re-test the stub. What doctor still owns is the message-level report, above.
    assert any("/apply-template" in str(c) for c in mock_httpx.post.call_args_list)


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_llamacpp_router_mode_names_the_model(mock_httpx, tmp_path):
    """#141: llama.cpp ROUTER mode (one llama-server fronting several models) 400s any body
    without a "model" name. Doctor's probes must carry it, or a perfectly healthy router
    false-fails as 'tokenize unreachable' — the same gap that broke `start`."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="router-model")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "router-model"}]})

    def _post(url, **kw):
        body = kw.get("json") or {}
        r = MagicMock()
        if not body.get("model"):
            r.status_code = 400
            r.json.return_value = {
                "error": {"code": 400, "message": "model name is missing from the request"}
            }
            return r
        r.status_code = 200
        r.json.return_value = (
            {"prompt": "<|im_start|>user\nx<|im_end|>\n"}
            if url.endswith("/apply-template")
            else {"tokens": [1, 2]}
        )
        return r

    mock_httpx.post.side_effect = _post

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Tokenizer endpoint reachable" in result.output
    assert "message-level" in result.output  # /apply-template named the model too


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_vllm_tokenize_absent_still_fails(mock_httpx, tmp_path):
    """#9: vLLM SHOULD serve /tokenize — a 404 there stays a real FAILURE (exit 1)."""
    _StubCounter.mode_override = "approximate"  # /tokenize 404 -> counter finds none
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


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_llamacpp_warns_on_parallel_slot_split(mock_httpx, tmp_path):
    """llama-server defaults to multiple parallel slots and divides --ctx-size among them
    (observed live: a 32k launch served 8k/request via 4 slots). The harness is
    single-stream, so doctor surfaces the split with the --parallel 1 remedy — advisory
    only, exit stays 0 (a shared server may be deliberate).

    The budget is what makes 8k/request a PROBLEM, so the fixture now supplies one. Slot
    count alone is no longer treated as evidence: with a unified KV cache many slots share
    one undivided window, and keying on `total_slots > 1` false-fired on healthy servers."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _write_orchestrator(tmp_path, 28_672)

    def _get(url, *args, **kwargs):
        if str(url).endswith("/props"):
            return _models_resp(
                {"total_slots": 4, "default_generation_settings": {"n_ctx": 8192}}
            )
        return _models_resp({"data": [{"id": "m"}]})

    mock_httpx.get.side_effect = _get
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"tokens": [1, 2]}
    mock_httpx.post.return_value = tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "--parallel 1" in result.output
    assert "4 parallel slots" in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_llamacpp_single_slot_stays_silent(mock_httpx, tmp_path):
    """total_slots == 1 (or /props absent) must not emit the slot-split advisory."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")

    def _get(url, *args, **kwargs):
        if str(url).endswith("/props"):
            return _models_resp(
                {"total_slots": 1, "default_generation_settings": {"n_ctx": 32768}}
            )
        return _models_resp({"data": [{"id": "m"}]})

    mock_httpx.get.side_effect = _get
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"tokens": [1, 2]}
    mock_httpx.post.return_value = tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "--parallel 1" not in result.output


# --- #144: AMD GPU present + llama.cpp binary linked against Vulkan, not HIP -> warn --------


def _write_managed_llamacpp_config(tmp_path: Path, binary: str, model: str = "m") -> None:
    (tmp_path / "config.yaml").write_text(
        'version: "1"\n'
        "provider:\n"
        "  provider_type: llamacpp\n"
        "  base_url: http://localhost:8080/v1\n"
        f"  default_model: {model}\n"
        "  available_models:\n"
        f"    - {model}\n"
        "  supports_function_calling: true\n"
        "  timeout_seconds: 600.0\n"
        "server:\n"
        "  runtime: llamacpp\n"
        f"  binary: {binary}\n"
        f"  model: {model}.gguf\n"
        "  port: 8080\n"
    )


def _basic_llamacpp_mocks(mock_httpx, model: str = "m") -> None:
    mock_httpx.get.return_value = _models_resp({"data": [{"id": model}]})
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"tokens": [1, 2]}
    mock_httpx.post.return_value = tok


@patch("localharness.cli.doctor_cmd._has_amd_gpu", return_value=True)
@patch("localharness.cli.doctor_cmd._llama_server_linked_backends", return_value={"vulkan"})
@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_warns_amd_gpu_vulkan_only_binary(mock_httpx, mock_backends, mock_amd, tmp_path):
    """AMD GPU present + a managed llama.cpp binary linked ONLY against Vulkan -> a WARN
    advisory naming the fix, but doctor still exits 0 (a Vulkan build still works)."""
    binary = str(tmp_path / "llama-server")
    _write_managed_llamacpp_config(tmp_path, binary)
    _basic_llamacpp_mocks(mock_httpx)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Vulkan, not HIP/ROCm" in result.output
    assert "-DGGML_HIP=ON" in result.output
    mock_backends.assert_called_once_with(binary)


@patch("localharness.cli.doctor_cmd._has_amd_gpu", return_value=True)
@patch("localharness.cli.doctor_cmd._llama_server_linked_backends", return_value={"hip", "vulkan"})
@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_silent_when_binary_is_hip_linked(mock_httpx, mock_backends, mock_amd, tmp_path):
    """A binary linked against HIP (even alongside Vulkan support compiled in) is fine —
    the warning is specifically for Vulkan-without-HIP."""
    binary = str(tmp_path / "llama-server")
    _write_managed_llamacpp_config(tmp_path, binary)
    _basic_llamacpp_mocks(mock_httpx)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "HIP/ROCm" not in result.output


@patch("localharness.cli.doctor_cmd._has_amd_gpu", return_value=False)
@patch("localharness.cli.doctor_cmd._llama_server_linked_backends", return_value={"vulkan"})
@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_silent_without_amd_gpu(mock_httpx, mock_backends, mock_amd, tmp_path):
    """A Vulkan-only build is fine on non-AMD hardware (e.g. Intel/NVIDIA Vulkan) — the
    warning is AMD-specific (that's where the HIP/ROCm backend applies)."""
    binary = str(tmp_path / "llama-server")
    _write_managed_llamacpp_config(tmp_path, binary)
    _basic_llamacpp_mocks(mock_httpx)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "HIP/ROCm" not in result.output


@patch("localharness.cli.doctor_cmd._has_amd_gpu", return_value=True)
@patch("localharness.cli.doctor_cmd._llama_server_linked_backends", return_value=None)
@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_silent_when_backend_undeterminable(mock_httpx, mock_backends, mock_amd, tmp_path):
    """None (ldd missing/failed, non-Linux) means "can't tell" — must never be treated as
    evidence of a problem."""
    binary = str(tmp_path / "llama-server")
    _write_managed_llamacpp_config(tmp_path, binary)
    _basic_llamacpp_mocks(mock_httpx)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "HIP/ROCm" not in result.output


@patch("localharness.cli.doctor_cmd._has_amd_gpu", return_value=True)
@patch("localharness.cli.doctor_cmd._llama_server_linked_backends")
@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_skips_amd_check_for_attach_only_llamacpp(mock_httpx, mock_backends, mock_amd, tmp_path):
    """No `server:` block (attach-only endpoint) -> the harness has no binary path to inspect,
    so the check must skip entirely rather than guess."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _basic_llamacpp_mocks(mock_httpx)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "HIP/ROCm" not in result.output
    mock_backends.assert_not_called()


def test_has_amd_gpu_reads_sysfs_pci_vendor(tmp_path, monkeypatch):
    """_has_amd_gpu reads /sys/class/drm/card*/device/vendor for AMD's PCI vendor id
    (0x1002) — no subprocess, so it works even without rocm-smi/lspci installed."""
    from localharness.cli import doctor_cmd

    drm = tmp_path / "drm"
    (drm / "card0" / "device").mkdir(parents=True)
    (drm / "card0" / "device" / "vendor").write_text("0x1002\n")
    monkeypatch.setattr(doctor_cmd, "Path", lambda p: Path(str(drm)) if p == "/sys/class/drm" else Path(p))
    monkeypatch.setattr(doctor_cmd.sys, "platform", "linux")
    assert doctor_cmd._has_amd_gpu() is True


def test_has_amd_gpu_false_for_other_vendor(tmp_path, monkeypatch):
    from localharness.cli import doctor_cmd

    drm = tmp_path / "drm"
    (drm / "card0" / "device").mkdir(parents=True)
    (drm / "card0" / "device" / "vendor").write_text("0x10de\n")  # NVIDIA
    monkeypatch.setattr(doctor_cmd, "Path", lambda p: Path(str(drm)) if p == "/sys/class/drm" else Path(p))
    monkeypatch.setattr(doctor_cmd.sys, "platform", "linux")
    assert doctor_cmd._has_amd_gpu() is False


def test_has_amd_gpu_false_off_linux(monkeypatch):
    from localharness.cli import doctor_cmd

    monkeypatch.setattr(doctor_cmd.sys, "platform", "win32")
    assert doctor_cmd._has_amd_gpu() is False


def test_linked_backends_none_off_linux(monkeypatch):
    from localharness.cli import doctor_cmd

    monkeypatch.setattr(doctor_cmd.sys, "platform", "darwin")
    assert doctor_cmd._llama_server_linked_backends("/usr/bin/anything") is None


def test_linked_backends_none_when_binary_missing(monkeypatch):
    from localharness.cli import doctor_cmd

    monkeypatch.setattr(doctor_cmd.sys, "platform", "linux")
    assert doctor_cmd._llama_server_linked_backends("/no/such/binary") is None


def test_linked_backends_detects_hip_and_vulkan(tmp_path, monkeypatch):
    """Recognizes both the current library names and the old-tree spellings the #144 docs
    fix also had to account for (LLAMA_HIPBLAS/GGML_HIPBLAS -> GGML_HIP rename)."""
    from localharness.cli import doctor_cmd

    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(doctor_cmd.sys, "platform", "linux")

    def fake_run(cmd, capture_output, text, timeout):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7\n"
        return result
    monkeypatch.setattr(doctor_cmd.subprocess, "run", fake_run)
    assert doctor_cmd._llama_server_linked_backends(str(binary)) == {"hip"}


def test_linked_backends_none_when_ldd_fails(tmp_path, monkeypatch):
    from localharness.cli import doctor_cmd

    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(doctor_cmd.sys, "platform", "linux")

    def fake_run(cmd, capture_output, text, timeout):
        raise FileNotFoundError("ldd not found")
    monkeypatch.setattr(doctor_cmd.subprocess, "run", fake_run)
    assert doctor_cmd._llama_server_linked_backends(str(binary)) is None


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_vllm_message_level_exact(mock_httpx, tmp_path):
    """vLLM whose /tokenize answers messages-mode: doctor reports message-level exactness —
    the capability line was previously untested in the PASS direction."""
    _write_config(tmp_path, "vllm", "http://localhost:8000/v1", model="m")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m"}]})
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"count": 3}
    mock_httpx.post.return_value = tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "message-level" in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_vllm_messages_mode_absent_degrades(mock_httpx, tmp_path):
    """An older vLLM without messages-mode /tokenize: content counting stays PASS-exact and
    doctor prints the honest summed-estimate INFO — exit 0, never a failure record."""
    _write_config(tmp_path, "vllm", "http://localhost:8000/v1", model="m")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m"}]})
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"count": 3}
    bad = MagicMock()
    bad.status_code = 400
    bad.json.return_value = {}
    mock_httpx.post.side_effect = (
        lambda url, **kw: bad if "messages" in kw.get("json", {}) else tok
    )

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "no messages mode" in result.output
    assert "message-level (chat template applied server-side)" not in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_llamacpp_apply_template_absent_degrades(mock_httpx, tmp_path):
    """llama-server without /apply-template: /tokenize stays PASS-exact, doctor prints the
    honest INFO about summed message counts — exit 0, no failure appended."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m"}]})
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"tokens": [1, 2]}

    def post(url, **kw):
        if url.endswith("/apply-template"):
            raise RuntimeError("404: no such route on this build")
        return tok
    mock_httpx.post.side_effect = post

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "apply-template not served" in result.output
# --- slot-window check must key on the REAL per-request window, not on slot count -------
# Modern llama.cpp shares ONE KV cache across slots (kv_unified), so --ctx-size is NOT
# divided among them and /props reports the true per-request window. The old check keyed
# purely on `total_slots > 1` and asserted "each request gets ~1/N" — which contradicted
# the very number it printed alongside. Compare the reported window to the budget instead.


def _props_resp(total_slots: int, slot_ctx: int) -> MagicMock:
    r = MagicMock()
    r.json.return_value = {
        "total_slots": total_slots,
        "default_generation_settings": {"n_ctx": slot_ctx},
    }
    return r


def _write_orchestrator(
    tmp_path: Path,
    max_ctx: int,
    overrides: dict[str, int] | None = None,
    model: str = "inherit",
) -> None:
    d = tmp_path / "agents"
    d.mkdir(exist_ok=True)
    body = (
        f"name: orchestrator\nmodel: {model}\nrole: test\n"
        f"context:\n  max_context_tokens: {max_ctx}\n"
    )
    if overrides:
        body += "  model_context_overrides:\n" + "".join(
            f"    {k}: {v}\n" for k, v in overrides.items()
        )
    (d / "orchestrator.yaml").write_text(body)


def _llamacpp_mocks(mock_httpx, *, served: int, slots: int, slot_ctx: int) -> None:
    def _get(url, **kw):
        if url.endswith("/props"):
            return _props_resp(slots, slot_ctx)
        return _models_resp({"data": [{"id": "m", "meta": {"n_ctx": served}}]})
    mock_httpx.get.side_effect = _get
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"tokens": [1]}
    mock_httpx.post.return_value = tok


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_silent_when_slot_window_covers_budget(mock_httpx, tmp_path):
    """kv_unified: 4 slots but each sees the FULL 131072 window, and the budget fits.
    Nothing is wrong, so doctor must not warn (this was the false positive)."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _write_orchestrator(tmp_path, 126_976)
    _llamacpp_mocks(mock_httpx, served=131_072, slots=4, slot_ctx=131_072)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "parallel slots" not in result.output
    assert "1/4" not in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_warns_when_slot_window_starves_budget(mock_httpx, tmp_path):
    """Divided KV: /props reports the small per-slot window. That genuinely starves the
    budget, so the warning must still fire — and quote the real numbers, not a guess.
    budget(28672) sits just under served(32768) so neither the too-high nor the
    under-utilisation check fires — isolating this one."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _write_orchestrator(tmp_path, 28_672)
    _llamacpp_mocks(mock_httpx, served=32_768, slots=4, slot_ctx=8_192)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "8,192" in result.output
    assert "--parallel 1" in result.output


# --- #137: doctor must reconcile the EFFECTIVE budget — the same pin-vs-scalar resolution
# `start` runs — and say WHICH value it checked. The pins feature (#132) and this blindness
# shipped together in 0.12.4: `start` ran on the pin while doctor blessed the scalar. ---


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_reconciles_the_pinned_budget_and_names_the_model(mock_httpx, tmp_path):
    """#137: with a pin for the served model, doctor checks the PINNED number (and says so).
    The scalar (120,000) would EXCEED this window — so a green line here can only come from
    reading the pin, exactly as `start` does."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _write_orchestrator(tmp_path, 120_000, overrides={"m": 56_000})
    _llamacpp_mocks(mock_httpx, served=65_536, slots=1, slot_ctx=65_536)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "56,000" in result.output
    assert "pinned for m" in result.output       # names the value it actually checked
    assert "120,000" not in result.output        # the scalar is NOT this model's budget
    assert "EXCEEDS" not in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_without_a_pin_reports_the_scalar_unchanged(mock_httpx, tmp_path):
    """#137 control: no map (and a map naming a DIFFERENT model) leaves the scalar in charge,
    with the original wording — no phantom 'pinned' attribution."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _write_orchestrator(tmp_path, 56_000, overrides={"some-other-model": 8_000})
    _llamacpp_mocks(mock_httpx, served=65_536, slots=1, slot_ctx=65_536)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Context budget 56,000 fits served window 65,536" in result.output
    assert "pinned" not in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_fails_on_a_pin_that_exceeds_the_served_window(mock_httpx, tmp_path):
    """#137: the check semantics are unchanged — only the number changes. An over-ceiling PIN
    must fail and cite the pinned value (the scalar, 30,000, fits fine and must not mask it)."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _write_orchestrator(tmp_path, 30_000, overrides={"m": 200_000})
    _llamacpp_mocks(mock_httpx, served=65_536, slots=1, slot_ctx=65_536)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "200,000" in result.output
    assert "EXCEEDS" in result.output
    assert "pinned for m" in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_pin_lookup_uses_the_root_agents_own_model(mock_httpx, tmp_path):
    """#137: `start` resolves the served model as agent.model unless 'inherit' — doctor must
    key the pin off the SAME name, or a root agent pinned to its own model reads the wrong
    budget. Here the pin is keyed to the agent's model, not the provider default."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _write_orchestrator(tmp_path, 120_000, overrides={"agent-model": 56_000}, model="agent-model")
    _llamacpp_mocks(mock_httpx, served=65_536, slots=1, slot_ctx=65_536)

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert "56,000" in result.output
    assert "pinned for agent-model" in result.output


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_served_window_follows_the_root_agents_model(mock_httpx, tmp_path):
    """#137 critic repro: multi-model endpoint + root agent on its own model — the served
    window must come from THAT model's entry, not provider.default_model's, or the check
    pairs one model's pin with another model's window (a backwards EXCEEDS verdict: the
    56,000 pin fits agent-model's 131,072 window with room to spare, but was previously
    judged against m's 8,192)."""
    _write_config(tmp_path, "llamacpp", "http://localhost:8080/v1", model="m")
    _write_orchestrator(tmp_path, 120_000, overrides={"agent-model": 56_000}, model="agent-model")

    def _get(url, **kw):
        if url.endswith("/props"):
            return _props_resp(1, 131_072)
        return _models_resp({"data": [
            {"id": "m", "meta": {"n_ctx": 8_192}},
            {"id": "agent-model", "meta": {"n_ctx": 131_072}},
        ]})
    mock_httpx.get.side_effect = _get
    tok = MagicMock()
    tok.status_code = 200
    tok.json.return_value = {"tokens": [1]}
    mock_httpx.post.return_value = tok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert "EXCEEDS" not in result.output          # the backwards verdict is gone
    assert "pinned for agent-model" in result.output
    assert "131,072" in result.output              # judged against the ROOT model's window
    # The honest verdict for a 56k pin on a ~127k usable window is the under-use warning:
    assert "BELOW" in result.output


# ---------------------------------------------------------------------------
# doctor consults the RESOLVER, not the stored config (2026-09-03)
# Found live: a box whose config said provider_type=llamacpp while the server spoke vLLM.
# TokenCounter treats provider_type as a HINT and probes both shapes, so counting was EXACT —
# but doctor branched on the config, sent a llama.cpp-shaped body, and reported
# "token accounting falls back to tiktoken cl100k". The tool you run for reassurance was wrong.
# ---------------------------------------------------------------------------

@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_reports_the_resolved_mode_not_the_configured_one(mock_httpx, tmp_path):
    """Config drifted to llamacpp; the server speaks vLLM. Counting is EXACT, so doctor must
    pass — and must SAY the config is stale rather than inventing a token-accounting failure."""
    _StubCounter.mode_override = "vllm"          # what the counter actually resolved
    _write_config(tmp_path, "llamacpp", "http://localhost:8000/v1", model="m")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m"}]})
    ok = MagicMock(); ok.status_code = 200; ok.json.return_value = {"count": 1}
    mock_httpx.post.return_value = ok

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    out = result.output
    assert "exact" in out.lower(), out
    assert "tiktoken" not in out.lower(), "must not claim a fallback that isn't happening"
    assert "llamacpp" in out and "vllm" in out, "the drift itself must be surfaced"
    assert "tokenize-unreachable" not in out


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_does_not_cascade_a_stale_model_into_a_second_failure(mock_httpx, tmp_path):
    """A stale default_model made doctor report TWO issues that were one: the tokenize probe
    reuses that model name, so it failed for the same reason. Dependent checks must be skipped
    and labelled, or the user chases a phantom."""
    _write_config(tmp_path, "vllm", "http://localhost:8000/v1", model="gone-model")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "actually-served"}]})

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    out = result.output
    assert "Model not found" in out
    assert "not checked" in out.lower(), "the dependent probe must be skipped, not re-failed"
    assert "falls back to tiktoken" not in out.lower()


@patch("localharness.cli.doctor_cmd.httpx")
def test_doctor_quantifies_the_approximate_counting_error(mock_httpx, tmp_path):
    """'approximate' alone is not actionable. Measured vs a real Qwen tokenizer: ~0% on
    English/JSON, ~4% on code, >100% on CJK — the error is content-shaped, so say so."""
    _StubCounter.mode_override = "approximate"
    _write_config(tmp_path, "ollama", "http://localhost:11434", model="m")
    mock_httpx.get.return_value = _models_resp({"models": [{"name": "m"}]})

    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert "CJK" in result.output, result.output
