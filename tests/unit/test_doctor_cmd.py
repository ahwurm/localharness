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
    # probed with llama.cpp's {content} shape, not vLLM's {model,prompt}
    first_kwargs = mock_httpx.post.call_args_list[0].kwargs
    assert "content" in first_kwargs.get("json", {})


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
