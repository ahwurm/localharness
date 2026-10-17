"""doctor's Window row and its far-below branch (DOCTOR-01, DOCTOR-02, AUTO-02 "doctor always").

`probe_window` is faked at its seam for the row's numbers; one test drives the real probe through
test_doctor_cmd's httpx router (the autouse fixture imported below). Nothing here starts a server,
calls nvidia-smi or makes a real HTTP request.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from localharness.agent.context import WindowProbe, response_reserve
from localharness.cli.app import app
from localharness.provider.autosense import FreeMemory
from localharness.provider.speed_stats import default_speed_stats_path
from tests.unit.gguf_support import write_gguf
from tests.unit.test_doctor_cmd import (  # noqa: F401  (autouse fixtures, imported for their effect)
    _models_resp,
    _route_the_window_probe,
    _stub_token_counter,
    _write_orchestrator,
)

runner = CliRunner()

GIB = 2**30
# Header A (the tracer's hand-computed shape): 64 layers x 8 kv heads x 128 dim, trained 262,144.
HEADER_A = {
    "general.architecture": "qwen3",
    "qwen3.block_count": 64,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128,
    "qwen3.context_length": 262144,
}
# The 0.18 launch: the tracer's hand-computed f16 numbers hold only when the launch runs f16 (WR-06).
F16 = "autosense:\n  kv_cache_type: f16\n"
ROW = re.compile(r"Window: served (\S+) · trained (\S+) · fits (\S+)")


@pytest.fixture
def mock_httpx(monkeypatch):
    """doctor_cmd's httpx, answering the model list; the probe itself is faked per test."""
    import localharness.cli.doctor_cmd as doctor

    mock = MagicMock()
    mock.get.return_value = _models_resp({"data": [{"id": "m"}]})
    monkeypatch.setattr(doctor, "httpx", mock)
    return mock


def _config(tmp_path: Path, *, provider_type: str = "llamacpp", server: str = "",
            autosense: str = "", api_key: str | None = None) -> None:
    key = f"  api_key: {api_key}\n" if api_key else ""
    (tmp_path / "config.yaml").write_text(
        'version: "1"\n'
        "provider:\n"
        f"  provider_type: {provider_type}\n"
        "  base_url: http://localhost:8080/v1\n"
        "  default_model: m\n"
        "  available_models:\n"
        "    - m\n"
        "  supports_function_calling: true\n"
        "  timeout_seconds: 600.0\n" + key + server + autosense
    )


def _llamacpp_server(model: Path | str, *, runtime: str = "llamacpp", extra: str = "") -> str:
    return (
        "server:\n"
        f"  runtime: {runtime}\n"
        "  binary: /nonexistent/llama-server\n"
        f"  model: {model}\n"
        "  port: 8080\n" + extra
    )


def _doctor(tmp_path: Path):
    return runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])


def _flat(output: str) -> str:
    """The output with rich's line wrapping undone, for a message that is longer than a line."""
    return " ".join(output.split())


def _row(output: str) -> tuple[str, str, str]:
    rows = ROW.findall(output)
    assert len(rows) == 1, output
    return rows[0]


def _memory(free_gib: float = 6.0):
    return patch("localharness.provider.autosense.free_memory",
                 return_value=FreeMemory(int(free_gib * GIB), int(32.6 * GIB), "nvidia-smi"))


def _probe(served, trained):
    return patch("localharness.agent.context.probe_window", return_value=WindowProbe(served, trained))


def _gguf_a(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "a.gguf", HEADER_A)


# --- the tracer: the row, the single probe call, the ledger left alone ---------------------------


def test_a_llamacpp_record_prints_the_attached_fit_once_and_never_writes_the_ledger(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(_gguf_a(tmp_path)), autosense=F16)
    with _probe(65536, 262144), _memory(6.0), \
            patch("localharness.provider.autosense.weights_bytes", return_value=int(18.5 * GIB)):
        result = _doctor(tmp_path)
    assert "Window: served 65,536 · trained 262,144 · fits 90,112" in result.output
    assert _row(result.output) == ("65,536", "262,144", "90,112")  # exactly once
    assert not default_speed_stats_path(tmp_path).exists()


def test_the_probe_is_called_once_with_the_root_agents_model(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(_gguf_a(tmp_path)), autosense=F16)
    with _probe(65536, 262144) as probe, _memory():
        _doctor(tmp_path)
    probe.assert_called_once_with("http://localhost:8080/v1", "m", "llamacpp")


def test_an_unreported_window_prints_dashes_and_the_not_reported_line(tmp_path, mock_httpx):
    _config(tmp_path)
    _write_orchestrator(tmp_path, 28_672)
    with _probe(None, None), _memory():
        result = _doctor(tmp_path)
    assert "Window: served — · trained — · fits —" in result.output
    assert "Served max_model_len not reported" in result.output


def test_an_unreported_served_keeps_the_header_trained_and_has_no_fit(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(_gguf_a(tmp_path)), autosense=F16)
    with _probe(None, None), _memory():
        result = _doctor(tmp_path)
    assert "Window: served — · trained 262,144 · fits —" in result.output


def test_an_attach_only_config_has_no_fit(tmp_path, mock_httpx):
    _config(tmp_path)  # no server: block
    with _probe(65536, 262144):
        result = _doctor(tmp_path)
    assert "Window: served 65,536 · trained 262,144 · fits —" in result.output


def test_a_vllm_record_has_no_fit(tmp_path, mock_httpx):
    _config(tmp_path, provider_type="vllm", server=_llamacpp_server("org/model", runtime="vllm"))
    with _probe(65536, 65536), _memory():
        result = _doctor(tmp_path)
    assert "Window: served 65,536 · trained 65,536 · fits —" in result.output


def test_the_row_prints_even_when_the_model_is_not_served(tmp_path, mock_httpx):
    _config(tmp_path)
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "other"}]})
    with _probe(65536, 262144):
        result = _doctor(tmp_path)
    assert result.exit_code == 1
    assert "Model not found" in result.output
    assert _row(result.output) == ("65,536", "262,144", "—")


def test_a_trained_the_probe_did_not_report_comes_from_the_header(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(_gguf_a(tmp_path)), autosense=F16)
    with _probe(65536, None), _memory(6.0):
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "262,144", "90,112")


# --- the key rides the probe only when the launched server requires it ---------------------------


def test_a_server_that_requires_its_key_gets_it_on_the_probe(tmp_path, mock_httpx):
    _config(tmp_path, api_key="sekrit", server=_llamacpp_server(_gguf_a(tmp_path), extra="  require_api_key: true\n"))
    with _probe(65536, 262144) as probe, _memory():
        _doctor(tmp_path)
    probe.assert_called_once_with("http://localhost:8080/v1", "m", "llamacpp", api_key="sekrit")


def test_a_server_that_does_not_require_it_gets_no_key(tmp_path, mock_httpx):
    _config(tmp_path, api_key="sekrit", server=_llamacpp_server(_gguf_a(tmp_path)))
    with _probe(65536, 262144) as probe, _memory():
        _doctor(tmp_path)
    probe.assert_called_once_with("http://localhost:8080/v1", "m", "llamacpp")


# --- D-39: a llama-server behind a non-llamacpp provider_type keeps its window -------------------


def test_a_vllm_provider_over_a_llama_server_reads_meta_n_ctx(tmp_path, mock_httpx):
    _config(tmp_path, provider_type="vllm")
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "m", "meta": {"n_ctx": 32768}}]})
    result = _doctor(tmp_path)  # the real probe, routed to mock_httpx by the autouse fixture
    assert _row(result.output)[0] == "32,768"
    assert "not reported" not in result.output.lower()


# --- the far-below branch: response_reserve(served) and 1 - notify_ratio -------------------------


def _threshold(served: int, notify_ratio: float = 0.25) -> float:
    return (served - response_reserve(served)) * (1 - notify_ratio)


def test_the_tracer_reserve_and_threshold_numbers():
    assert response_reserve(65536) == 4096
    assert _threshold(65536) == 46080


@pytest.mark.parametrize("served", [65536, 131072])
def test_adjacency_at_the_threshold_is_not_far_below_one_token_less_is(tmp_path, mock_httpx, served):
    at = int(_threshold(served))
    assert at == _threshold(served)  # both served windows give a whole-token threshold
    _config(tmp_path)
    _write_orchestrator(tmp_path, at)
    with _probe(served, 262144):
        passing = _doctor(tmp_path)
    assert "far BELOW" not in passing.output
    assert f"fits served window {served:,}" in _flat(passing.output)
    _write_orchestrator(tmp_path, at - 1)
    with _probe(served, 262144):
        failing = _doctor(tmp_path)
    assert failing.exit_code == 1
    assert "far BELOW" in _flat(failing.output)
    assert "wasting more than 25% of the window" in _flat(failing.output)
    assert f"(e.g. {served - response_reserve(served):,})" in _flat(failing.output)


def test_the_threshold_follows_notify_ratio(tmp_path, mock_httpx):
    _config(tmp_path, autosense="autosense:\n  notify_ratio: 0.5\n")
    assert _threshold(65536, 0.5) == 30720
    _write_orchestrator(tmp_path, 30720)
    with _probe(65536, 262144):
        passing = _doctor(tmp_path)
    assert "far BELOW" not in passing.output
    _write_orchestrator(tmp_path, 30719)
    with _probe(65536, 262144):
        failing = _doctor(tmp_path)
    assert "wasting more than 50% of the window" in _flat(failing.output)


def test_the_failure_code_is_unchanged(tmp_path, mock_httpx):
    _config(tmp_path)
    _write_orchestrator(tmp_path, 1_000)
    with _probe(65536, 262144):
        result = _doctor(tmp_path)
    assert "1 issue(s) found" in result.output


def test_no_constant_is_left_in_doctor():
    import localharness.cli.doctor_cmd as doctor

    source = Path(doctor.__file__).read_text(encoding="utf-8")
    assert "4_096" not in source
    assert "0.75" not in source
    assert "reserve = response_reserve(served)" in source
    assert "1 - harness.autosense.notify_ratio" in source


# --- hostile and missing model files -------------------------------------------------------------


def test_a_fifo_model_path_neither_hangs_doctor_nor_gives_a_fit(tmp_path, mock_httpx):
    fifo = tmp_path / "model.gguf"
    os.mkfifo(fifo)
    _config(tmp_path, server=_llamacpp_server(fifo))
    result: list = []

    def _run() -> None:
        with _probe(65536, None), _memory():
            result.append(_doctor(tmp_path))

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "doctor blocked on a FIFO model path"
    assert _row(result[0].output) == ("65,536", "—", "—")


def test_a_missing_gguf_is_not_computed_and_not_a_traceback(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(tmp_path / "gone.gguf"))
    with _probe(65536, 262144), _memory():
        result = _doctor(tmp_path)
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert _row(result.output) == ("65,536", "262,144", "—")


def test_a_file_that_is_not_a_gguf_is_not_computed(tmp_path, mock_httpx):
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    _config(tmp_path, server=_llamacpp_server(junk))
    with _probe(65536, 262144), _memory():
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "262,144", "—")


# --- encoding and idempotency --------------------------------------------------------------------


def test_every_number_carries_thousands_separators(tmp_path, mock_httpx):
    _config(tmp_path)
    with _probe(131072, 1048576):
        result = _doctor(tmp_path)
    assert _row(result.output) == ("131,072", "1,048,576", "—")


def test_doctor_twice_prints_the_row_twice_and_leaves_no_ledger(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(_gguf_a(tmp_path)), autosense=F16)
    ledger = default_speed_stats_path(tmp_path)
    with _probe(65536, 262144), _memory(6.0):
        first = _doctor(tmp_path)
        second = _doctor(tmp_path)
    assert _row(first.output) == _row(second.output) == ("65,536", "262,144", "90,112")
    assert not ledger.exists()


# --- more edges: what makes the fit "not computed" without making doctor fail --------------------


def test_a_directory_as_the_model_path_is_not_computed(tmp_path, mock_httpx):
    folder = tmp_path / "model.gguf"
    folder.mkdir()
    _config(tmp_path, server=_llamacpp_server(folder))
    with _probe(65536, 262144), _memory():
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "262,144", "—")


def test_a_partial_offload_is_not_computed(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(
        _gguf_a(tmp_path), extra="  extra_args:\n    - -ngl\n    - '20'\n"))
    with _probe(65536, 262144), _memory():
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "262,144", "—")


def test_hand_placed_tensors_are_not_computed(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(
        _gguf_a(tmp_path), extra="  extra_args:\n    - -ot\n    - 'exps=CPU'\n"))
    with _probe(65536, 262144), _memory():
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "262,144", "—")


def test_ill_typed_header_values_are_not_computed_and_not_an_unreachable_endpoint(tmp_path, mock_httpx):
    """WR-01: `block_count` declared as a string with `-ngl` in `extra_args` raised a TypeError that the
    endpoint `except` reported as an unreachable endpoint."""
    header = {**HEADER_A, "qwen3.block_count": "abc", "qwen3.context_length": [1]}
    _config(tmp_path, server=_llamacpp_server(
        write_gguf(tmp_path / "bad.gguf", header), extra="  extra_args:\n    - -ngl\n    - '20'\n"))
    with _probe(65536, None), _memory():
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "\u2014", "\u2014")
    assert "unreachable" not in result.output.lower()


def test_a_header_without_its_head_dimension_is_not_computed(tmp_path, mock_httpx):
    header = {k: v for k, v in HEADER_A.items() if not k.endswith("key_length")}
    _config(tmp_path, server=_llamacpp_server(write_gguf(tmp_path / "b.gguf", header)))
    with _probe(65536, 262144), _memory():
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "262,144", "—")


def test_unknown_free_memory_is_not_computed(tmp_path, mock_httpx):
    _config(tmp_path, server=_llamacpp_server(_gguf_a(tmp_path)), autosense=F16)
    with _probe(65536, 262144), patch("localharness.provider.autosense.free_memory", return_value=None):
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "262,144", "—")


def test_an_existing_ledger_is_left_byte_for_byte_and_mtime_for_mtime(tmp_path, mock_httpx):
    ledger = default_speed_stats_path(tmp_path)
    ledger.write_text('{"keep": "me"}\n', encoding="utf-8")
    os.utime(ledger, (1_700_000_000, 1_700_000_000))
    before = (ledger.read_bytes(), ledger.stat().st_mtime_ns)
    _config(tmp_path, server=_llamacpp_server(_gguf_a(tmp_path)), autosense=F16)
    with _probe(65536, 262144), _memory(6.0):
        _doctor(tmp_path)
        _doctor(tmp_path)
    assert (ledger.read_bytes(), ledger.stat().st_mtime_ns) == before


def test_the_row_comes_before_the_reconciliation_verdict(tmp_path, mock_httpx):
    _config(tmp_path)
    _write_orchestrator(tmp_path, 1_000)
    with _probe(65536, 262144):
        result = _doctor(tmp_path)
    assert result.output.index("Window: served") < result.output.index("far BELOW")


# --- the row is priced as the launch is (WR-06) ---------------------------------------------------

HEADER_MTP = {**HEADER_A, "qwen3.nextn_predict_layers": 1}


def test_the_row_prices_the_derived_kv_type_and_the_draft_share(tmp_path, mock_httpx):
    from localharness.provider.gguf_header import read_header
    from localharness.config.models import AutosenseConfig
    from localharness.provider import autosense as _auto

    gguf = write_gguf(tmp_path / "mtp.gguf", HEADER_MTP)
    _config(tmp_path, server=_llamacpp_server(gguf))
    header = _auto.fit_header(read_header(str(gguf), _auto.FIT_KEYS))
    expected = _auto.fit(header, 6 * GIB, AutosenseConfig(), 65536, cache_type_k="q8_0",
                         cache_type_v="q8_0", weights=None, draft_share=1 + 1 / 64)
    f16 = _auto.fit(header, 6 * GIB, AutosenseConfig(), 65536, cache_type_k="f16", cache_type_v="f16")
    assert expected != f16
    with _probe(65536, 262144), _memory(6.0):
        result = _doctor(tmp_path)
    assert _row(result.output) == ("65,536", "262,144", f"{expected:,}")
