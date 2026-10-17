"""The runtime table answers each column with the answer the if-chain it replaced gave."""
from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, detector, lifecycle, server, tuning, units
from localharness.provider.tuning import Tuned
from localharness.provider.runtimes import RUNTIMES, row_of


COLUMNS = {
    "strategy", "unit_servable", "label", "key_env", "probe", "price", "argv", "facts", "record_fields", "name",
    "gguf", "checkpoint", "path_binary", "exposed_by_default", "scrub", "window_note", "derived_args"}


def _record(runtime, **over):
    return ManagedServerConfig(**{"runtime": runtime, "model": "m", "port": 8000, **over})


def test_the_table_has_the_five_runtimes_in_order_and_is_read_only():
    assert list(RUNTIMES) == ["llamacpp", "vllm", "ollama", "lmstudio", "strata"]
    with pytest.raises(TypeError):
        RUNTIMES["other"] = RUNTIMES["vllm"]


@pytest.mark.parametrize("name", ["llamacpp", "vllm", "ollama", "lmstudio", "strata"])
def test_row_of_a_known_name_is_its_row(name):
    assert row_of(name) is RUNTIMES[name]


@pytest.mark.parametrize("name", [None, "", 7, "vLLM"])
def test_row_of_anything_else_is_none(name):
    assert row_of(name) is None


@pytest.mark.parametrize("spec,cls", [
    (_record("llamacpp", binary="/x/llama-server"), lifecycle.SpawnedProcessStrategy),
    (_record("vllm", launch="binary", binary="/x/vllm"), lifecycle.ManagedVllmStrategy),
    (_record("vllm", launch="docker", docker_image="img"), lifecycle.ManagedVllmStrategy),
    (_record("ollama"), lifecycle.DaemonStrategy),
    (_record("lmstudio"), lifecycle.LmsStrategy),
])
def test_strategy_for_returns_the_class_of_the_row(spec, cls):
    assert type(lifecycle.strategy_for(spec, unit=None)) is cls


@pytest.mark.parametrize("spec,wrapped", [
    (_record("llamacpp", binary="/x/llama-server"), True),
    (_record("vllm", launch="binary", binary="/x/vllm"), True),
    (_record("vllm", launch="docker", docker_image="img"), False),
    (_record("ollama"), False),
    (_record("lmstudio"), False),
])
def test_only_the_own_unit_and_a_servable_record_wrap_in_the_unit_strategy(spec, wrapped):
    own = units.UnitRuntimeState(None, 0, own=True)
    raw = units.UnitRuntimeState(None, 0)
    assert isinstance(lifecycle.strategy_for(spec, unit=own), lifecycle.UnitStrategy) is wrapped
    assert not isinstance(lifecycle.strategy_for(spec, unit=raw), lifecycle.UnitStrategy)


@pytest.mark.parametrize("spec,servable,kind", [
    (_record("llamacpp", binary="/x/llama-server"), True, "llama.cpp"),
    (_record("vllm", launch="binary", binary="/x/vllm"), True, "vLLM binary"),
    (_record("vllm", launch="docker", docker_image="img"), False, "docker"),
    (_record("ollama"), False, "Ollama"),
    (_record("lmstudio"), False, "LM Studio"),
    (_record("strata", engine_dir="/e", engine_config="c.json", served_name="s"), True, "Strata"),
    (None, False, "attach-only"),
])
def test_what_the_unit_serves_and_what_a_record_is_called(spec, servable, kind):
    assert lifecycle.unit_servable(spec) is servable
    assert lifecycle.unit_kind(spec) == kind


def test_the_server_key_and_label_tables_are_views_over_the_rows():
    assert dict(server.KEY_ENV) == {"vllm": "VLLM_API_KEY", "llamacpp": "LLAMA_API_KEY", "strata": "STRATA_API_KEY"}
    assert dict(server.RUNTIME_LABELS) == {
        "vllm": "vLLM", "llamacpp": "llama.cpp", "ollama": "Ollama", "lmstudio": "LM Studio", "strata": "Strata"}
    with pytest.raises(TypeError):
        server.KEY_ENV["ollama"] = "X"
    with pytest.raises(TypeError):
        server.RUNTIME_LABELS["ollama"] = "X"


def test_the_config_literal_and_the_detector_type_name_the_table_keys():
    runtime = ManagedServerConfig.model_fields["runtime"].annotation
    assert set(typing.get_args(runtime)) == set(RUNTIMES)
    assert set(typing.get_args(detector.ProviderType)) - {"unknown"} == set(RUNTIMES)


def test_every_row_answers_every_column():
    for row in RUNTIMES.values():
        assert {f.name for f in dataclasses.fields(row)} == COLUMNS
        assert all(callable(c) for c in (row.strategy, row.unit_servable, row.label, row.probe))
        assert row.price is None or callable(row.price)
        assert row.facts is None or isinstance(row.facts(), Mapping)
        assert isinstance(row.name, str) and isinstance(row.scrub, tuple)
        assert isinstance(row.record_fields, tuple) and isinstance(row.derived_args, Mapping)


def test_scrub_joins_to_the_line_the_script_shows():
    for name in ("llamacpp", "vllm"):
        assert "/".join(RUNTIMES[name].scrub) == tuning.SCRUBBED_SHOWN[name]


def test_record_fields_are_the_runtime_only_fields():
    assert RUNTIMES["llamacpp"].record_fields == (
        "mmproj", "served_name", "cache_type_k", "cache_type_v", "cache_ram", "api_key_file", "env",
        "config_ini", "speculative")
    assert RUNTIMES["vllm"].record_fields == ("reasoning_parser", "tool_call_parser")
    assert RUNTIMES["ollama"].record_fields == RUNTIMES["lmstudio"].record_fields == ()
    assert RUNTIMES["strata"].record_fields == ("mmproj", "served_name", "engine_dir", "engine_config")


def test_the_site_answers_each_row_carries():
    llama, vllm, ollama, lms, strata = RUNTIMES.values()
    assert [r.key_env for r in RUNTIMES.values()] == ["LLAMA_API_KEY", "VLLM_API_KEY", None, None, "STRATA_API_KEY"]
    assert [r.gguf for r in RUNTIMES.values()] == [True, False, False, False, True]
    assert [r.checkpoint for r in RUNTIMES.values()] == [False, True, False, False, False]
    assert [r.path_binary for r in RUNTIMES.values()] == ["llama-server", None, None, None, None]
    assert [r.exposed_by_default for r in RUNTIMES.values()] == [False, True, False, False, False]
    assert [r.window_note for r in RUNTIMES.values()] == [None, None, " · Ollama decides the window", "", None]
    assert vllm.facts() is tuning.MODEL_TYPE_PARSERS and llama.facts is None
    assert dict(strata.facts()) == {} and strata.price is None and not strata.derived_args
    assert list(lms.derived_args) == ["--gpu"] and not ollama.derived_args


GIB = 2**30
HEADER = {"block_count": 64, "head_count_kv": 8, "key_length": 128, "context_length": 262144}


def _sensed(header=HEADER, free=40 * GIB, weights=18 * GIB):
    mem = autosense.FreeMemory(free, 48 * GIB, "nvidia-smi")
    return lifecycle.Sensed("llamacpp", None, header, "gpu", mem, weights, {"MemAvailable": 90 * GIB, "MemTotal": 128 * GIB})


def test_price_llamacpp_prices_the_fit_and_the_cache_ram_through_the_row():
    spec, knobs = _record("llamacpp", binary="/x/llama-server"), AutosenseConfig()
    tuned = Tuned(cache_type="q8_0")
    got = RUNTIMES["llamacpp"].price(spec, _sensed(), tuned, knobs)
    fits = autosense.fit(HEADER, 40 * GIB, knobs, cache_type_k="q8_0", cache_type_v="q8_0", weights=18 * GIB) // 256 * 256
    ram = autosense.cache_ram_mib(90 * GIB, 128 * GIB, 18 * GIB, "gpu", [], knobs)
    assert isinstance(got, autosense.Price) and got.refusal is None
    assert (got.context, got.cache_ram, got.full_fit) == (fits, ram, fits) and fits
    assert (got.report.mode, got.report.asked, got.report.fits, got.report.trained) == ("fit", fits, fits, 262144)


def test_price_llamacpp_with_weights_beyond_the_free_memory_is_a_refusal():
    got = autosense.price_llamacpp(
        _record("llamacpp", binary="/x/llama-server"), _sensed(free=18 * GIB), Tuned(), AutosenseConfig())
    assert (got.context, got.full_fit, got.report.fits) == (None, None, None)
    assert got.refusal.endswith("it needs 18.0 GiB of GPU memory and 18.0 GiB is free — stop what holds the GPU, or pick a smaller quant.")


@pytest.mark.parametrize("over,mode", [({"context": 4096}, "record"), ({"extra_args": ["-c", "8192"]}, "extra_args")])
def test_price_llamacpp_sets_no_context_when_the_record_or_extra_args_did(over, mode):
    got = autosense.price_llamacpp(
        _record("llamacpp", binary="/x/llama-server", **over), _sensed(), Tuned(), AutosenseConfig())
    assert got.context is None and got.full_fit is None and got.report.mode == mode


@pytest.mark.parametrize("name", ["ollama", "lmstudio"])
def test_a_row_without_a_price_plans_no_window(name):
    assert RUNTIMES[name].price is None


@pytest.mark.parametrize("name", ["vllm", "ollama", "lmstudio"])
def test_only_a_gguf_row_has_a_fit(name):
    assert autosense.fit(HEADER, 40 * GIB, AutosenseConfig(), weights=18 * GIB, runtime=name) is None
    assert autosense.fit(HEADER, 40 * GIB, AutosenseConfig(), weights=18 * GIB, runtime="llamacpp")


@pytest.mark.parametrize("spec,tokens", [
    (_record("llamacpp", binary="/x/llama-server", model="/x/m.gguf", served_name="qwen"), ("-a", "qwen")),
    (_record("llamacpp", binary="/x/llama-server", model="/x/m.gguf"), ("-m", "/x/m.gguf")),
    (_record("vllm", launch="binary", binary="/x/vllm", model="org/m"), ("org/m",)),
    (_record("vllm", launch="docker", docker_image="img", model="org/m"), ("org/m",)),
    (_record("ollama", model="qwen"), ()),
])
def test_record_identity_adds_the_records_own_tokens(spec, tokens):
    base = lifecycle.strategy_for(spec, unit=None).launch_identity(spec)
    got = lifecycle.record_identity(spec)
    assert got.argv_contains == base.argv_contains + tokens
    assert (got.argv_one_of != ()) is (spec.runtime == "vllm" and spec.launch == "docker")


def test_lm_studio_has_no_record_identity():
    assert lifecycle.record_identity(_record("lmstudio")) is None


BASE_ENV = {"LLAMA_ARG_CTX_SIZE": "1", "LLAMA_API_KEY": "k", "MTMD_BACKEND_DEVICE": "d", "VLLM_API_KEY": "v",
            "VLLM_LOGGING_LEVEL": "x", "HF_TOKEN": "t", "PATH": "/bin"}


@pytest.mark.parametrize("runtime,dropped", [
    ("llamacpp", ["LLAMA_API_KEY", "LLAMA_ARG_CTX_SIZE", "MTMD_BACKEND_DEVICE"]),
    ("vllm", ["VLLM_API_KEY"]), ("ollama", []), ("lmstudio", []), ("unknown", [])])
def test_child_environment_drops_the_rows_own_namespace_only(runtime, dropped):
    env, got = server.child_environment(runtime, BASE_ENV)
    assert got == dropped
    assert (env is None) is not bool(dropped)


@pytest.mark.parametrize("runtime,var", [("llamacpp", "LLAMA_API_KEY"), ("vllm", "VLLM_API_KEY")])
def test_child_environment_sets_the_rows_key_variable_last(runtime, var):
    env, _ = server.child_environment(runtime, BASE_ENV, record_env={var: "other"}, api_key="held")
    assert env[var] == "held"


@pytest.mark.parametrize("runtime", ["ollama", "lmstudio", "unknown"])
def test_a_runtime_without_a_key_variable_refuses_a_key(runtime):
    with pytest.raises(ValueError):
        server.child_environment(runtime, BASE_ENV, api_key="held")


@pytest.mark.parametrize("runtime", ["ollama", "lmstudio"])
def test_launch_decision_binds_loopback_for_a_runtime_with_no_key(runtime):
    assert server.launch_decision(_record(runtime, bind_all=True), False) == server.LaunchDecision(None, True, False, None)


def test_launch_decision_refuses_a_llamacpp_record_without_a_binary():
    spec = _record("llamacpp", binary="/x/llama-server")
    object.__setattr__(spec, "binary", None)
    assert server.launch_decision(spec, False).refusal == "no llama-server — set server.binary"
    assert server.launch_decision(spec, False, peer="p").refusal == (
        "no llama-server — set extra_endpoints[p].lifecycle.binary")


@pytest.mark.parametrize("spec,verdict", [
    (_record("llamacpp", binary="/x/llama-server"), False),
    (_record("llamacpp", binary="/x/llama-server", bind_all=True), True),
    (_record("vllm", launch="binary", binary="/x/vllm"), False),
    (_record("vllm", launch="binary", binary="/x/vllm", bind_all=True), True),
    (_record("vllm", launch="docker", docker_image="img"), False),
    (_record("vllm", launch="docker", docker_image="img", bind_all=True), True),
    (_record("ollama"), False), (_record("lmstudio", bind_all=True), False)])
def test_effective_posture_exposure_per_runtime(spec, verdict):
    decision = server.launch_decision(spec, False)
    assert server.effective_posture(spec, keyed=False, decision=decision)[0] is verdict


def test_the_argv_column_is_what_serve_command_builds_per_runtime():
    llama = _record("llamacpp", binary="/x/llama-server", model="/x/m.gguf")
    assert server.serve_command(llama, context=4096)[:3] == ["/x/llama-server", "-m", "/x/m.gguf"]
    assert server.serve_command(llama, context=4096)[-2:] == ["--host", "127.0.0.1"]
    assert server.serve_command(_record("ollama"))[:3] == ["env", "OLLAMA_HOST=127.0.0.1:8000", "OLLAMA_KEEP_ALIVE=-1"]
    binary = _record("vllm", launch="binary", binary="/x/vllm", model="org/m")
    assert server.serve_command(binary, reasoning_parser="r", tool_call_parser="t") == [
        "/x/vllm", "serve", "org/m", "--port", "8000", "--reasoning-parser", "r", "--tool-call-parser", "t",
        "--enable-auto-tool-choice", "--host", "127.0.0.1"]
    lms = _record("lmstudio", binary="/x/lms", model="org/m")
    assert server.serve_command(lms)[:3] == ["/x/lms", "serve", "org/m"]


def test_tuning_reads_the_rows_for_checkpoint_and_parsers():
    vllm = _record("vllm", launch="binary", binary="/x/vllm", model="/nonexistent/ckpt")
    assert tuning.read_model_config(_record("llamacpp", binary="/x/llama-server")) is None
    assert tuning.read_model_config(_record("ollama")) is None
    assert tuning.read_model_config(vllm) is None
    parsers = tuning.MODEL_TYPE_PARSERS["qwen3"]
    got = tuning.derive(vllm, {}, AutosenseConfig(), model_config={"model_type": "qwen3"})
    assert (got.reasoning_parser, got.tool_call_parser) == parsers
    assert tuning.derive(_record("ollama"), {}, AutosenseConfig(), model_config={"model_type": "qwen3"}) == Tuned()
    assert tuning.derive(_record("lmstudio"), {}, AutosenseConfig()) == Tuned()


@pytest.mark.parametrize("name,url,body,served", [
    ("llamacpp", "http://h/v1/models", {"data": [{"id": "m", "meta": {"n_ctx": 4096, "n_ctx_train": 8192}}]}, 4096),
    ("vllm", "http://h/v1/models", {"data": [{"id": "m", "max_model_len": 2048}]}, 2048),
    ("ollama", "http://h/api/ps", {"models": [{"name": "m", "context_length": 1024}]}, 1024),
    ("lmstudio", "http://h/api/v0/models", {"data": [{"id": "m", "loaded_context_length": 512}]}, 512)])
def test_a_rows_probe_takes_root_model_and_headers(monkeypatch, name, url, body, served):
    import httpx

    class Reply:
        def json(self):
            return body if url in seen else {}

    seen: list[str] = []
    monkeypatch.setattr(httpx, "get", lambda u, **kw: (seen.append(u), Reply())[1])
    assert RUNTIMES[name].probe("http://h/v1", "m", {}).served == served


@pytest.mark.parametrize("runtime,tail", [("ollama", " · Ollama decides the window"), ("lmstudio", "")])
def test_window_lines_takes_the_rows_note(runtime, tail):
    report = autosense.WindowReport(
        served=4096, trained=8192, fits=None, free_bytes=None, total_bytes=None, weights_bytes=None,
        asked=None, mode="not-computed", runtime=runtime)
    assert autosense.window_lines(report, AutosenseConfig()) == [f"window: served 4,096 · trained 8,192{tail}"]
