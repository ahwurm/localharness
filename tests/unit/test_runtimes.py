"""The runtime table answers each column with the answer the if-chain it replaced gave."""
from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import detector, lifecycle, server, tuning, units
from localharness.provider.runtimes import RUNTIMES, row_of


COLUMNS = {
    "strategy", "unit_servable", "label", "key_env", "probe", "price", "facts", "record_fields", "name",
    "gguf", "checkpoint", "path_binary", "exposed_by_default", "scrub", "window_note", "derived_args"}


def _record(runtime, **over):
    return ManagedServerConfig(runtime=runtime, model="m", port=8000, **over)


def test_the_table_has_the_four_runtimes_in_order_and_is_read_only():
    assert list(RUNTIMES) == ["llamacpp", "vllm", "ollama", "lmstudio"]
    with pytest.raises(TypeError):
        RUNTIMES["strata"] = RUNTIMES["vllm"]


@pytest.mark.parametrize("name", ["llamacpp", "vllm", "ollama", "lmstudio"])
def test_row_of_a_known_name_is_its_row(name):
    assert row_of(name) is RUNTIMES[name]


@pytest.mark.parametrize("name", [None, "", 7, "strata", "vLLM"])
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
    (None, False, "attach-only"),
])
def test_what_the_unit_serves_and_what_a_record_is_called(spec, servable, kind):
    assert lifecycle.unit_servable(spec) is servable
    assert lifecycle.unit_kind(spec) == kind


def test_the_server_key_and_label_tables_are_views_over_the_rows():
    assert dict(server.KEY_ENV) == {"vllm": "VLLM_API_KEY", "llamacpp": "LLAMA_API_KEY"}
    assert dict(server.RUNTIME_LABELS) == {
        "vllm": "vLLM", "llamacpp": "llama.cpp", "ollama": "Ollama", "lmstudio": "LM Studio"}
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


def test_the_site_answers_each_row_carries():
    llama, vllm, ollama, lms = RUNTIMES.values()
    assert [r.key_env for r in RUNTIMES.values()] == ["LLAMA_API_KEY", "VLLM_API_KEY", None, None]
    assert [r.gguf for r in RUNTIMES.values()] == [True, False, False, False]
    assert [r.checkpoint for r in RUNTIMES.values()] == [False, True, False, False]
    assert [r.path_binary for r in RUNTIMES.values()] == ["llama-server", None, None, None]
    assert [r.exposed_by_default for r in RUNTIMES.values()] == [False, True, False, False]
    assert [r.window_note for r in RUNTIMES.values()] == [None, None, " · Ollama decides the window", ""]
    assert vllm.facts() is tuning.MODEL_TYPE_PARSERS and llama.facts is None
    assert list(lms.derived_args) == ["--gpu"] and not ollama.derived_args
