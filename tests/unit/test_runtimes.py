"""The runtime table answers each column with the answer the if-chain it replaced gave."""
from __future__ import annotations

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, units
from localharness.provider.runtimes import RUNTIMES, row_of


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
