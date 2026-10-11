"""`recovery_config_dir` (D-20): each strategy answers which config dir a reconnect poll for a server it
launched should pass to `wait_ready`. The three that write a pidfile give the dir `activate` polls with;
LM Studio, whose `activate` polls with None and writes no pidfile, gives None by design.

Hermetic (D-34): only the pure `recovery_config_dir` method is called; nothing probes or launches (LM
Studio's `liveness` is a blocking HTTP call and is never touched here).
"""
from __future__ import annotations

import inspect

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle
from localharness.provider.lifecycle import (
    DaemonStrategy,
    LifecycleStrategy,
    LmsStrategy,
    ManagedVllmStrategy,
    SpawnedProcessStrategy,
    strategy_for,
)

_ACTIVATE_PARAMETERS = [
    "self", "spec", "config_dir", "base_url", "timeout_seconds", "on_poll", "api_key", "peer", "autosense",
]


def _spec(runtime: str, **over) -> ManagedServerConfig:
    base = {
        "vllm": dict(launch="binary", binary="/usr/bin/vllm", model="m", port=8001),
        "llamacpp": dict(launch="binary", binary="/x/llama-server", model="/x/model.gguf", port=8080),
        "ollama": dict(model="qwen2.5:7b", port=11434, gpu=False),
        "lmstudio": dict(model="qwen2.5-0.5b-instruct", binary="/home/u/.lmstudio/bin/lms", port=1234,
                         gpu=False),
    }[runtime]
    base.update(over)
    return ManagedServerConfig(runtime=runtime, **base)


@pytest.mark.parametrize("cls", [ManagedVllmStrategy, SpawnedProcessStrategy, DaemonStrategy])
def test_a_strategy_that_writes_a_pidfile_polls_with_the_config_dir_it_launched_under(cls, tmp_path):
    assert cls().recovery_config_dir(tmp_path) == tmp_path


def test_lm_studio_has_nothing_to_read_a_death_from(tmp_path):
    assert LmsStrategy().recovery_config_dir(tmp_path) is None


@pytest.mark.parametrize(("runtime", "expect_dir"), [
    ("vllm", True), ("llamacpp", True), ("ollama", True), ("lmstudio", False),
])
def test_the_strategy_for_a_spec_answers_by_runtime(runtime, expect_dir, tmp_path):
    answer = strategy_for(_spec(runtime)).recovery_config_dir(tmp_path)
    assert answer == (tmp_path if expect_dir else None)


@pytest.mark.parametrize("cls", [ManagedVllmStrategy, SpawnedProcessStrategy, DaemonStrategy, LmsStrategy])
def test_a_missing_config_dir_stays_missing(cls):
    assert cls().recovery_config_dir(None) is None


@pytest.mark.parametrize("cls", [ManagedVllmStrategy, SpawnedProcessStrategy, DaemonStrategy, LmsStrategy])
def test_every_strategy_still_conforms_to_the_protocol(cls):
    assert isinstance(cls(), LifecycleStrategy)
    assert "recovery_config_dir" in dir(LifecycleStrategy)


@pytest.mark.parametrize("cls", [LifecycleStrategy, ManagedVllmStrategy, SpawnedProcessStrategy,
                                 DaemonStrategy, LmsStrategy])
def test_no_activate_signature_changed(cls):
    assert list(inspect.signature(cls.activate).parameters) == _ACTIVATE_PARAMETERS


def test_the_method_is_on_the_protocol_and_all_four_classes():
    for cls in (LifecycleStrategy, ManagedVllmStrategy, SpawnedProcessStrategy, DaemonStrategy, LmsStrategy):
        assert "recovery_config_dir" in vars(cls), cls.__name__
    assert lifecycle.LifecycleStrategy is LifecycleStrategy
