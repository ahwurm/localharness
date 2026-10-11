"""`provider.reconnect_wait_seconds`: the longest the loop waits for the model server to come back
after a dropped connection. Default 1800.0, bounded, `0` = off, settable from a workspace (it only
lengthens or shortens a wait for a dead server), mirrored off-by-default on LLMConfig, and distinct
from `inference_queue_wait_seconds`. Hermetic: every config dir is under tmp_path."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from localharness.config.loader import is_harness_global_only
from localharness.config.models import ProviderConfig
from localharness.provider.client import LLMConfig, _QueueWaitState
from tests.unit.test_floor_machine_level import _layers, _warnings

KEY = "provider.reconnect_wait_seconds"


def _pc(**kw) -> ProviderConfig:
    return ProviderConfig(
        provider_type="vllm", base_url="http://127.0.0.1:8000/v1", default_model="m", **kw)


def test_default_is_half_an_hour():
    assert _pc().reconnect_wait_seconds == 1800.0


@pytest.mark.parametrize("ok", [0, 86400])
def test_boundaries_validate(ok):
    assert _pc(reconnect_wait_seconds=ok).reconnect_wait_seconds == float(ok)


@pytest.mark.parametrize("bad", [-0.1, 86400.1])
def test_out_of_range_is_refused(bad):
    with pytest.raises(ValidationError):
        _pc(reconnect_wait_seconds=bad)


def test_the_stored_bound_is_a_float_and_never_rounded():
    v = _pc(reconnect_wait_seconds=30).reconnect_wait_seconds
    assert v == 30.0 and isinstance(v, float)
    assert _pc(reconnect_wait_seconds=45.5).reconnect_wait_seconds == 45.5


@pytest.mark.parametrize("value", [1800.0, 45.5])
def test_dump_and_revalidate_keeps_the_value(value):
    cfg = _pc(reconnect_wait_seconds=value)
    assert ProviderConfig.model_validate(cfg.model_dump()).reconnect_wait_seconds == value


def test_description_says_off_liveness_and_is_not_the_queue_wait():
    desc = ProviderConfig.model_fields["reconnect_wait_seconds"].description
    assert "0 turns the wait off" in desc
    assert "liveness" in desc
    assert "inference_queue_wait_seconds" in desc


def test_a_workspace_may_set_it(tmp_path):
    assert is_harness_global_only(KEY) is False
    loader, _ = _layers(tmp_path, ws_cfg={"provider": {"reconnect_wait_seconds": 45}})
    assert loader.load_harness().provider.reconnect_wait_seconds == 45.0
    assert _warnings(loader) == []


def test_without_a_workspace_value_the_default_stands(tmp_path):
    loader, _ = _layers(tmp_path)
    assert loader.load_harness().provider.reconnect_wait_seconds == 1800.0


def test_llm_config_is_reconnect_off_until_start_threads_it():
    assert LLMConfig(base_url="http://x/v1", model="m").reconnect_wait_seconds == 0.0


def test_the_queue_wait_keeps_its_meaning():
    # 0 and None still mean "wait indefinitely" for the queue; the reconnect bound's 0 means off.
    assert _QueueWaitState(0)._ceiling is None
    assert _QueueWaitState(None)._ceiling is None
    assert _pc().inference_queue_wait_seconds == 600.0

