"""AUTO-08 (spec 20 §9.3, D-17, R-14-08, R-14-10): `sense_launch(unit=)` reads the boot marker and says
whether the last boot died, `plan_launch` then picks the boot retry shape, and sensing still writes
nothing.

Hermetic: the marker is a real file under `tmp_path` written through `units.write_marker`; free memory,
the weights and the probe are the `Box` fakes the other launch tests use; nothing starts a process.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from localharness.config.models import AutosenseConfig
from localharness.provider import lifecycle, server, units
from localharness.provider.autosense import FreeMemory
from tests.unit.test_launch_fit_retry import Box
from tests.unit.test_launch_plan import FLASH_NEXT, SWIFT, _gguf_spec, _vllm_spec

GIB = 2**30
KNOBS = AutosenseConfig()
NOW = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
WINDOW = units.TIMEOUT_START_SECONDS  # 1800


@pytest.fixture
def box(monkeypatch):
    world = Box(monkeypatch)
    world.mem = FreeMemory(24 * GIB, 48 * GIB, "nvidia-smi")
    world.weights = 18 * GIB
    return world


def _marker(spec, age: float = 60, **over) -> units.Marker:
    fields = dict(
        record=lifecycle.record_key(spec), binary=str(spec.binary), invocation="inv-1", cause="death",
        shape="full", planned_context=40_000, full_fit=80_000, free_bytes=20 * GIB, dropped=(),
        at=(NOW - timedelta(seconds=age)).isoformat())
    fields.update(over)
    return units.Marker(**fields)


def _put(tmp_path: Path, marker: units.Marker) -> Path:
    units.write_marker(tmp_path, marker)
    return tmp_path / "vllm" / units.BOOT_MARKER_NAME


def _sense(spec, tmp_path, unit="restarted", **kw):
    state = units.UnitRuntimeState("inv-2", 1) if unit == "restarted" else unit
    return lifecycle.sense_launch(spec, tmp_path, KNOBS, unit=state, now=NOW, **kw)


def _plan(spec, sensed):
    return lifecycle.plan_launch(spec, sensed, server.launch_decision(spec, False), autosense=KNOBS,
                                 api_key_held=False)


# ------------------------------------------------------------------- the tracer


def test_a_death_marker_and_a_restarted_unit_plan_the_boot_retry_shape(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    marker = _marker(spec)
    _put(tmp_path, marker)

    sensed = _sense(spec, tmp_path)
    assert sensed.marker == marker and sensed.prior_death is True

    plan = _plan(spec, sensed)
    full = _plan(spec, _sense(spec, tmp_path, unit=None))
    assert (plan.shape, plan.cause) == ("retry", "boot")
    assert "draft-mtp" in plan.dropped and "--spec-type" not in plan.argv
    assert full.shape == "full" and full.boot_retry() == plan


def test_without_a_unit_the_marker_is_not_read(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    _put(tmp_path, _marker(spec))
    sensed = _sense(spec, tmp_path, unit=None)
    assert sensed.marker is None and sensed.prior_death is False
    assert _plan(spec, sensed).shape == "full"


def test_no_marker_file_with_a_unit_is_no_prior_death(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    sensed = _sense(spec, tmp_path)
    assert sensed.marker is None and sensed.prior_death is False


# ------------------------------------------------------------------- the boundaries


@pytest.mark.parametrize("age,expected", [(0, True), (1, True), (WINDOW - 1, True), (WINDOW, False), (WINDOW + 1, False)])
def test_the_marker_age_boundary_is_the_unit_load_window(tmp_path, box, age, expected):
    spec = _gguf_spec(tmp_path, SWIFT)
    _put(tmp_path, _marker(spec, age=age))
    assert _sense(spec, tmp_path).prior_death is expected


def test_the_window_is_the_units_own_timeout_start(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    _put(tmp_path, _marker(spec, age=600))
    short = units.UnitRuntimeState("inv-2", 1, timeout_start_seconds=300)
    assert _sense(spec, tmp_path, unit=short).prior_death is False
    assert _sense(spec, tmp_path).prior_death is True


def test_a_unit_that_never_restarted_is_no_prior_death(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    _put(tmp_path, _marker(spec))
    assert _sense(spec, tmp_path, unit=units.UnitRuntimeState("inv-2", 0)).prior_death is False
    assert _sense(spec, tmp_path, unit=units.UnitRuntimeState("inv-2", 2)).prior_death is True


@pytest.mark.parametrize("over", [{"record": "llamacpp:other"}, {"binary": "/x/another-llama-server"}])
def test_a_marker_for_another_record_or_binary_is_ignored_and_left_in_place(tmp_path, box, over):
    spec = _gguf_spec(tmp_path, SWIFT)
    path = _put(tmp_path, _marker(spec, **over))
    before = path.read_bytes()
    sensed = _sense(spec, tmp_path)
    assert sensed.prior_death is False
    assert path.read_bytes() == before  # `serve` replaces it; sensing never removes it (R-14-08)


@pytest.mark.parametrize("cause,shape,expected", [
    ("none", "retry", False), ("none", "full", True), ("remembered", "retry", False),
    ("remembered", "full", True), ("death", "retry", True)])
def test_cause_and_shape_decide_the_verdict(tmp_path, box, cause, shape, expected):
    spec = _gguf_spec(tmp_path, SWIFT)
    _put(tmp_path, _marker(spec, cause=cause, shape=shape))
    assert _sense(spec, tmp_path).prior_death is expected


# ------------------------------------------------------------------- sensing writes nothing


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("age", [60, WINDOW, 10**6])
def test_the_marker_file_is_byte_identical_after_sensing(tmp_path, box, age):
    spec = _gguf_spec(tmp_path, SWIFT)
    path = _put(tmp_path, _marker(spec, age=age))
    stat = os.stat(path)
    before = _tree(tmp_path)
    _sense(spec, tmp_path)
    _plan(spec, _sense(spec, tmp_path))
    assert _tree(tmp_path) == before
    assert os.stat(path).st_mtime_ns == stat.st_mtime_ns


def test_an_unparsable_marker_gives_none_and_is_left_in_place(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    path = tmp_path / "vllm" / units.BOOT_MARKER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    sensed = _sense(spec, tmp_path)
    assert sensed.marker is None and sensed.prior_death is False
    assert path.read_text() == "{not json"


# ------------------------------------------------------------------- other runtimes


def test_a_vllm_binary_record_may_say_prior_death_but_has_nothing_to_drop_at_boot(tmp_path):
    spec = _vllm_spec(tmp_path)
    _put(tmp_path, _marker(spec))
    sensed = _sense(spec, tmp_path)
    assert sensed.prior_death is True and sensed.marker is not None
    full = _plan(spec, sensed)
    assert full.shape == "full" and full.boot_retry() is None


def test_a_runtime_the_unit_does_not_serve_keeps_the_defaults(tmp_path):
    from localharness.config.models import ManagedServerConfig

    spec = ManagedServerConfig(runtime="ollama", model="qwen3:8b", port=11434)
    sensed = lifecycle.sense_launch(spec, tmp_path, KNOBS, unit=units.UnitRuntimeState("inv-2", 1), now=NOW)
    assert sensed.marker is None and sensed.prior_death is False


def test_boot_retry_is_none_for_a_plan_with_nothing_to_drop_or_halve(tmp_path, box):
    spec = _gguf_spec(tmp_path, FLASH_NEXT, context=32768)
    full = _plan(spec, _sense(spec, tmp_path, unit=None))
    assert full.boot_retry() is None


def test_boot_retry_is_none_for_a_plan_already_in_the_retry_shape(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    _put(tmp_path, _marker(spec))
    retry = _plan(spec, _sense(spec, tmp_path))
    assert retry.shape == "retry" and retry.boot_retry() is None


# ------------------------------------------------------------------- the scope predicate


def _record(runtime, **over):
    from localharness.config.models import ManagedServerConfig

    return ManagedServerConfig(runtime=runtime, model="m", port=8000, **over)


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


def test_the_not_served_line_is_the_contract():
    assert lifecycle.NOT_SERVED_BY_UNIT_LINE.format(name="qwen-docker", kind="docker") == (
        "qwen-docker is docker — not served by the unit")
