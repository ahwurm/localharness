"""TUNE-08 / SEC-05 (spec 20 §6.0, §9.3, D-18, D-28): `confirm_launch` in its two forms.

The plan form is covered end to end in test_launch_remembered.py; here are its direct conditions and the
marker form, on constructed values: a `Marker`, a `UnitRuntimeState`, a `ProcRecord` and a
`LiveEndpoint` built by hand, the ledger a real file in `tmp_path` and the boot marker a real
`vllm/boot-attempt` file there. Nothing reads a unit, a journal or the `/proc` of another process.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from localharness.agent.gguf_header import read_header
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import lifecycle, server, speed_stats, tuning, units
from localharness.provider.autosense import FIT_KEYS, FreeMemory, WindowReport, fit_header
from localharness.provider.procfs import ProcRecord
from tests.unit.gguf_support import write_gguf
from tests.unit.test_launch_plan import FLASH_NEXT, SWIFT

GIB = 2**30
KNOBS = AutosenseConfig()
TOTAL = 48 * GIB
CONTEXT_LENGTH = 262144


def _spec(tmp_path: Path, kv: dict = SWIFT, **over) -> ManagedServerConfig:
    model = write_gguf(tmp_path / "model.gguf", kv)
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": str(model), "port": 8080, **over})


def _header(spec) -> dict:
    return fit_header(read_header(spec.model, FIT_KEYS))


def _marker(spec, **over) -> units.Marker:
    fields = dict(record=lifecycle.record_key(spec), binary=str(spec.binary), invocation="inv-1",
                  cause="death", shape="retry", planned_context=40_000, full_fit=80_000,
                  free_bytes=20 * GIB, dropped=(), at="2026-10-13T11:59:30+00:00")
    fields.update(over)
    return units.Marker(**fields)


def _unit(invocation="inv-1", restarts=1) -> units.UnitRuntimeState:
    return units.UnitRuntimeState(invocation, restarts)


def _live(served=40_000) -> lifecycle.LiveEndpoint:
    report = WindowReport(served=served, trained=CONTEXT_LENGTH, fits=None, free_bytes=None,
                          total_bytes=None, weights_bytes=None, asked=None, mode="fit-by-server")
    return lifecycle.LiveEndpoint(base_url="http://127.0.0.1:8080/v1", served_models=["m"], handle=1,
                                  window=report)


def _process(*argv: str) -> ProcRecord:
    return ProcRecord(pid=4242, uid=1000, argv=("/x/llama-server", "-m", "m.gguf", *argv), exe=None,
                      cwd=None, start_time=1, unit="llama.service", environ=None, mapped_ggufs=())


def _confirm(tmp_path, record, marker, **over) -> None:
    kw = dict(marker=marker, live=_live(), process=_process(), unit=_unit(), spec=record, autosense=KNOBS,
              header=_header(record), total_bytes=TOTAL)
    kw.update(over)
    lifecycle.confirm_launch(tmp_path, lifecycle.record_key(record), **kw)


def _recorded(tmp_path, spec):
    return speed_stats.last_retry(speed_stats.default_speed_stats_path(tmp_path), lifecycle.record_key(spec))


def _plant_marker(tmp_path) -> Path:
    path = server.server_dir(tmp_path) / units.BOOT_MARKER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


# ------------------------------------------------------------------- the marker form writes under every condition


def test_a_marker_bound_to_the_live_run_writes_the_record(tmp_path):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec))
    rec = _recorded(tmp_path, spec)
    assert rec is not None
    assert (rec.full_fit, rec.free_bytes, rec.dropped) == (80_000, 20 * GIB, ("draft-mtp",))
    assert (rec.binary, rec.kv_cache_type, rec.draft_p_min) == (str(spec.binary), "q8_0", 0.6)


FLIPS = {
    "another invocation": dict(unit=_unit("inv-9")),
    "no live invocation": dict(unit=_unit(None)),
    "no unit": dict(unit=None),
    "no restart behind it": dict(unit=_unit(restarts=0)),
    "window not the planned one": dict(live=_live(served=20_000)),
    "window unknown": dict(live=_live(served=None)),
    "no live endpoint": dict(live=None),
    "no verified process": dict(process=None),
    "no spec": dict(spec=None),
    "no knobs": dict(autosense=None),
}


@pytest.mark.parametrize("flip", sorted(FLIPS))
def test_a_marker_writes_nothing_when_any_condition_is_missing(tmp_path, flip):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec), **FLIPS[flip])
    assert _recorded(tmp_path, spec) is None


MARKER_FLIPS = {
    "cause remembered": dict(cause="remembered"),
    "cause none": dict(cause="none"),
    "another record": dict(record="llamacpp:another"),
    "another binary": dict(binary="/x/other-llama-server"),
    "no invocation": dict(invocation=None),
    "no planned window": dict(planned_context=None),
}


@pytest.mark.parametrize("flip", sorted(MARKER_FLIPS))
def test_a_marker_that_is_not_this_death_writes_nothing(tmp_path, flip):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, **MARKER_FLIPS[flip]))
    assert _recorded(tmp_path, spec) is None


# ------------------------------------------------------------------- dropped comes from the record's derive and the live argv


@pytest.mark.parametrize("argv", [(), ("--spec-type", "draft-mtp"), ("--spec-type=draft-mtp",)])
def test_dropped_is_draft_mtp_only_when_the_live_argv_has_no_spec_type(tmp_path, argv):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, dropped=("draft-mtp",) if argv else ()), process=_process(*argv))
    assert _recorded(tmp_path, spec).dropped == (() if argv else ("draft-mtp",))


def test_a_marker_claiming_no_drop_cannot_hide_one_the_live_argv_shows(tmp_path):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, dropped=()), process=_process())
    assert _recorded(tmp_path, spec).dropped == ("draft-mtp",)


def test_a_speculative_off_record_is_never_recorded_as_dropping_the_draft(tmp_path):
    """R-09e: with `speculative: off` the record's own derive emits no spec type, so whatever the marker
    claims, `draft-mtp` is not recorded as dropped."""
    spec = _spec(tmp_path, speculative="off")
    _confirm(tmp_path, spec, _marker(spec, dropped=("draft-mtp",)))
    rec = _recorded(tmp_path, spec)
    assert rec is None or rec.dropped == ()


def test_a_header_without_mtp_layers_never_records_a_dropped_draft(tmp_path):
    spec = _spec(tmp_path, FLASH_NEXT)
    _confirm(tmp_path, spec, _marker(spec, dropped=("draft-mtp",)))
    assert _recorded(tmp_path, spec).dropped == ()


def test_the_markers_own_dropped_is_never_read(tmp_path):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, dropped=("rm -rf", "draft-mtp", "whatever")))
    assert _recorded(tmp_path, spec).dropped == ("draft-mtp",)


def test_a_speculative_record_set_by_hand_extra_args_derives_no_drop(tmp_path):
    spec = _spec(tmp_path, extra_args=["--spec-type", "draft-mtp"])
    assert tuning.derive(spec, _header(spec), KNOBS).spec_type is None
    _confirm(tmp_path, spec, _marker(spec), process=_process("--spec-type", "draft-mtp"))
    assert _recorded(tmp_path, spec).dropped == ()


# ------------------------------------------------------------------- planted values are clamped


def test_a_planted_full_fit_is_clamped_to_the_headers_context_length(tmp_path):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, full_fit=10**12))
    assert _recorded(tmp_path, spec).full_fit == CONTEXT_LENGTH


def test_an_explicit_context_length_wins_over_the_header_in_the_clamp(tmp_path):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, full_fit=10**12), context_length=100_000)
    assert _recorded(tmp_path, spec).full_fit == 100_000


@pytest.mark.parametrize("over", [
    dict(header={}),                                     # an unreadable header, no context_length given
    dict(header={"block_count": 65, "nextn_predict_layers": 1}),
    dict(header={"context_length": True}),
    dict(context_length=0),
    dict(context_length=-1),
])
def test_a_full_fit_that_cannot_be_clamped_is_not_recorded(tmp_path, over):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, full_fit=10**12), **over)
    assert _recorded(tmp_path, spec) is None


def test_a_planted_free_bytes_is_clamped_to_total_memory(tmp_path):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, free_bytes=10**15))
    assert _recorded(tmp_path, spec).free_bytes == TOTAL


def test_free_bytes_is_not_recorded_without_a_known_total(tmp_path):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec), total_bytes=None)
    rec = _recorded(tmp_path, spec)
    assert rec is not None and rec.free_bytes is None


def test_a_negative_or_boolean_free_bytes_is_not_recorded(tmp_path):
    spec = _spec(tmp_path)
    for bad in (-1, True):
        _confirm(tmp_path, spec, _marker(spec, free_bytes=bad))
        assert _recorded(tmp_path, spec).free_bytes is None


@pytest.mark.parametrize("bad", [None, 0, -4, True, "80000", 3.5])
def test_a_missing_or_non_positive_full_fit_writes_nothing(tmp_path, bad):
    spec = _spec(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, full_fit=bad))
    assert _recorded(tmp_path, spec) is None


# ------------------------------------------------------------------- the marker is removed when its invocation is the live one


@pytest.mark.parametrize("over", [
    dict(planned_context=1),            # the window did not match
    dict(cause="remembered"),
    dict(cause="none"),
    dict(record="llamacpp:another"),
    dict(full_fit=None),
    dict(),                              # the write happens
])
def test_a_marker_of_the_live_invocation_is_removed_whatever_the_outcome(tmp_path, over):
    spec = _spec(tmp_path)
    path = _plant_marker(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, **over))
    assert not path.exists()


def test_a_marker_removed_on_a_missing_restart_writes_nothing(tmp_path):
    spec = _spec(tmp_path)
    path = _plant_marker(tmp_path)
    _confirm(tmp_path, spec, _marker(spec), unit=_unit(restarts=0))
    assert not path.exists() and _recorded(tmp_path, spec) is None


def test_a_marker_of_another_invocation_is_left_for_its_own_run(tmp_path):
    spec = _spec(tmp_path)
    path = _plant_marker(tmp_path)
    _confirm(tmp_path, spec, _marker(spec, invocation="inv-other"))
    _confirm(tmp_path, spec, _marker(spec, invocation=None))
    _confirm(tmp_path, spec, _marker(spec), unit=None)
    assert path.exists()


def test_confirm_launch_never_raises_when_the_removal_fails(tmp_path, monkeypatch):
    spec = _spec(tmp_path)

    def broken(config_dir):
        raise OSError("read-only file system")

    monkeypatch.setattr(units, "remove_marker", broken)
    _confirm(tmp_path, spec, _marker(spec))
    assert _recorded(tmp_path, spec) is not None  # the write was not lost to the failing removal


def test_confirm_launch_never_raises_when_the_ledger_write_fails(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    path = _plant_marker(tmp_path)

    def broken(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(speed_stats, "record_retry", broken)
    _confirm(tmp_path, spec, _marker(spec))
    assert not path.exists()  # the marker is still removed


# ------------------------------------------------------------------- the plan form, directly


def _planned(tmp_path, monkeypatch, *, shape):
    from localharness.provider import autosense

    spec = _spec(tmp_path)
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(24 * GIB, TOTAL, "nvidia-smi"))
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: None)
    monkeypatch.setattr(autosense, "weights_bytes", lambda model, mmproj=None: 18 * GIB)
    sensed = lifecycle.sense_launch(spec, tmp_path, KNOBS)
    full = lifecycle.plan_launch(spec, sensed, server.launch_decision(spec, False), autosense=KNOBS,
                                 api_key_held=False)
    return spec, sensed, full, (full if shape == "full" else full.retry())


def test_the_plan_form_writes_a_death_retry_from_the_pre_launch_values(tmp_path, monkeypatch):
    spec, sensed, full, retry = _planned(tmp_path, monkeypatch, shape="retry")
    lifecycle.confirm_launch(tmp_path, lifecycle.record_key(spec), plan=retry, live=_live())
    rec = _recorded(tmp_path, spec)
    assert (rec.full_fit, rec.free_bytes, rec.dropped) == (full.full_fit, 24 * GIB, ("draft-mtp",))


def test_the_plan_form_leaves_a_remembered_or_boot_shape_as_it_is(tmp_path, monkeypatch):
    spec, sensed, full, _ = _planned(tmp_path, monkeypatch, shape="full")
    for cause in ("remembered", "boot"):
        shaped = full._shaped(cause, drop_spec=True)
        lifecycle.confirm_launch(tmp_path, lifecycle.record_key(spec), plan=shaped, live=_live())
    assert _recorded(tmp_path, spec) is None


def test_the_plan_form_clears_a_remembered_record_on_a_full_shape_success(tmp_path, monkeypatch):
    spec, sensed, full, _ = _planned(tmp_path, monkeypatch, shape="full")
    path = speed_stats.default_speed_stats_path(tmp_path)
    speed_stats.record_retry(path, lifecycle.record_key(spec), full_fit=1000, free_bytes=None,
                             dropped=[], binary=str(spec.binary), kv_cache_type="q8_0", draft_p_min=0.6)
    speed_stats.record_last_served(path, "keep-me")
    remembered = dataclasses.replace(sensed, remembered=speed_stats.last_retry(path, lifecycle.record_key(spec)))
    plan = lifecycle.plan_launch(spec, remembered, server.launch_decision(spec, False), autosense=KNOBS,
                                 api_key_held=False)
    assert plan.shape == "full"
    lifecycle.confirm_launch(tmp_path, lifecycle.record_key(spec), plan=plan, live=_live())
    assert _recorded(tmp_path, spec) is None
    assert speed_stats.last_served(path) == "keep-me"


def test_the_plan_form_never_touches_the_marker(tmp_path, monkeypatch):
    spec, sensed, full, retry = _planned(tmp_path, monkeypatch, shape="retry")
    path = _plant_marker(tmp_path)
    lifecycle.confirm_launch(tmp_path, lifecycle.record_key(spec), plan=retry, live=_live())
    assert path.exists()


def test_confirm_launch_with_neither_form_does_nothing(tmp_path):
    lifecycle.confirm_launch(tmp_path, "llamacpp:x")
    assert list(tmp_path.iterdir()) == []
