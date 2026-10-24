"""The pass records, the action labels and the pure functions that decide whether closing Plugins restarts."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
import yaml
from pydantic import BaseModel

from localharness.cli.plugins_cmd import (
    ACTION_CONFIGURE, ACTION_DISABLE, ACTION_ENABLE, ACTION_INFO, ACTION_INSTALL, ACTION_RERUN,
    ACTION_SHOW_INSTALL, Kind, PassProgress, PassResult, Phase, PluginSession, ReloadPlan, ReportKind,
    StepOutcome, TailKey, Touch, loads_from_plan, prior_touch, refused, touch_after_stop, touch_from,
)
from localharness.plugins import builtin, discovery
from localharness.plugins.api import Plugin, PluginManifest
from localharness.plugins.lifecycle import DoctorRow

_OUTCOMES = {
    "enabled", "enabled_unconfigured", "check_failed", "check_failed_noload", "check_skipped",
    "enabled_not_loading", "rerun_ok", "rerun_failed", "rerun_blocked", "disabled", "saved",
    "saved_overridden", "override", "declined", "refused",
}
_PHASES = {"consent", "install", "question", "question_installed", "download", "check", "edit", "disable"}
_CONFIG = {
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model", "available_models": ["test-model"]},
}


class _Settings(BaseModel):
    pass


class Stub(Plugin):
    """a bundled plugin with nothing to ask"""

    manifest = PluginManifest(name="stub", version="0.1.0", kind="tools", enabled_by_default=False)
    ConfigModel = _Settings


def _flat(alias) -> set[str]:
    return {m for arm in get_args(alias) for m in get_args(arm)}


def _session(cfg: Path = Path("/nowhere"), **kw) -> PluginSession:
    return PluginSession(cfg, None, kw.pop("running", frozenset()), kw.pop("unconfigured", {}),
                         kw.pop("failed", {}), **kw)


def _resolution(state: str | None):
    entry = None if state is None else SimpleNamespace(state=state)
    return SimpleNamespace(plan=SimpleNamespace(entry=lambda n: entry if n == "a" else None))


def _row(state: str, detail: str = "") -> DoctorRow:
    return DoctorRow("a", state, detail)


@pytest.fixture
def machine(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub,))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


def _turn(g: Path, on: bool) -> None:
    (g / "overrides.yaml").write_text(yaml.safe_dump({"stub": {"enabled": on}}), encoding="utf-8")


def test_kind_is_every_outcome_plus_the_two_unprinted() -> None:
    assert set(get_args(Kind)) == _OUTCOMES | {"install_skipped", "enabled_manual"}


def test_phase_members() -> None:
    assert set(get_args(Phase)) == _PHASES


def test_tail_key_is_the_phases_and_two_selectors() -> None:
    assert _flat(TailKey) == _PHASES | {"install_batch", "edit_written"}


def test_report_kind_members() -> None:
    assert set(get_args(ReportKind)) == {"wrote", "note", "checking", "row", "help", "next_steps",
                                         "stays_off", "refused"}


def test_literals_are_exact_strings() -> None:
    assert "Enabled" not in get_args(Kind) and "enabled " not in get_args(Kind)
    assert all(type(m) is str for m in get_args(Kind) + get_args(Phase) + get_args(ReportKind))


def test_pass_result_fields_hold_no_typed_value() -> None:
    assert [f.name for f in dataclasses.fields(PassResult)] == [
        "kind", "step", "install", "questions", "answered", "downloaded", "row", "loads", "rebind",
        "override", "shown", "detail", "next_steps", "help"]


def test_pass_result_defaults() -> None:
    r = PassResult("enabled", StepOutcome("x", True))
    assert (r.install, r.questions, r.answered, r.downloaded, r.row) == ("none", "none", (), False, None)
    assert (r.loads, r.rebind, r.override, r.shown, r.detail, r.next_steps, r.help) == (
        False, False, None, "", "", "", ())


def test_pass_result_answered_keeps_the_producers_order() -> None:
    assert PassResult("saved", StepOutcome("x", True), answered=("b", "a")).answered == ("b", "a")


@pytest.mark.parametrize("record", [
    PassResult("enabled", StepOutcome("x", True)), Touch(False, False, None),
    ReloadPlan(frozenset(), 0, 0), _session()])
def test_records_are_frozen(record) -> None:
    field = dataclasses.fields(record)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, field, None)


def test_pass_progress_is_the_one_mutable_holder() -> None:
    p = PassProgress()
    assert (p.phase, p.written, p.wrote, p.answered, p.overridden, p.rerun) == (
        "consent", False, False, False, False, False)
    p.phase, p.wrote = "download", True
    assert (p.phase, p.wrote) == ("download", True)


@pytest.mark.parametrize("parked, waiting, names, reload", [
    (0, 0, {"a"}, True), (1, 0, {"a"}, False), (0, 1, {"a"}, False), (0, 0, set(), False)])
def test_reload_only_with_names_and_nothing_parked_or_waiting(parked, waiting, names, reload) -> None:
    assert ReloadPlan(frozenset(names), parked, waiting).reload is reload


def test_a_session_without_touched_gets_its_own_dict() -> None:
    a, b = _session(), _session()
    assert a.touched == {} and a.touched is not b.touched


def test_action_labels() -> None:
    assert (ACTION_ENABLE, ACTION_INSTALL, ACTION_SHOW_INSTALL, ACTION_CONFIGURE, ACTION_RERUN,
            ACTION_DISABLE, ACTION_INFO) == (
        "Enable", "Install and set up", "Show install step", "Configure…", "Re-run setup", "Disable",
        "Info")


def test_prior_touch_seeds_from_running() -> None:
    s = _session(running=frozenset({"a"}))
    assert prior_touch(s, "a") == Touch(False, True, None)
    assert prior_touch(s, "b") == Touch(False, False, None)


def test_refused_carries_the_text_the_install_and_the_prior_loads() -> None:
    s = _session(running=frozenset({"a"}), touched={"a": Touch(False, False, None)})
    r = refused(s, "a", "why", "done")
    assert (r.kind, r.install, r.detail, r.loads, r.step) == ("refused", "done", "why", False,
                                                                StepOutcome("a", False))
    assert refused(_session(running=frozenset({"a"})), "a", "why", "none").loads is True


def test_refused_has_no_default_install() -> None:
    with pytest.raises(TypeError):
        refused(_session(), "a", "why")


def test_refused_leaves_touched_alone() -> None:
    s = _session()
    refused(s, "a", "why", "none")
    assert s.touched == {}


@pytest.mark.parametrize("state, probe, prior, extra, expect", [
    ("off", None, None, {}, False),
    (None, None, None, {}, False),
    ("on", None, None, {}, True),
    ("on", None, ("on", ""), {}, True),
    ("on", None, ("failed", "x"), {}, False),
    ("on", None, ("unconfigured", "x"), {}, False),
    ("on", None, None, {"failed": {"a": "boom"}}, False),
    ("on", None, None, {"unconfigured": {"a": "set k"}}, False),
    ("on", _row("failed", "failed — x"), None, {}, False),
    ("on", _row("unconfigured", "set k"), None, {}, False),
    ("on", _row("on"), None, {}, True),
    ("on", _row("on"), ("failed", "x"), {}, True),
    ("on", _row("on"), None, {"failed": {"a": "boom"}}, True),
    ("off", _row("on"), None, {}, False),
])
def test_loads_from_plan_per_conjunct(state, probe, prior, extra, expect) -> None:
    s = _session(**extra)
    assert loads_from_plan(s, "a", Touch(False, False, prior), probe, resolution=_resolution(state)) is expect


def test_loads_from_plan_resolves_fresh_through_the_session_dirs(machine) -> None:
    s, prior = _session(machine), Touch(False, False, None)
    _turn(machine, True)
    assert loads_from_plan(s, "stub", prior, None) is True
    assert loads_from_plan(s, "nosuch", prior, None) is False
    _turn(machine, False)
    assert loads_from_plan(s, "stub", prior, None) is False


def test_the_bookkeeping_never_stores_into_touched() -> None:
    s = _session(running=frozenset({"a"}))
    prior_touch(s, "a")
    loads_from_plan(s, "a", Touch(False, False, None), None, resolution=_resolution("on"))
    refused(s, "a", "why", "none")
    assert s.touched == {}


def _result(kind: str = "saved", on: bool = True, **kw) -> PassResult:
    return PassResult(kind, StepOutcome("a", on), **kw)


@pytest.mark.parametrize("result", [_result("refused"), _result("declined"), _result("rerun_blocked"),
                                    _result("override", True)])
def test_touch_from_leaves_touched_alone_when_nothing_was_written(result) -> None:
    assert touch_from(_session(), "a", result, Touch(True, True, ("failed", "x"))) is None


def test_touch_from_an_overridden_disable_wrote_and_keeps_the_probe() -> None:
    prior = Touch(False, True, ("failed", "x"))
    assert touch_from(_session(), "a", _result("override", False, loads=True), prior) == Touch(False, True, ("failed", "x"))


def test_touch_from_probe_from_the_row() -> None:
    got = touch_from(_session(), "a", _result("enabled", row=_row("failed", "failed — no answer"),
                                              questions="skipped"), Touch(False, False, ("on", "")))
    assert got.probe == ("failed", "failed — no answer")


def test_touch_from_probe_from_skipped_questions_without_a_row() -> None:
    got = touch_from(_session(), "a", _result("enabled_unconfigured", questions="skipped"),
                     Touch(False, False, ("on", "")))
    assert got.probe == ("unconfigured", "setup questions skipped")


def test_touch_from_a_disable_clears_the_probe() -> None:
    got = touch_from(_session(), "a", _result("disabled", False), Touch(False, True, ("failed", "x")))
    assert got.probe is None


def test_touch_from_otherwise_keeps_the_prior_probe() -> None:
    got = touch_from(_session(), "a", _result("saved", loads=True), Touch(False, True, ("failed", "x")))
    assert got.probe == ("failed", "x")


def test_touch_from_ors_rebind_and_takes_loads_from_the_result() -> None:
    assert touch_from(_session(), "a", _result("saved", rebind=False), Touch(True, True, None)) == Touch(True, False, None)
    assert touch_from(_session(), "a", _result("saved", rebind=True, loads=True),
                      Touch(False, False, None)) == Touch(True, True, None)


def _stop(session: PluginSession, prior: Touch, **kw) -> Touch | None:
    return touch_after_stop(session, "stub", PassProgress(**kw), prior)


def test_touch_after_stop_before_any_write_is_none() -> None:
    assert _stop(_session(), Touch(False, False, None), phase="download") is None


def test_touch_after_stop_a_running_plugins_written_configure_rebinds() -> None:
    s, prior = _session(running=frozenset({"stub"})), Touch(False, True, ("on", ""))
    assert _stop(s, prior, phase="edit", wrote=True) == Touch(True, True, ("on", ""))


def test_touch_after_stop_an_overridden_configure_does_not_rebind() -> None:
    s, prior = _session(running=frozenset({"stub"})), Touch(False, True, None)
    assert _stop(s, prior, phase="edit", wrote=True, overridden=True) == Touch(False, True, None)


def test_touch_after_stop_written_answers_rebind_a_running_plugin_only() -> None:
    prior = Touch(False, True, None)
    assert _stop(_session(running=frozenset({"stub"})), prior, phase="question", answered=True).rebind is True
    assert _stop(_session(), prior, phase="question", answered=True).rebind is False
    assert _stop(_session(), Touch(True, True, None), phase="question", answered=True).rebind is True


def test_touch_after_stop_a_plugin_not_running_does_not_rebind_on_edit() -> None:
    assert _stop(_session(), Touch(False, False, None), phase="edit", wrote=True).rebind is False


def test_touch_after_stop_at_the_download_reads_the_fresh_plan(machine) -> None:
    _turn(machine, True)
    s = _session(machine, running=frozenset({"stub"}))
    assert _stop(s, Touch(False, False, None), phase="download", wrote=True) == Touch(False, True, None)
    assert _stop(s, Touch(False, False, None), phase="check", wrote=True).loads is True


def test_touch_after_stop_a_stopped_rerun_keeps_prior_loads(machine) -> None:
    _turn(machine, True)
    s = _session(machine, running=frozenset({"stub"}))
    assert _stop(s, Touch(False, False, None), phase="check", wrote=True, rerun=True).loads is False


def test_touch_after_stop_a_build_failed_plugin_re_enabled_does_not_load(machine) -> None:
    _turn(machine, True)
    s = _session(machine, failed={"stub": "boom"})
    assert _stop(s, Touch(False, False, None), phase="download", wrote=True).loads is False


def test_touch_after_stop_keeps_the_prior_probe(machine) -> None:
    _turn(machine, True)
    got = _stop(_session(machine), Touch(False, False, ("failed", "x")), phase="download", wrote=True)
    assert got.probe == ("failed", "x") and got.loads is False


def test_the_derivations_never_store_into_touched(machine) -> None:
    _turn(machine, True)
    s = _session(machine, touched={"stub": Touch(False, True, None)})
    before = dict(s.touched)
    touch_from(s, "stub", _result("saved"), Touch(False, True, None))
    _stop(s, Touch(False, True, None), phase="download", wrote=True)
    _stop(s, Touch(False, True, None), phase="download")
    assert s.touched == before
