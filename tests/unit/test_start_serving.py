"""`cli/start_serving.py`: what `start` decides once it knows what serves (spec 20 §8.2, §1 item 3, §9.3).

Offline: `list_live_models` is faked (`Probes`), harnesses come from `ConfigLoader` over a tmp dir,
the `/proc` side is the fake tree of `fake_proc`, the unit's state is patched, the server boundaries
a swap crosses are recorded fakes. No socket, no process, no service manager (D-30, D-32). Test
fakes take `**kw`.
"""
from __future__ import annotations

import contextlib
import json
import os

import pytest

from localharness.cli import model_ops, start_serving, workspace
from localharness.cli.model_ops import (
    RUNG_ACTIVE,
    RUNG_DEFAULT,
    RUNG_PENDING,
    RUNG_PRESET,
    Serving,
    ServingSense,
    launchable_loopback_others,
    resolve_serving,
    sense_serving,
    shares_base_url,
)
from localharness.cli.start_serving import (
    ATTACH_LINE,
    OTHER_SERVING_LINE,
    RESUME_CLAUSE,
    Decision,
    decide,
    selected_answers,
    settle_attach,
)
from localharness.cli.workspace import MachineTrust
from localharness.config.overlay import load_overlay
from localharness.config.models import AutosenseConfig
from localharness.provider import lifecycle, procfs, server, speed_stats, units
from localharness.provider import server as managed_server
from tests.unit.gguf_support import write_gguf
from tests.unit.procfs_support import fake_proc  # noqa: F401
from tests.unit.test_sense_serving import (  # noqa: F401
    FLASH_ACTIVE,
    PEER_URL,
    PRIMARY,
    PROVIDER_KEY,
    PRIMARY_URL,
    Probes,
    home,
    load,
    peer,
    probes,
    record,
    seed,
)


def ledger(home):
    return speed_stats.default_speed_stats_path(home)


def serve(home, preset=None, **kw):
    got = resolve_serving(load(home, **kw), MachineTrust(), preset, config_dir=home)
    assert isinstance(got, Serving), got
    return got


# ----------------------------------------------------------------------------------------- the tracer


@pytest.mark.asyncio
async def test_a_resumed_peer_answers_and_the_attach_lands_as_a_remembered_attach(
        home, probes, monkeypatch):
    seed(home, active=FLASH_ACTIVE, last_served=PRIMARY)
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    assert (serving.record.name, serving.rung) == ("flash-next", RUNG_ACTIVE)

    others = launchable_loopback_others(harness, MachineTrust(), serving.record)
    assert [r.name for r in others] == [PRIMARY]
    sense = await sense_serving(harness, others)
    assert sense.answering == () and probes.urls == [PRIMARY_URL]

    last = speed_stats.last_served(ledger(home))
    decision = decide(serving, selected_answers=True, sense=sense, last_served=last)
    assert (decision.kind, decision.record.name, decision.rung) == ("attach", "flash-next", RUNG_ACTIVE)
    assert decision.lines == ("Serving flash-next (your last /model choice)",)
    assert decision.lines == (ATTACH_LINE.format(name="flash-next", clause=RESUME_CLAUSE),)

    overrides = (home / "overrides.yaml").read_bytes()
    spied: list[dict] = []
    real = model_ops.persist_landed

    async def spy(harness, rec, **kw):
        spied.append({"record": rec.name, **kw})
        return await real(harness, rec, **kw)

    monkeypatch.setattr(model_ops, "persist_landed", spy)
    await settle_attach(harness, decision, config_dir=home)

    assert spied == [{"record": "flash-next", "outcome": "attach", "rung": RUNG_ACTIVE,
                      "config_dir": home}]
    assert speed_stats.last_served(ledger(home)) == "flash-next"
    assert (home / "overrides.yaml").read_bytes() == overrides  # no selection was written
    assert load_overlay(home / "overrides.yaml")["active_endpoint"] == FLASH_ACTIVE


def test_start_serving_does_not_import_start_cmd():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(start_serving.__file__).read_text(encoding="utf-8"))
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [n for n in names if "start_cmd" in n]
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
    assert "start_cmd" not in imported


# ------------------------------------------------------------------------------------- the decide matrix


def _sense(harness, *names, ids=None) -> ServingSense:
    records = {r.name: r for r in model_ops.serving_records(harness)}
    return ServingSense(tuple(records[n] for n in names), dict(ids or {}))


NOTHING = ServingSense((), {})


def test_decide_attaches_on_the_preset_and_pending_rungs_with_no_clause(home):
    seed(home, overrides={"pending_preset": "flash-next"}, last_served=PRIMARY)
    for preset, rung in (("flash-next", RUNG_PRESET), (None, RUNG_PENDING)):
        serving = serve(home, preset)
        assert serving.rung == rung
        got = decide(serving, selected_answers=True, sense=NOTHING, last_served=PRIMARY)
        assert got == Decision("attach", serving.record, None, (), rung)


def test_decide_prints_no_resume_clause_when_the_active_endpoint_is_the_last_served(home):
    seed(home, active=FLASH_ACTIVE, last_served="flash-next")
    serving = serve(home)
    got = decide(serving, selected_answers=True, sense=NOTHING, last_served="flash-next")
    assert (got.kind, got.lines, got.rung) == ("attach", (), RUNG_ACTIVE)


def test_decide_prints_the_resume_clause_when_nothing_was_ever_served(home):
    seed(home, active=FLASH_ACTIVE)
    got = decide(serve(home), selected_answers=True, sense=NOTHING, last_served=None)
    assert got.lines == ("Serving flash-next (your last /model choice)",)


def test_decide_prints_no_clause_on_the_server_rung(home):
    seed(home, last_served="flash-next")
    serving = serve(home)
    assert serving.rung == RUNG_DEFAULT
    got = decide(serving, selected_answers=True, sense=NOTHING, last_served="flash-next")
    assert (got.kind, got.lines, got.rung) == ("attach", (), RUNG_DEFAULT)


def test_decide_attaches_to_another_with_one_clause_and_persists_no_rung(home):
    seed(home, active=FLASH_ACTIVE)
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    sense = _sense(harness, PRIMARY)
    got = decide(serving, selected_answers=False, sense=sense, last_served="flash-next")
    assert (got.kind, got.record.name, got.incumbent, got.rung) == ("attach_other", PRIMARY, None, None)
    assert got.lines == (
        f"serving {PRIMARY}; your last choice was flash-next — /model flash-next swaps",)
    assert got.lines == (OTHER_SERVING_LINE.format(serving=PRIMARY, selected="flash-next"),)


def test_decide_prints_no_clause_when_the_one_that_answers_is_the_last_served(home):
    seed(home, active=FLASH_ACTIVE)
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    got = decide(serving, selected_answers=False, sense=_sense(harness, PRIMARY), last_served=PRIMARY)
    assert (got.kind, got.lines) == ("attach_other", ())


def test_decide_swaps_when_the_selection_is_explicit_and_another_answers(home):
    seed(home, last_served=PRIMARY)
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), "flash-next", config_dir=home)
    got = decide(serving, selected_answers=False, sense=_sense(harness, PRIMARY), last_served=PRIMARY)
    assert got.kind == "swap" and got.record.name == "flash-next" and got.incumbent.name == PRIMARY
    assert got.lines == () and got.rung == RUNG_PRESET


def test_decide_swaps_for_a_pending_preset_with_another_answering(home):
    seed(home, overrides={"pending_preset": "flash-next"})
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    got = decide(serving, selected_answers=False, sense=_sense(harness, PRIMARY), last_served=None)
    assert (got.kind, got.incumbent.name, got.rung) == ("swap", PRIMARY, RUNG_PENDING)


def test_decide_launches_when_nothing_answers(home):
    seed(home, active=FLASH_ACTIVE)
    serving = serve(home)
    got = decide(serving, selected_answers=False, sense=NOTHING, last_served=PRIMARY)
    assert got == Decision("launch", serving.record, None, (), RUNG_ACTIVE)


def test_decide_never_prints_both_clauses_for_one_event(home):
    seed(home, active=FLASH_ACTIVE, overrides={"pending_preset": "flash-next"})
    harness = load(home)
    for preset in (None, "flash-next", PRIMARY):
        serving = resolve_serving(harness, MachineTrust(), preset, config_dir=home)
        for answers in (True, False):
            for sense in (NOTHING, _sense(harness, PRIMARY), _sense(harness, "flash-next")):
                for last in (None, PRIMARY, "flash-next"):
                    got = decide(serving, selected_answers=answers, sense=sense, last_served=last)
                    assert len(got.lines) <= 1, (preset, answers, last, got)


def test_decide_ignores_the_selected_record_among_the_answering(home):
    seed(home, active=FLASH_ACTIVE)
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    got = decide(serving, selected_answers=False, sense=_sense(harness, "flash-next"), last_served=None)
    assert got.kind == "launch"  # the probe said no; the sensor's own claim does not override it


# ----------------------------------------------------------------------------------- selected_answers


def test_selected_answers_needs_the_probe(home):
    seed(home)
    harness = load(home)
    flash = record(harness, "flash-next")
    assert selected_answers(False, flash, NOTHING, shared=False) is False
    assert selected_answers(False, flash, _sense(harness, "flash-next"), shared=True) is False


def test_selected_answers_when_the_address_is_unshared_is_the_probe(home):
    seed(home)
    harness = load(home)
    assert selected_answers(True, record(harness, "flash-next"), NOTHING, shared=False) is True


def test_selected_answers_on_a_shared_address_needs_the_sensor_to_name_the_selected_record(home):
    seed(home, peers=[peer("flash-next", 8080, url=PRIMARY_URL)])
    harness = load(home)
    primary, flash = record(harness, PRIMARY), record(harness, "flash-next")
    assert shares_base_url(primary, model_ops.serving_records(harness)) is True
    assert selected_answers(True, primary, _sense(harness, "flash-next"), shared=True) is False
    assert selected_answers(True, primary, _sense(harness, PRIMARY), shared=True) is True
    assert selected_answers(True, flash, _sense(harness, "flash-next"), shared=True) is True


# --------------------------------------------------------------------------------------- settle_attach


@pytest.mark.asyncio
async def test_settling_an_attach_on_the_pending_preset_rung_removes_the_key_and_records_last_served(home):
    seed(home, active=FLASH_ACTIVE, overrides={"pending_preset": "flash-next"})
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    decision = decide(serving, selected_answers=True, sense=NOTHING, last_served=None)
    await settle_attach(harness, decision, config_dir=home)
    overlay = load_overlay(home / "overrides.yaml")
    assert "pending_preset" not in overlay and overlay["active_endpoint"] == FLASH_ACTIVE
    assert speed_stats.last_served(ledger(home)) == "flash-next"


@pytest.mark.asyncio
async def test_settling_an_attach_on_the_preset_rung_touches_only_the_ledger(home):
    seed(home, overrides={"pending_preset": "gemma"}, peers=[peer("flash-next"), peer("gemma", 8082)])
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), "flash-next", config_dir=home)
    before = (home / "overrides.yaml").read_bytes()
    await settle_attach(harness, decide(serving, selected_answers=True, sense=NOTHING, last_served=None),
                        config_dir=home)
    assert (home / "overrides.yaml").read_bytes() == before
    assert speed_stats.last_served(ledger(home)) == "flash-next"


@pytest.mark.asyncio
async def test_settling_an_attach_to_another_records_that_ones_last_served_and_nothing_else(home):
    seed(home, active=FLASH_ACTIVE)
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    decision = decide(serving, selected_answers=False, sense=_sense(harness, PRIMARY), last_served=None)
    assert decision.kind == "attach_other"
    before = (home / "overrides.yaml").read_bytes()
    await settle_attach(harness, decision, config_dir=home)
    assert (home / "overrides.yaml").read_bytes() == before  # the active_endpoint stays on the older choice
    assert speed_stats.last_served(ledger(home)) == PRIMARY


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["swap", "launch"])
async def test_settle_attach_refuses_a_decision_that_is_not_an_attach(home, kind):
    seed(home)
    harness = load(home)
    decision = Decision(kind, record(harness, "flash-next"), None, (), RUNG_PRESET)
    with pytest.raises(ValueError):
        await settle_attach(harness, decision, config_dir=home)
    assert speed_stats.last_served(ledger(home)) is None


# ------------------------------------------------------------------------------------ liveness verdict

LLAMA = "/x/llama-server"
UNIT_CGROUP = (f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/"
               f"{procfs.HARNESS_UNIT}")
SERVE_ARGV = ("/usr/bin/python3", "-I", "-m", "localharness", "serve")


def _pidfile(home, text: str) -> None:
    directory = server.server_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / procfs.PIDFILE_NAME).write_text(text, encoding="utf-8")


def _flash_launch(fake_proc, home, *, pid=4242) -> None:
    fake_proc.process(pid, argv=(LLAMA, "-a", "flash-next", "--port", "8081"), exe=LLAMA, start_time=777)
    _pidfile(home, f"{pid} 777 8081")


@pytest.fixture
def named_peers(home):
    seed(home, peers=[peer("flash-next", served_name="flash-next")])
    return load(home)


def _verdict(harness, home, selected_name, **kw):
    return start_serving.liveness_verdict(
        harness, MachineTrust(), record(harness, selected_name), home, **kw)


def test_liveness_with_no_pidfile_and_no_listener_is_none(named_peers, home, fake_proc):  # noqa: F811
    got = _verdict(named_peers, home, "flash-next")
    assert (got.kind, got.pid, got.record, got.line) == ("none", None, None, None)


def test_liveness_a_verified_live_launch_of_the_selected_record_is_this_one_still_loading(
        named_peers, home, fake_proc):  # noqa: F811
    _flash_launch(fake_proc, home)
    got = _verdict(named_peers, home, "flash-next")
    assert (got.kind, got.pid, got.record.name) == ("this", 4242, "flash-next")


def test_liveness_a_verified_live_launch_of_another_record_is_the_swaps_incumbent(
        named_peers, home, fake_proc):  # noqa: F811
    _flash_launch(fake_proc, home)
    got = _verdict(named_peers, home, PRIMARY)
    assert (got.kind, got.pid, got.record.name) == ("other", 4242, "flash-next")


def test_liveness_a_pidfile_naming_a_process_that_is_no_launch_is_none_when_nothing_listens(
        named_peers, home, fake_proc):  # noqa: F811
    fake_proc.process(4243, argv=("/usr/bin/sleep", "600"), exe="/usr/bin/sleep", start_time=777)
    _pidfile(home, "4243 777 8080")
    got = _verdict(named_peers, home, PRIMARY)
    assert (got.kind, got.record) == ("none", None)


def test_liveness_verdict_a_launch_of_the_right_binary_on_another_port_is_not_this_record(
        named_peers, home, fake_proc):  # noqa: F811
    fake_proc.process(4244, argv=(LLAMA, "-a", "flash-next", "--port", "9999"), exe=LLAMA, start_time=777)
    _pidfile(home, "4244 777 8081")
    assert _verdict(named_peers, home, "flash-next").kind == "none"


def test_liveness_a_foreign_holder_of_the_selected_port_is_named_by_the_incumbent_sentence(
        named_peers, home, fake_proc):  # noqa: F811
    fake_proc.listen(8080, 555, os.getuid())
    fake_proc.process(5000, argv=("/usr/bin/python3", "-m", "http.server", "8080"),
                      exe="/usr/bin/python3", fds={3: 555})
    got = _verdict(named_peers, home, PRIMARY)
    holder = workspace.port_holder(8080)
    assert holder is not None and holder.pid == 5000
    expected = workspace.incumbent_sentence(
        holder, port=8080, form="imperative", served=PRIMARY, target=PRIMARY)
    assert (got.kind, got.pid, got.record, got.line) == ("foreign", 5000, None, expected)
    assert "not by the harness" in got.line and "None" not in got.line


def test_liveness_a_listener_on_a_remote_selected_address_is_not_ours_to_name(home, fake_proc):  # noqa: F811
    seed(home, peers=[peer("flash-next", url="http://10.0.0.5:8081/v1", lifecycle=False)])
    harness = load(home)
    fake_proc.listen(8081, 556, os.getuid())
    fake_proc.process(5001, argv=("/usr/bin/python3",), exe="/usr/bin/python3", fds={3: 556})
    assert _verdict(harness, home, "flash-next").kind == "none"


def _serve_process(fake_proc, home, *, cgroup=UNIT_CGROUP) -> None:
    fake_proc.process(4242, argv=SERVE_ARGV, exe="/usr/bin/python3", start_time=777, cgroup=cgroup)
    _pidfile(home, "4242 777")


def test_liveness_the_units_serve_process_is_starting_with_the_default_bound(
        named_peers, home, fake_proc, monkeypatch):  # noqa: F811
    _serve_process(fake_proc, home)
    monkeypatch.setattr(start_serving, "unit_runtime_state", lambda: pytest.fail("the verdict asked the seam again"))
    got = _verdict(named_peers, home, "flash-next", main_pid=4242, unit_state=None)
    assert (got.kind, got.pid, got.record.name) == ("starting", 4242, "flash-next")
    assert got.line == "autostart is starting flash-next (pid 4242) — waiting up to 600s"


def test_liveness_the_starting_bound_is_the_units_timeout_start_when_it_has_one(
        named_peers, home, fake_proc, monkeypatch):  # noqa: F811
    _serve_process(fake_proc, home)

    class State:
        timeout_start_seconds = 90

    monkeypatch.setattr(start_serving, "unit_runtime_state", lambda: pytest.fail("the verdict asked the seam again"))
    got = _verdict(named_peers, home, "flash-next", main_pid=4242, unit_state=State())
    assert got.line.endswith("waiting up to 90s")


def test_liveness_a_real_unit_state_with_the_main_pid_and_the_default_timeout_waits_1800s(
        named_peers, home, fake_proc):  # noqa: F811
    """The sensed state is a `UnitRuntimeState`: its MainPID marks the unit's own `serve` as starting and its
    `TimeoutStartSec` (1800 s unless the manager says otherwise) is the bound."""
    _serve_process(fake_proc, home)
    state = units.UnitRuntimeState(None, 0, main_pid=4242)
    got = _verdict(named_peers, home, "flash-next", main_pid=state.main_pid, unit_state=state)
    assert (got.kind, got.pid) == ("starting", 4242)
    assert got.line == "autostart is starting flash-next (pid 4242) — waiting up to 1800s"


def test_liveness_a_serve_process_that_is_not_the_units_main_pid_is_not_starting(
        named_peers, home, fake_proc):  # noqa: F811
    _serve_process(fake_proc, home)
    state = units.UnitRuntimeState(None, 0, main_pid=9999)
    got = _verdict(named_peers, home, "flash-next", main_pid=state.main_pid, unit_state=state)
    assert got.kind != "starting"


@pytest.mark.parametrize("bad", [None, 0, -5, True, "90", float("nan")])
def test_a_timeout_start_that_is_no_positive_number_falls_back_to_the_default(bad):
    class State:
        timeout_start_seconds = bad

    assert start_serving.starting_seconds(State()) == start_serving.STARTING_WAIT_SECONDS
    assert start_serving.starting_seconds(None) == start_serving.STARTING_WAIT_SECONDS


def test_liveness_without_the_units_main_pid_the_serve_process_is_not_starting(
        named_peers, home, fake_proc):  # noqa: F811
    _serve_process(fake_proc, home)
    got = _verdict(named_peers, home, "flash-next", main_pid=None)
    assert got.kind != "starting"
    assert got.kind == "none"


def test_liveness_a_serve_process_outside_the_units_cgroup_is_not_starting(named_peers, home, fake_proc):  # noqa: F811
    _serve_process(fake_proc, home, cgroup="/user.slice/other.service")
    assert _verdict(named_peers, home, "flash-next", main_pid=4242).kind == "none"


def test_liveness_a_withheld_record_is_not_an_incumbent_the_verdict_names(home, fake_proc):  # noqa: F811
    seed(home, peers=[peer("flash-next", served_name="flash-next")])
    harness = load(home)
    _flash_launch(fake_proc, home)
    trust = MachineTrust(withheld={
        "config.yaml": frozenset({("launch_model", "extra_endpoints.flash-next.lifecycle.model")})})
    got = start_serving.liveness_verdict(harness, trust, record(harness, PRIMARY), home)
    assert got.kind == "none"


# ------------------------------------------------------------------------------------------ the wait


@pytest.fixture
def slept(monkeypatch):
    calls: list[float] = []

    async def fake_sleep(seconds, **kw):
        calls.append(seconds)

    monkeypatch.setattr(start_serving.asyncio, "sleep", fake_sleep)
    return calls


@pytest.mark.asyncio
async def test_wait_starting_is_true_when_the_probe_turns_true_on_its_second_poll(slept):
    answers = iter([False, True])
    got = await start_serving.wait_starting(lambda: next(answers), seconds=6, poll_seconds=3)
    assert got is True and slept == [3]


@pytest.mark.asyncio
async def test_wait_starting_gives_up_after_the_bound(slept):
    polled: list[int] = []

    async def probe():
        polled.append(1)
        return False

    assert await start_serving.wait_starting(probe, seconds=6, poll_seconds=3) is False
    assert slept == [3, 3] and len(polled) == 3


@pytest.mark.asyncio
async def test_wait_starting_treats_a_probe_that_raises_as_not_yet(slept):
    calls = iter([OSError("down"), True])

    def probe():
        got = next(calls)
        if isinstance(got, BaseException):
            raise got
        return got

    assert await start_serving.wait_starting(probe, seconds=6, poll_seconds=3) is True


@pytest.mark.asyncio
async def test_wait_starting_never_signals_anything(slept, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("signalled")

    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden)
    assert await start_serving.wait_starting(lambda: False, seconds=3, poll_seconds=3) is False


# ------------------------------------------------------------------------------------- the boot clause

KNOBS = AutosenseConfig()
CLAUSE = ("retried at boot: mtp off, -c halved · "
          "extra_endpoints.flash-next.lifecycle.speculative: off keeps the full window")


class Boot:
    """A `flash-next` peer whose model is a real (tiny) Swift-shaped GGUF header, a verified launch of
    it in the fake `/proc`, and helpers to plant a boot marker and a retry record."""

    def __init__(self, home, tmp_path, fake_proc) -> None:
        from tests.unit.test_tuning_derive import SWIFT

        self.home = home
        self.model = write_gguf(tmp_path / "flash.gguf", SWIFT)
        seed(home, peers=[peer("flash-next", served_name="flash-next", model=str(self.model))])
        self.harness = load(home)
        self.record = record(self.harness, "flash-next")
        self.key = lifecycle.record_key(self.record.spec)
        from localharness.provider.gguf_header import read_header
        from localharness.provider.autosense import FIT_KEYS, fit_header

        self.header = fit_header(read_header(self.record.spec.model, FIT_KEYS))
        fake_proc.process(4242, argv=(LLAMA, "-a", "flash-next", "-c", "40000", "--port", "8081"), exe=LLAMA,
                          start_time=777)
        _pidfile(home, "4242 777 8081")

    def marker(self, **over) -> units.Marker:
        fields = dict(record=self.key, binary=str(self.record.spec.binary), invocation="inv-1",
                      cause="death", shape="retry", planned_context=40_000, full_fit=80_000,
                      free_bytes=20 * 2**30, dropped=("draft-mtp",), at="2026-10-13T11:59:30+00:00")
        fields.update(over)
        return units.Marker(**fields)

    def plant(self, **over):
        directory = server.server_dir(self.home)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / units.BOOT_MARKER_NAME
        data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(self.marker(**over)).items()}
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def clause(self, *, unit_state="default", served=40_000):
        state = units.UnitRuntimeState("inv-1", 1) if unit_state == "default" else unit_state
        return start_serving.boot_clause(
            self.home, self.record, autosense=KNOBS, unit_state=state, served_window=served,
            header=self.header)

    def retry(self):
        return speed_stats.last_retry(ledger(self.home), self.key)


@pytest.fixture
def boot(home, tmp_path, fake_proc):  # noqa: F811
    return Boot(home, tmp_path, fake_proc)


def test_boot_clause_with_no_marker_is_none_and_writes_nothing(boot):
    assert boot.clause() is None
    assert boot.retry() is None


def test_boot_clause_for_a_death_marker_says_what_was_retried_once(boot):
    marker_path = boot.plant()
    assert boot.clause() == CLAUSE
    assert not marker_path.exists()  # the confirm removed the marker it served
    got = boot.retry()
    assert got is not None and (got.full_fit, got.dropped) == (80_000, ("draft-mtp",))
    assert boot.clause() is None  # nothing left to confirm: the clause prints once


def test_boot_clause_is_none_when_the_ledger_already_held_a_retry(boot):
    speed_stats.record_retry(ledger(boot.home), boot.key, full_fit=70_000, free_bytes=None,
                             dropped=["draft-mtp"], binary="/x/llama-server", kv_cache_type="q8_0",
                             draft_p_min=0.6)
    marker_path = boot.plant()
    assert boot.clause() is None
    assert not marker_path.exists()


def test_boot_clause_for_a_remembered_marker_is_none_and_the_marker_is_removed(boot):
    marker_path = boot.plant(cause="remembered", shape="full", dropped=())
    assert boot.clause() is None
    assert not marker_path.exists()
    assert boot.retry() is None


def test_boot_clause_without_a_unit_state_is_none_and_leaves_the_marker(boot):
    marker_path = boot.plant()
    assert boot.clause(unit_state=None) is None
    assert marker_path.exists()
    assert boot.retry() is None


def test_the_seam_delegates_to_the_units_runner_and_answers_none_under_the_hermetic_stub(boot, monkeypatch):
    marker_path = boot.plant()
    assert start_serving.unit_runtime_state() is None  # the suite's stub of the manager runner answers nothing
    assert boot.clause(unit_state=start_serving.unit_runtime_state()) is None
    assert marker_path.exists()
    asked: list = []
    sentinel = units.UnitRuntimeState("inv-9", 3)
    monkeypatch.setattr(units, "unit_runtime_state", lambda: asked.append(1) or sentinel)
    assert start_serving.unit_runtime_state() is sentinel and asked == [1]  # one ask per call, no memo
    assert start_serving.unit_runtime_state() is sentinel and asked == [1, 1]


def test_boot_clause_with_a_marker_bound_to_another_run_leaves_it_and_says_nothing(boot):
    marker_path = boot.plant(invocation="inv-other")
    assert boot.clause() is None
    assert marker_path.exists() and boot.retry() is None


def test_boot_clause_with_a_marker_for_another_record_says_nothing(boot):
    boot.plant(record="llamacpp:another")
    assert boot.clause() is None
    assert boot.retry() is None


def test_boot_clause_without_a_verified_process_records_nothing(boot, fake_proc):  # noqa: F811
    boot.plant()
    fake_proc.process(4242, argv=("/usr/bin/sleep", "600"), exe="/usr/bin/sleep", start_time=777)
    assert boot.clause() is None
    assert boot.retry() is None


def test_boot_clause_is_not_fooled_by_a_marker_that_claims_dropped_work(boot):
    """The clause is built from the recorded retry (the live argv decides `dropped`), never from the
    marker's own claims: a hand-set spec type on the live process records no mtp drop."""
    boot.plant(dropped=("draft-mtp",))
    server_dir = server.server_dir(boot.home)
    assert (server_dir / units.BOOT_MARKER_NAME).exists()
    proc_dir = os.path.join(procfs.PROC_ROOT, "4242")
    with open(os.path.join(proc_dir, "cmdline"), "wb") as fh:
        fh.write(b"\0".join(a.encode() for a in (
            LLAMA, "-a", "flash-next", "-c", "40000", "--port", "8081", "--spec-type", "draft-mtp")) + b"\0")
    got = boot.clause()
    assert got is not None and "mtp off" not in got
    assert got.startswith("retried at boot: -c halved")


def test_boot_clause_for_the_primary_names_server_speculative(home, tmp_path, fake_proc):  # noqa: F811
    from tests.unit.test_tuning_derive import SWIFT

    model = write_gguf(tmp_path / "primary.gguf", SWIFT)
    seed(home)
    data = (home / "config.yaml").read_text(encoding="utf-8")
    (home / "config.yaml").write_text(data.replace("/m/a.gguf", str(model)), encoding="utf-8")
    harness = load(home)
    primary = record(harness, PRIMARY)
    key = lifecycle.record_key(primary.spec)
    fake_proc.process(4242, argv=(LLAMA, "-a", PRIMARY, "-c", "40000", "--port", "8080"), exe=LLAMA, start_time=777)
    _pidfile(home, "4242 777 8080")
    from localharness.provider.gguf_header import read_header
    from localharness.provider.autosense import FIT_KEYS, fit_header

    directory = server.server_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    fields = dict(record=key, binary=str(primary.spec.binary), invocation="inv-1", cause="death",
                  shape="retry", planned_context=40_000, full_fit=80_000, free_bytes=None,
                  dropped=["draft-mtp"], at="2026-10-13T11:59:30+00:00")
    (directory / units.BOOT_MARKER_NAME).write_text(json.dumps(fields), encoding="utf-8")
    got = start_serving.boot_clause(
        home, primary, autosense=KNOBS, unit_state=units.UnitRuntimeState("inv-1", 1),
        served_window=40_000, header=fit_header(read_header(primary.spec.model, FIT_KEYS)))
    assert got == ("retried at boot: mtp off, -c halved · server.speculative: off "
                   "keeps the full window")


# ------------------------------------------------------------------------------------------- the swap


class SwapRig:
    """The three process boundaries a swap crosses, recorded in order, plus the launch lock."""

    def __init__(self, harness, home) -> None:
        self.harness = harness
        self.home = home
        self.events: list[str] = []
        self.stops: list[dict] = []
        self.waits: list[dict] = []
        self.alive = {"flash-next": False, PRIMARY: False}
        self.wait_raises: dict[str, BaseException] = {}
        self.printed: list[str] = []

    def print_err(self, text, **kw):
        self.printed.append(text)

    def reporter(self, key=None) -> start_serving.ConsoleSwapReporter:
        return start_serving.ConsoleSwapReporter(self.print_err, self.harness, key)


@pytest.fixture
def swap_rig(home, monkeypatch):
    seed(home, peers=[peer("flash-next", served_name="flash-next")])
    rig = SwapRig(load(home), home)
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])

    def fake_stop(config_dir, launch="binary", timeout_seconds=30.0, **kw):
        rig.events.append("stop")
        rig.stops.append(kw)
        rig.alive["flash-next"] = False
        rig.alive[PRIMARY] = False
        return True

    def fake_start(config_dir, cmd, **kw):
        joined = " ".join(cmd)
        rig.events.append("start:" + joined)
        name = "flash-next" if "flash-next.gguf" in joined else PRIMARY
        rig.alive[name] = True
        return 4321

    async def fake_wait(base_url, config_dir=None, **kw):
        rig.events.append(f"wait:{base_url}")
        rig.waits.append({"url": base_url, "kw": kw, "flash_alive": rig.alive["flash-next"]})
        if base_url in rig.wait_raises:
            raise rig.wait_raises[base_url]
        return [PRIMARY if base_url == PRIMARY_URL else "flash-next"]

    @contextlib.asynccontextmanager
    async def spy_lock(config_dir, *, on_wait=None):
        rig.events.append("lock-enter")
        try:
            yield
        finally:
            rig.events.append("lock-exit")

    monkeypatch.setattr(managed_server, "stop_server", fake_stop)
    monkeypatch.setattr(managed_server, "start_server", fake_start)
    monkeypatch.setattr(managed_server, "wait_ready", fake_wait)
    monkeypatch.setattr(lifecycle, "launch_lock", spy_lock)
    return rig


async def _run(rig: SwapRig, incumbent_name, target_name, **kw):
    incumbent = record(rig.harness, incumbent_name) if incumbent_name else None
    target = resolve_serving(rig.harness, MachineTrust(), target_name, config_dir=rig.home)
    assert isinstance(target, Serving), target
    reporter = kw.pop("reporter", None) or rig.reporter()
    return await start_serving.run_swap(
        rig.harness, incumbent, target, config_dir=rig.home, autosense=None, reporter=reporter,
        redact=reporter.redact, **kw)


@pytest.mark.asyncio
async def test_run_swap_stops_then_starts_then_waits_inside_the_launch_lock(swap_rig):
    swap_rig.alive[PRIMARY] = True
    outcome = await _run(swap_rig, PRIMARY, "flash-next")
    events = swap_rig.events
    assert events[0] == "lock-enter" and events[-1] == "lock-exit"
    assert [e.split(":")[0] for e in events[1:-1]] == ["stop", "start", "wait"]
    assert "/m/flash-next.gguf" in events[2]
    assert outcome.error is None and outcome.live.base_url == PEER_URL
    assert outcome.stopped is not None and outcome.stopped.peer is None
    assert swap_rig.printed == []  # a success prints nothing: the launch line is the caller's


@pytest.mark.asyncio
async def test_run_swap_with_nothing_to_stop_just_launches(swap_rig):
    outcome = await _run(swap_rig, None, "flash-next")
    assert [e.split(":")[0] for e in swap_rig.events] == ["lock-enter", "start", "wait", "lock-exit"]
    assert outcome.stopped is None and outcome.error is None


@pytest.mark.asyncio
async def test_run_swap_to_a_record_that_cannot_be_launched_stops_nothing(home, monkeypatch, swap_rig):
    seed(home, peers=[peer("flash-next", served_name="flash-next"), peer("remote", 8082, lifecycle=False)])
    swap_rig.harness = load(home)
    target = model_ops.Serving(record(swap_rig.harness, "remote"), RUNG_PRESET, True)
    reporter = swap_rig.reporter()
    with pytest.raises(ValueError):
        await start_serving.run_swap(
            swap_rig.harness, record(swap_rig.harness, PRIMARY), target, config_dir=home,
            autosense=None, reporter=reporter, redact=reporter.redact)
    assert swap_rig.events == []  # not even the lock, and certainly no stop


@pytest.mark.asyncio
async def test_a_planted_key_in_a_failed_launch_tail_never_reaches_the_console_or_the_error(swap_rig):
    tail = f"the model server exited. Log tail:\nBearer sk-planted ... {PROVIDER_KEY} ... done"
    swap_rig.wait_raises[PEER_URL] = RuntimeError(tail)
    swap_rig.alive[PRIMARY] = True
    outcome = await _run(swap_rig, PRIMARY, "flash-next")
    assert outcome.stage == "launch" and outcome.error is not None
    failed = [line for line in swap_rig.printed if line.startswith("Launch of flash-next failed:")]
    assert len(failed) == 1
    for secret in ("sk-planted", PROVIDER_KEY):
        assert secret not in failed[0]
        assert secret not in str(outcome.error)
        assert not any(secret in line for line in swap_rig.printed)


@pytest.mark.asyncio
async def test_the_console_reporter_redacts_the_key_file_key_and_prints_with_markup_off(swap_rig):
    seen: list[dict] = []

    def print_err(text, **kw):
        seen.append({"text": text, **kw})

    rep = start_serving.ConsoleSwapReporter(print_err, swap_rig.harness, "k-from-a-file-9")
    await rep.line("log: k-from-a-file-9 and https://user:hunter2@host/x [bold]")
    await rep.freed()
    await rep.launched(None)
    await rep.restored(None)
    assert len(seen) == 1
    assert "k-from-a-file-9" not in seen[0]["text"] and "hunter2" not in seen[0]["text"]
    assert seen[0]["markup"] is False and seen[0]["soft_wrap"] is True
    assert rep.redact("Bearer sk-planted") == model_ops.redact_for_channel(
        "Bearer sk-planted", swap_rig.harness, "k-from-a-file-9")


# ---------------------------------------------------------------- stop then launch (D-18, the 30-minute hang)


@pytest.mark.asyncio
async def test_stop_then_launch_another_records_live_launch_is_stopped_before_the_selected_is_started(
        swap_rig, home, fake_proc, probes):  # noqa: F811
    """`/model flash-next` then quit leaves flash-next running on :8081 with its pidfile. A later
    `start` selects the primary on `server:`, nothing answers, and a verified launch of flash-next
    names the pidfile: the swap's incumbent, stopped first. No wait on :8080 is ever entered while
    flash-next lives."""
    fake_proc.process(4242, argv=(LLAMA, "-a", "flash-next", "--port", "8081"), exe=LLAMA, start_time=777)
    _pidfile(home, "4242 777 8081")
    swap_rig.alive["flash-next"] = True
    harness = swap_rig.harness
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    assert (serving.record.name, serving.rung) == (PRIMARY, RUNG_DEFAULT)
    sense = await sense_serving(harness, launchable_loopback_others(harness, MachineTrust(), serving.record))
    assert decide(serving, selected_answers=False, sense=sense, last_served=None).kind == "launch"

    verdict = start_serving.liveness_verdict(harness, MachineTrust(), serving.record, home)
    assert (verdict.kind, verdict.record.name) == ("other", "flash-next")
    outcome = await start_serving.run_swap(
        harness, verdict.record, serving, config_dir=home, autosense=None,
        reporter=swap_rig.reporter(), redact=swap_rig.reporter().redact)

    events = [e.split(":")[0] for e in swap_rig.events if not e.startswith("lock")]
    assert events == ["stop", "start", "wait"]
    flash = verdict.record.spec
    assert swap_rig.stops[0]["expect"] == lifecycle.strategy_for(flash).launch_identity(flash)
    primary_start = next(e for e in swap_rig.events if e.startswith("start:"))
    assert "/m/a.gguf" in primary_start and "flash-next.gguf" not in primary_start
    primary_waits = [w for w in swap_rig.waits if w["url"] == PRIMARY_URL]
    assert primary_waits and all(not w["flash_alive"] for w in primary_waits)
    assert all("expect" not in w["kw"] for w in swap_rig.waits)  # no "still loading" wait was entered
    assert outcome.error is None and outcome.stopped.peer == "flash-next"


# ------------------------------------------------------------------------------------------ the line


def test_the_swap_launch_line_names_who_is_stopped_and_what_is_started():
    line = start_serving.swap_launch_line(
        incumbent=PRIMARY, target="flash-next", inside="kv q8_0 · mtp (1 layer); ")
    assert line == ("stopping qwen-27b-swift-1.5…, starting flash-next "
                    "(kv q8_0 · mtp (1 layer); model load can take several minutes)...")


def test_the_launch_line_with_nothing_to_stop_is_the_plain_start_form():
    line = start_serving.swap_launch_line(incumbent=None, target="flash-next", inside="")
    assert line == "starting flash-next (model load can take several minutes)..."
    assert line == start_serving.START_TARGET_LINE.format(target="flash-next", inside="")


def test_the_remaining_start_strings_are_the_spec_wording():
    assert start_serving.STARTING_GAVE_UP_LINE.format(pid=7, seconds=600) == (
        "autostart's server (pid 7) did not answer within 600s — not touched")
    assert start_serving.NOT_LAUNCHABLE_LINE.format(name="gemma", base_url="http://h/v1") == (
        "gemma (http://h/v1) is not answering and the harness does not launch it — starting on the primary")
    assert start_serving.STARTING_WAIT_SECONDS == 600
