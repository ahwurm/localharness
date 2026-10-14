"""`cli/start_serving.py`: what `start` decides once it knows what serves (spec 20 §8.2, §1 item 3, §9.3).

Offline: `list_live_models` is faked (`Probes`), harnesses come from `ConfigLoader` over a tmp dir,
the `/proc` side is the fake tree of `fake_proc`, the unit's state is patched, the server boundaries
a swap crosses are recorded fakes. No socket, no process, no service manager (D-30, D-32). Test
fakes take `**kw`.
"""
from __future__ import annotations

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
from localharness.provider import procfs, server, speed_stats
from tests.unit.procfs_support import fake_proc  # noqa: F401
from tests.unit.test_sense_serving import (  # noqa: F401
    FLASH_ACTIVE,
    PEER_URL,
    PRIMARY,
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
    monkeypatch.setattr(start_serving, "unit_runtime_state", lambda: None)
    got = _verdict(named_peers, home, "flash-next", main_pid=4242)
    assert (got.kind, got.pid, got.record.name) == ("starting", 4242, "flash-next")
    assert got.line == "autostart is starting flash-next (pid 4242) — waiting up to 600s"


def test_liveness_the_starting_bound_is_the_units_timeout_start_when_it_has_one(
        named_peers, home, fake_proc, monkeypatch):  # noqa: F811
    _serve_process(fake_proc, home)

    class State:
        timeout_start_seconds = 90

    monkeypatch.setattr(start_serving, "unit_runtime_state", lambda: State())
    assert _verdict(named_peers, home, "flash-next", main_pid=4242).line.endswith("waiting up to 90s")


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
