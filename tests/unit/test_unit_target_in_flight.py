"""The in-flight form of `unit-target` and the ladder rung it earns (spec 20 §8.1, R-14-38..R-14-41, D-14, D-20).

While `UnitStrategy.activate` restarts the harness unit it writes `unit-target` as `<name> <pid> <start>` (the
requesting process), and `resolve_serving` ranks that name above `pending_preset` ONLY while the requester still
runs. Every exit of `activate` leaves the plain `<name>` (or the prior value) in ONE `finally`.

Nothing real runs: the service manager is `ScriptedManager`, the unit's `serve` is a `FakeServe` thread whose
record is chosen through the real ladder, `/proc` is `fake_proc` (the test process is registered in it), the HTTP
wait answers through `server._TRANSPORT`.
"""
from __future__ import annotations

import asyncio
import logging
import os

import pytest
import yaml

from localharness.cli import model_ops
from localharness.cli.model_ops import RUNG_PENDING, RUNG_UNIT, GoneSelection, Serving
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig
from localharness.provider import lifecycle, procfs, server, units
from localharness.provider.server import server_dir
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.unit_strategy_support import PRIMARY, unit_world, vanish

fcntl = pytest.importorskip("fcntl")
AUTOSENSE = AutosenseConfig()
OWN_START = 111  # the start time `unit_world` registers for the test process


@pytest.fixture(autouse=True)
def _no_leaked_holds():
    yield
    assert lifecycle._HELD == {}


@pytest.fixture
def world(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    made = unit_world(tmp_path, monkeypatch, fake_proc)
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("the harness started a server"))
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: pytest.fail("the harness stopped a process"))
    yield made
    made.stop_servers()


@pytest.fixture
def cfg(tmp_path):
    return tmp_path / "cfg"


# ------------------------------------------------------------------------------------------- helpers


def plant(g, body: str) -> None:
    """The `unit-target` file with exactly `body` (a writer other than `write_target`: a planted or stale file)."""
    directory = server_dir(g)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / units.TARGET_NAME).write_text(body, encoding="utf-8")


def raw(g) -> str | None:
    path = server_dir(g) / units.TARGET_NAME
    return path.read_text(encoding="utf-8") if path.exists() else None


def own(start: int = OWN_START) -> str:
    return f"{os.getpid()} {start}"


def fresh(g):
    return ConfigLoader(config_dir=g).load_harness()


def ladder(g, trust=None):
    """The real ladder over a harness loaded fresh from the global dir, as `serve` reads it (D-14)."""
    return model_ops.resolve_serving(fresh(g), trust or MachineTrust(), None, config_dir=g)


def set_pending(world, name: str) -> None:
    model_ops.write_pending_preset(world.harness, name, config_dir=world.g)


def pending_in_overlay(g):
    path = g / "overrides.yaml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("pending_preset") if path.exists() else None


def activate(world, spec, *, peer=None, error=None, timeout=60, autosense=AUTOSENSE):
    """`UnitStrategy.activate` inside the caller's launch lock. With `error`, returns the raised exception."""
    async def go():
        async with lifecycle.launch_lock(world.g):
            strat = world.strategy(spec)
            call = strat.activate(spec, world.g, world.url(peer), timeout_seconds=timeout, peer=peer,
                                  autosense=autosense)
            if error is None:
                return await call
            with pytest.raises(error) as caught:
                await call
            return caught.value

    return asyncio.run(go())


def serving_the_ladder(world, spec, seen: list):
    """A `FakeServe` that starts when the unit is restarted and presents whichever record the real ladder
    chooses at that moment; `seen` collects the `Serving` and the file text it was chosen against."""
    def choose():
        text = raw(world.g)
        got = ladder(world.g)
        seen.append((got, text))
        return got.record.spec

    return world.serve_on_restart(spec, choose=choose)


def cancel_on_poll(world, holder: list, at: int = 2) -> None:
    world.http(answer_after=10**6, hook=lambda n: holder[0].cancel() if n == at else None)


def activate_cancelled(world, spec):
    holder: list = []

    async def go():
        async def locked():
            async with lifecycle.launch_lock(world.g):
                await world.strategy(spec).activate(
                    spec, world.g, world.url(), timeout_seconds=60, autosense=AUTOSENSE)

        task = asyncio.ensure_future(locked())
        holder.append(task)
        with pytest.raises(asyncio.CancelledError):
            await task

    world.fast_polls()
    world.fast_loan()
    cancel_on_poll(world, holder)
    serving = world.serve_on_restart(spec)
    asyncio.run(go())
    serving.join()


# =================================================================================== the forms (units)


@pytest.fixture
def registered(fake_proc):  # noqa: F811
    fake_proc.process(os.getpid(), argv=("python",), start_time=OWN_START, comm="python")
    return fake_proc


def test_the_in_flight_write_names_this_process_and_both_readers_see_it(cfg, registered):
    units.write_target(cfg, "flash-next", in_flight=True)
    assert raw(cfg) == f"flash-next {os.getpid()} {OWN_START}\n"
    assert units.read_target(cfg) == "flash-next"
    assert units.in_flight_target(cfg) == "flash-next"
    assert units.in_flight_request(cfg) == units.InFlight("flash-next", os.getpid(), OWN_START)


def test_the_plain_write_is_the_bare_name_and_has_no_requester(cfg, registered):
    units.write_target(cfg, "flash-next")
    assert raw(cfg) == "flash-next\n"
    assert units.read_target(cfg) == "flash-next"
    assert units.in_flight_target(cfg) is None and units.in_flight_request(cfg) is None


def test_a_requester_of_another_uid_is_never_live(cfg, registered, monkeypatch):
    plant(cfg, "flash-next 1 5\n")
    asked = []

    def read(pid):
        asked.append(pid)
        return procfs.ProcRefusal("is owned by another uid")

    monkeypatch.setattr(procfs, "read_process", read)
    assert units.in_flight_request(cfg) is None and units.in_flight_target(cfg) is None
    assert units.read_target(cfg) == "flash-next"
    assert asked == [1, 1]


def test_a_reused_pid_and_a_pid_absent_from_proc_are_not_live(cfg, registered):
    plant(cfg, f"flash-next {os.getpid()} {OWN_START + 1}\n")
    assert units.in_flight_target(cfg) is None and units.read_target(cfg) == "flash-next"
    plant(cfg, "flash-next 987654 5\n")
    assert units.in_flight_target(cfg) is None and units.read_target(cfg) == "flash-next"


@pytest.mark.parametrize("body", [
    "flash-next 12 x\n", "flash-next 0 5\n", "flash-next 12\n", "flash-next 12 5 7\n", "flash-next  12 5\n",
    "flash-next 012 5\n", "flash-next 12345678901 5\n", "flash-next 12 123456789012345678901\n",
    "flash-next -1 5\n", "flash-next 1 5 \n x", "has space 12 5\n", " \n", "",
])
def test_off_shape_bodies_read_none_from_every_reader(cfg, registered, body):
    plant(cfg, body)
    assert units.read_target(cfg) is None
    assert units.in_flight_target(cfg) is None and units.in_flight_request(cfg) is None


def test_a_link_a_fifo_and_an_oversize_body_read_none(cfg, registered, tmp_path):
    good = tmp_path / "elsewhere"
    good.write_text(f"flash-next {own()}\n", encoding="utf-8")
    directory = server_dir(cfg)
    directory.mkdir(parents=True)
    link = directory / units.TARGET_NAME
    link.symlink_to(good)
    assert units.read_target(cfg) is None and units.in_flight_request(cfg) is None
    link.unlink()
    os.mkfifo(link)
    assert units.read_target(cfg) is None and units.in_flight_request(cfg) is None
    link.unlink()
    link.write_text("a" * (units.TARGET_CAP + 1), encoding="utf-8")
    assert units.read_target(cfg) is None and units.in_flight_request(cfg) is None


def test_a_writer_that_cannot_read_its_own_record_raises_and_leaves_the_previous_file(cfg, fake_proc):  # noqa: F811
    units.write_target(cfg, "a")
    before = raw(cfg)
    with pytest.raises(OSError, match="cannot name this process"):
        units.write_target(cfg, "b", in_flight=True)
    assert raw(cfg) == before == "a\n"


def test_a_name_that_is_not_one_record_name_writes_nothing_in_either_form(cfg, registered):
    for in_flight in (False, True):
        with pytest.raises(ValueError):
            units.write_target(cfg, "two words", in_flight=in_flight)
    assert raw(cfg) is None


# =================================================================================== the ladder's rung


def test_a_live_in_flight_target_outranks_pending_preset(world):
    set_pending(world, "qwen-27b")
    units.write_target(world.g, "flash-next", in_flight=True)
    got = ladder(world.g)
    assert isinstance(got, Serving) and got.record.name == "flash-next"
    assert got.rung == RUNG_UNIT and got.explicit is False and got.gone == ()


def test_a_plain_target_ranks_below_pending_preset(world):
    set_pending(world, "qwen-27b")
    units.write_target(world.g, "flash-next")
    got = ladder(world.g)
    assert got.record.name == "qwen-27b" and got.rung == RUNG_PENDING and got.explicit is True


def test_a_gone_requester_ranks_below_pending_preset(world):
    set_pending(world, "qwen-27b")
    plant(world.g, f"flash-next {own(OWN_START + 1)}\n")
    got = ladder(world.g)
    assert got.record.name == "qwen-27b" and got.rung == RUNG_PENDING


def test_without_a_pending_preset_a_gone_requester_is_the_plain_unit_target(world):
    plant(world.g, f"flash-next {own(OWN_START + 1)}\n")
    got = ladder(world.g)
    assert got.record.name == "flash-next" and got.rung == RUNG_UNIT and got.explicit is False


def test_a_live_in_flight_name_matching_no_record_is_one_gone_selection_and_the_ladder_goes_on(world):
    set_pending(world, "qwen-27b")
    plant(world.g, f"nonesuch {own()}\n")
    got = ladder(world.g)
    assert got.record.name == "qwen-27b" and got.rung == RUNG_PENDING
    assert len(got.gone) == 1 and isinstance(got.gone[0], GoneSelection)
    assert got.gone[0].store == RUNG_UNIT and got.gone[0].name == "nonesuch"


def test_the_ladder_reads_nothing_new_when_no_swap_is_in_flight(world):
    """Without a requester every order is the spec's: pending, then the plain target, then the primary."""
    assert ladder(world.g).record.name == PRIMARY
    units.write_target(world.g, "flash-next")
    assert ladder(world.g).record.name == "flash-next"
    set_pending(world, "qwen-27b")
    assert ladder(world.g).record.name == "qwen-27b"


# ================================================================================ end to end: a swap


def test_end_to_end_a_swap_is_served_with_its_target_beside_a_stale_pending_preset(world):
    set_pending(world, "qwen-27b")
    world.ready()
    world.http(served="flash-next")
    spec = world.peer("flash-next")
    seen: list = []
    serving = serving_the_ladder(world, spec, seen)
    live = activate(world, spec, peer="flash-next")
    serving.join()
    assert serving.error is None and live.served_models == ["flash-next"]
    got, text = seen[0]
    assert got.record.name == "flash-next" and got.rung == RUNG_UNIT
    assert text == f"flash-next {own()}\n"                      # what the restart saw: the in-flight form
    assert raw(world.g) == "flash-next\n" and units.read_target(world.g) == "flash-next"
    assert units.in_flight_target(world.g) is None
    assert pending_in_overlay(world.g) == "qwen-27b"            # activate never touches pending_preset


def test_end_to_end_pending_preset_is_untouched_by_a_swap_that_fails_after_its_restart(world):
    set_pending(world, "qwen-27b")
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=10**6)
    world.manager.on_restart.append(lambda: world.manager.base.update(ActiveState="failed", ExecMainStatus=1))
    exc = activate(world, world.peer("flash-next"), peer="flash-next", error=RuntimeError)
    assert exc.__cause__ is None
    assert raw(world.g) == "flash-next\n"
    assert pending_in_overlay(world.g) == "qwen-27b"


# ====================================================================================== every exit


def test_exit_success_leaves_the_plain_name(world):
    world.ready()
    serving = world.serve_on_restart(world.primary)
    activate(world, world.primary)
    serving.join()
    assert raw(world.g) == "server:\n" and units.in_flight_target(world.g) is None


def test_exit_unit_failed_after_the_restart_leaves_the_plain_name_then_raises_the_scrubbed_error(world):
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=10**6)
    world.manager.journal("loading\n")
    world.manager.on_restart.append(lambda: world.manager.base.update(ActiveState="failed", ExecMainStatus=78))
    exc = activate(world, world.primary, error=RuntimeError)
    assert "loading" in str(exc)
    assert raw(world.g) == "server:\n" and units.in_flight_target(world.g) is None


def test_exit_cancel_restores_a_prior_value_in_the_plain_form(world):
    plant(world.g, f"flash-next {own()}\n")                      # the prior was itself an in-flight body
    activate_cancelled(world, world.primary)
    assert raw(world.g) == "flash-next\n" and units.in_flight_target(world.g) is None


def test_exit_cancel_with_no_prior_leaves_no_file(world):
    activate_cancelled(world, world.primary)
    assert raw(world.g) is None


def test_exit_refusal_before_the_restart_restores_the_prior_and_asks_for_no_verb(world):
    units.write_target(world.g, "flash-next")
    world.manager.base["DropInPaths"] = "/home/x/.config/systemd/user/localharness-llm.service.d/o.conf"
    activate(world, world.primary, error=server.LaunchRefused)
    assert world.manager.verbs == []
    assert raw(world.g) == "flash-next\n"


def test_exit_refusal_before_the_restart_with_no_prior_leaves_no_file(world):
    world.manager.base["DropInPaths"] = "/home/x/.config/systemd/user/localharness-llm.service.d/o.conf"
    activate(world, world.primary, error=server.LaunchRefused)
    assert raw(world.g) is None


def test_exit_reset_failed_raising_restores_the_prior_and_records_no_restart(world):
    units.write_target(world.g, "flash-next")
    world.manager.answer_prefix("systemctl", "--user", "reset-failed", out=None)
    activate(world, world.primary, error=RuntimeError)
    assert [v for v in world.manager.verbs if v[0] == "restart"] == []
    assert raw(world.g) == "flash-next\n"


def test_exit_restart_verb_failing_restores_the_prior(world):
    units.write_target(world.g, "flash-next")
    world.manager.answer_prefix("systemctl", "--user", "restart", out="", code=1, err="no\n")
    activate(world, world.primary, error=RuntimeError)
    assert raw(world.g) == "flash-next\n"


def test_exit_in_flight_write_failing_is_a_refusal_with_no_verb_and_the_prior_untouched(world):
    units.write_target(world.g, "flash-next")
    before = (server_dir(world.g) / units.TARGET_NAME).read_bytes()
    vanish(world.proc, os.getpid())                              # this process cannot name itself
    exc = activate(world, world.primary, error=RuntimeError)
    assert "could not record the unit target" in str(exc) and exc.__cause__ is None
    assert world.manager.calls == []
    assert (server_dir(world.g) / units.TARGET_NAME).read_bytes() == before


# ================================================================== an exit nobody anticipated


class Abort(BaseException):
    """An exit no handler of `activate` names."""


def _after_restart_raise(world, monkeypatch, error, on_call=4):
    """`units.unit_runtime_state`, the read the wait polls, raises `error` on its `on_call`-th call after the
    restart (the real read before it)."""
    real = units.unit_runtime_state
    seen = {"n": 0}

    def flaky():
        if world.manager.restarted:
            seen["n"] += 1
            if seen["n"] == on_call:
                raise error
        return real()

    monkeypatch.setattr(units, "unit_runtime_state", flaky)


def test_arbitrary_exit_an_exception_out_of_the_unit_read_propagates_as_raised_with_the_plain_form(world, monkeypatch):
    boom = LookupError("scripted")
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=10**6)
    _after_restart_raise(world, monkeypatch, boom)
    serving = world.serve_on_restart(world.primary)
    got = activate(world, world.primary, error=LookupError)
    serving.join()
    assert got is boom                                           # the same object, nothing built over it
    assert raw(world.g) == "server:\n"
    assert units.in_flight_target(world.g) is None and units.read_target(world.g) == "server:"


def test_arbitrary_exit_a_base_exception_out_of_the_wait_propagates_and_the_form_is_plain(world, monkeypatch):
    world.fast_polls()
    world.fast_loan()
    world.http()
    boom = Abort()

    async def wait(*args, **kw):
        raise boom

    monkeypatch.setattr(server, "wait_ready", wait)
    serving = world.serve_on_restart(world.primary)
    got = activate(world, world.primary, error=Abort)
    serving.join()
    assert got is boom
    assert raw(world.g) == "server:\n" and units.in_flight_target(world.g) is None


def test_arbitrary_exit_a_timeout_out_of_the_wait_leaves_the_plain_form(world, monkeypatch):
    world.fast_polls()
    world.fast_loan()
    world.http()

    async def wait(*args, **kw):
        raise TimeoutError("scripted")

    monkeypatch.setattr(server, "wait_ready", wait)
    serving = world.serve_on_restart(world.primary)
    got = activate(world, world.primary, error=TimeoutError)
    serving.join()
    assert isinstance(got, TimeoutError)
    assert raw(world.g) == "server:\n" and units.in_flight_target(world.g) is None


def _plain_writes_fail(monkeypatch):
    real = units.write_target

    def write(config_dir, name, *, in_flight=False):
        if not in_flight:
            raise OSError(28, "No space left on device")
        return real(config_dir, name, in_flight=in_flight)

    monkeypatch.setattr(units, "write_target", write)


def test_oserror_on_the_plain_write_falls_back_to_removing_the_file_and_never_masks_success(
        world, monkeypatch, caplog):
    world.ready()
    _plain_writes_fail(monkeypatch)
    serving = world.serve_on_restart(world.primary)
    with caplog.at_level(logging.DEBUG, logger=lifecycle.log.name):
        live = activate(world, world.primary)
    serving.join()
    assert live.served_models == [PRIMARY]
    assert raw(world.g) is None                                  # removed: an unlink needs no space
    assert any("No space left" in (r.exc_text or "") or "settle" in r.getMessage() for r in caplog.records)


def test_oserror_on_the_plain_write_never_replaces_an_arbitrary_exit(world, monkeypatch):
    world.fast_polls()
    world.fast_loan()
    world.http()
    boom = Abort()

    async def wait(*args, **kw):
        raise boom

    monkeypatch.setattr(server, "wait_ready", wait)
    _plain_writes_fail(monkeypatch)
    serving = world.serve_on_restart(world.primary)
    got = activate(world, world.primary, error=Abort)
    serving.join()
    assert got is boom and raw(world.g) is None


# ============================================================ restore, boot and a later pending_preset


def test_restore_is_served_with_the_incumbent_not_the_pending_preset(world):
    set_pending(world, "qwen-27b")
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=10**6)
    world.manager.on_restart.append(lambda: world.manager.base.update(ActiveState="failed", ExecMainStatus=1))
    activate(world, world.peer("flash-next"), peer="flash-next", error=RuntimeError)
    assert raw(world.g) == "flash-next\n"                        # the failed swap left its own plain name

    world.manager.on_restart.clear()                             # the swap's restore re-activates the incumbent
    world.manager.base.update(ActiveState="active", ExecMainStatus=0)
    world.ready()
    seen: list = []
    serving = serving_the_ladder(world, world.primary, seen)
    activate(world, world.primary)
    serving.join()
    got, text = seen[0]
    assert got.record.name == PRIMARY and got.rung == RUNG_UNIT   # not qwen-27b
    assert text == f"server: {own()}\n"
    assert raw(world.g) == "server:\n"
    assert pending_in_overlay(world.g) == "qwen-27b"


def test_boot_with_a_stale_plain_or_gone_requester_target_serves_the_pending_preset(world):
    set_pending(world, "qwen-27b")
    plant(world.g, "server:\n")
    assert ladder(world.g).record.name == "qwen-27b"
    plant(world.g, f"server: {own(OWN_START + 1)}\n")            # a requester that is gone
    got = ladder(world.g)
    assert got.record.name == "qwen-27b" and got.rung == RUNG_PENDING


def test_boot_without_a_pending_preset_serves_the_target(world):
    set_pending(world, "qwen-27b")
    plant(world.g, f"server: {own(OWN_START + 1)}\n")
    asyncio.run(model_ops.persist_landed(
        fresh(world.g), ladder(world.g).record, outcome="swap", config_dir=world.g))
    assert pending_in_overlay(world.g) is None
    got = ladder(world.g)
    assert got.record.name == PRIMARY and got.rung == RUNG_UNIT


def test_a_later_pending_preset_wins_once_the_swap_has_ended(world):
    world.ready()
    world.http(served="flash-next")
    spec = world.peer("flash-next")
    serving = world.serve_on_restart(spec)
    activate(world, spec, peer="flash-next")
    serving.join()
    assert ladder(world.g).record.name == "flash-next"
    harness = fresh(world.g)
    target = next(r for r in model_ops.serving_records(harness) if r.name == "qwen-27b")
    asyncio.run(model_ops.select_pending(harness, target, config_dir=world.g))
    got = ladder(world.g)
    assert got.record.name == "qwen-27b" and got.rung == RUNG_PENDING
