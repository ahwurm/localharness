"""`ServerRecovery` (D-15, D-16, D-17, D-22, D-46, D-12): one flight per pair waits on the client's own
address with the client's own key, relaunches a dead launched process at most once inside the bound, fails
with one readable sentence to every awaiter, and is cancelled only when its last awaiter leaves.

Hermetic (D-34): `/models` is an `httpx.MockTransport` installed as `server._TRANSPORT`, or `wait_ready`
is replaced by a recorder; every config dir is `tmp_path`; no server, no port, nothing under the owner's
`~/.localharness`. `asyncio.sleep` is never patched (the poll loops need real suspension).
"""
from __future__ import annotations

import asyncio
import dataclasses
import gc
import logging
import os

import httpx
import pytest
from pydantic import SecretStr

from localharness.provider import client as client_mod
from localharness.provider import lifecycle, server
from localharness.provider.lifecycle import ServerRecovery
from tests.unit.reconnect_support import (
    DroppingChat,
    ModelsEndpoint,
    RecordingRelaunch,
    assert_no_key_leak,
    text_answer,
    write_dead_pidfile,
    write_live_pidfile,
    write_serve_log,
)

fcntl = pytest.importorskip("fcntl")

URL = "http://127.0.0.1:9/v1"
SENTINEL = "sk-sentinel-4f2a9c71-never-in-a-log"


@pytest.fixture(autouse=True)
def _fast_visibility(monkeypatch):
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.02)
    yield
    assert lifecycle._HELD == {}


@pytest.fixture
def models(monkeypatch):
    """Install a `/models` handler as the transport `wait_ready` uses; returns the handler."""

    def install(endpoint: ModelsEndpoint) -> ModelsEndpoint:
        monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(endpoint))
        return endpoint

    return install


class Announce:
    """An `on_lost` that counts how often it was awaited."""

    def __init__(self) -> None:
        self.count = 0

    async def __call__(self) -> None:
        self.count += 1


def _lockable(config_dir) -> bool:
    fd = os.open(server.lock_path(config_dir), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _served_while_alive(config_dir) -> ModelsEndpoint:
    """A `/models` endpoint that answers only while the pidfile names a live process: a dead server does
    not answer, so the flight's look at the address before a relaunch finds nothing there."""
    return ModelsEndpoint(down=lambda: server.server_pid(config_dir) is None)


class WaitReadyRecorder:
    """Stands in for `server.wait_ready`: records the keywords of each call, optionally sleeps, then
    returns or raises the next scripted outcome (an exception instance is raised, anything else returns)."""

    def __init__(self, outcomes=None, delay: float = 0.0) -> None:
        self.outcomes = list(outcomes or [])
        self.delay = delay
        self.calls: list[dict] = []

    async def __call__(self, base_url, config_dir=None, timeout_seconds=1800.0, poll_seconds=3.0,
                       on_poll=None, *, api_key=None, runtime=None, liveness=False):
        self.calls.append({"base_url": base_url, "config_dir": config_dir,
                           "timeout_seconds": timeout_seconds, "poll_seconds": poll_seconds,
                           "api_key": api_key, "runtime": runtime, "liveness": liveness})
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.outcomes.pop(0) if self.outcomes else ["m"]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


# --- construction ---------------------------------------------------------------------------- #


def test_a_pair_builds_compares_by_its_public_fields_and_is_frozen(tmp_path):
    async def relaunch():
        return None

    ServerRecovery(None, None, None)
    a = ServerRecovery(URL, tmp_path, relaunch, "llamacpp")
    b = ServerRecovery(URL, tmp_path, relaunch, "llamacpp")
    assert a == b
    assert a._flight is not b._flight  # each pair has its own flight slot
    assert "flight" not in repr(a).lower() and "_Flight" not in repr(a)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.base_url = "http://elsewhere/v1"


# --- the poll: the client's own address and key ---------------------------------------------- #


async def test_a_server_that_comes_back_after_a_few_polls_ends_the_wait(models):
    endpoint = models(ModelsEndpoint(down_for=3))
    announce = Announce()
    pair = ServerRecovery(None, None, None)
    await pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=announce)
    assert endpoint.calls == 4
    assert announce.count == 1


async def test_a_blip_that_answers_the_first_poll_announces_nothing(models):
    endpoint = models(ModelsEndpoint(down_for=0))
    announce = Announce()
    await ServerRecovery(None, None, None).recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=announce)
    assert endpoint.calls == 1
    assert announce.count == 0


async def test_the_poll_carries_the_callers_key_and_logs_none_of_it(models, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    endpoint = models(ModelsEndpoint(down_for=1))
    pair = ServerRecovery(None, None, None)
    await pair.recover(base_url=URL, api_key=SecretStr(SENTINEL), wait_seconds=5, on_lost=Announce())
    assert endpoint.auth and all(a == "Bearer " + SENTINEL for a in endpoint.auth)
    assert endpoint.urls[0] == URL + "/models"
    assert_no_key_leak(SENTINEL, repr(pair), caplog=caplog, capsys=capsys)


async def test_a_keyless_server_gets_no_authorization_header(models):
    endpoint = models(ModelsEndpoint(down_for=1))
    await ServerRecovery(None, None, None).recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert endpoint.auth and all(a is None for a in endpoint.auth)


async def test_wait_ready_gets_the_pairs_dir_and_runtime_and_the_remaining_bound(monkeypatch, tmp_path):
    dead = RuntimeError("llama.cpp exited during startup. Log tail:\nx")
    nothing_there = TimeoutError("llama.cpp not ready after 0s.")
    recorder = WaitReadyRecorder([dead, nothing_there, ["m"]])
    monkeypatch.setattr(server, "wait_ready", recorder)
    relaunch = RecordingRelaunch()
    pair = ServerRecovery(URL, tmp_path, relaunch, "llamacpp")
    await pair.recover(base_url=URL, api_key=SecretStr(SENTINEL), wait_seconds=30, on_lost=Announce())
    first, look, second = recorder.calls
    for call in (first, look, second):
        assert call["base_url"] == URL
        assert call["runtime"] == "llamacpp"
        assert call["api_key"] == SENTINEL
        assert call["liveness"] is True
    for call in (first, second):
        assert call["config_dir"] == tmp_path
        assert call["poll_seconds"] == 0.02  # the constant at call time, not a captured 2.0
    assert first["timeout_seconds"] <= 30
    assert second["timeout_seconds"] < first["timeout_seconds"]
    # the look at the address before the relaunch: no pidfile check, one poll long
    assert look["config_dir"] is None
    assert look["timeout_seconds"] == look["poll_seconds"] == 0.02
    assert relaunch.calls == 1


class Answering:
    """A `/models` handler that answers 503 for `down_for` calls, then `status` with `body` (raw bytes)."""

    def __init__(self, status: int, body: bytes, down_for: int = 0) -> None:
        self.status, self.body, self.down_for = status, body, down_for
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.down_for:
            return httpx.Response(503, json={"error": "loading"})
        return httpx.Response(self.status, content=self.body)


@pytest.mark.parametrize(("status", "body"), [
    (401, b'{"error": "invalid api key"}'),   # a hand-run server restarted with another key
    (403, b"forbidden"),                       # a proxy that hides /models
    (404, b"<html>not found</html>"),
    (200, b"not json at all"),                 # a 200 that is not the OpenAI shape
    (200, b"[1, 2, 3]"),
    (200, b'{"data": [{"name": "no id"}]}'),
])
async def test_any_non_5xx_answer_ends_the_wait_at_once(monkeypatch, status, body):
    """The server is back the moment it answers anything but a 5xx: the identical re-send then reports
    its real status, instead of a 30-minute wait or a KeyError ending the turn."""
    endpoint = Answering(status, body, down_for=1)
    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(endpoint))
    announce = Announce()
    await asyncio.wait_for(ServerRecovery(None, None, None).recover(
        base_url=URL, api_key="none", wait_seconds=30, on_lost=announce), 2.0)
    assert endpoint.calls == 2  # the 503, then the answer


async def test_a_5xx_is_still_not_back(monkeypatch):
    endpoint = Answering(500, b"internal error")
    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(endpoint))
    with pytest.raises(TimeoutError, match="did not come back within"):
        await ServerRecovery(None, None, None).recover(
            base_url=URL, api_key="none", wait_seconds=0.1, on_lost=Announce())
    assert endpoint.calls >= 2


async def test_the_flight_asks_wait_ready_for_liveness(monkeypatch):
    recorder = WaitReadyRecorder()
    monkeypatch.setattr(server, "wait_ready", recorder)
    await ServerRecovery(None, None, None).recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert recorder.calls[0]["liveness"] is True


async def test_a_launch_time_wait_still_needs_a_200(monkeypatch):
    """`liveness` is the flight's alone: `start` and `/model` (and `activate`) keep waiting for a 200."""
    endpoint = Answering(401, b'{"error": "invalid api key"}')
    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(endpoint))
    with pytest.raises(TimeoutError, match="not ready after"):
        await server.wait_ready(URL, None, timeout_seconds=0.1, poll_seconds=0.02)
    assert endpoint.calls >= 2


async def test_poll_seconds_follows_the_constant_at_call_time(monkeypatch):
    recorder = WaitReadyRecorder()
    monkeypatch.setattr(server, "wait_ready", recorder)
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.07)
    await ServerRecovery(None, None, None).recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert recorder.calls[0]["poll_seconds"] == 0.07


# --- single flight --------------------------------------------------------------------------- #


async def test_three_callers_on_one_pair_share_one_flight(monkeypatch):
    recorder = WaitReadyRecorder(delay=0.1)
    monkeypatch.setattr(server, "wait_ready", recorder)
    announce = Announce()
    pair = ServerRecovery(None, None, None)
    await asyncio.gather(*(
        pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=announce) for _ in range(3)
    ))
    assert len(recorder.calls) == 1
    assert announce.count == 1


async def test_two_deaths_back_to_back_are_two_flights_and_two_announcements(models):
    state = {"down": True}
    endpoint = models(ModelsEndpoint(down=lambda: state["down"]))
    announce = Announce()
    pair = ServerRecovery(None, None, None)
    loop = asyncio.get_running_loop()
    for expected in (1, 2):
        state["down"] = True
        loop.call_later(0.1, state.update, {"down": False})
        await pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=announce)
        assert announce.count == expected
    assert endpoint.calls > 2


async def test_a_resolved_flight_is_cleared(monkeypatch):
    recorder = WaitReadyRecorder()
    monkeypatch.setattr(server, "wait_ready", recorder)
    pair = ServerRecovery(None, None, None)
    await pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert pair._flight.current is None
    await pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert len(recorder.calls) == 2


# --- the one relaunch ------------------------------------------------------------------------ #


async def test_a_server_that_answers_is_never_relaunched(models, tmp_path):
    models(ModelsEndpoint())
    write_live_pidfile(tmp_path)
    relaunch = RecordingRelaunch()
    await ServerRecovery(URL, tmp_path, relaunch).recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert relaunch.calls == 0


async def test_a_dead_process_is_relaunched_once_then_polled(models, tmp_path):
    models(_served_while_alive(tmp_path))
    relaunch = RecordingRelaunch(lambda: write_live_pidfile(tmp_path))
    await ServerRecovery(URL, tmp_path, relaunch, "llamacpp").recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert relaunch.calls == 1


async def test_three_callers_on_one_dead_pair_relaunch_exactly_once(models, tmp_path):
    models(_served_while_alive(tmp_path))
    relaunch = RecordingRelaunch(lambda: write_live_pidfile(tmp_path))
    pair = ServerRecovery(URL, tmp_path, relaunch)
    await asyncio.gather(*(
        pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce()) for _ in range(3)
    ))
    assert relaunch.calls == 1


async def test_a_second_death_after_the_one_relaunch_ends_with_the_log_tail(models, tmp_path):
    models(_served_while_alive(tmp_path))
    write_serve_log(tmp_path, "boot\nCUDA out of memory\n")
    relaunch = RecordingRelaunch(lambda: write_dead_pidfile(tmp_path))
    with pytest.raises(RuntimeError) as info:
        await ServerRecovery(URL, tmp_path, relaunch).recover(
            base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    text = str(info.value)
    assert text.startswith("the model server exited; it was restarted once and exited again. Log tail:\n")
    assert "CUDA out of memory" in text
    assert relaunch.calls == 1


async def test_a_dead_process_with_no_relaunch_fails_at_once_with_the_log_tail(models, tmp_path):
    models(ModelsEndpoint())
    write_serve_log(tmp_path, "it crashed\n")
    with pytest.raises(RuntimeError) as info:
        await asyncio.wait_for(ServerRecovery(URL, tmp_path, None).recover(
            base_url=URL, api_key="none", wait_seconds=60, on_lost=Announce()), 2.0)
    assert str(info.value) == "the model server exited. Log tail:\nit crashed"


async def test_a_relaunch_that_fails_is_named_as_a_failed_restart(models, tmp_path):
    models(_served_while_alive(tmp_path))
    relaunch = RecordingRelaunch(RuntimeError("llama.cpp exited during startup. Log tail:\nboom"))
    with pytest.raises(RuntimeError) as info:
        await ServerRecovery(URL, tmp_path, relaunch).recover(
            base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    text = str(info.value)
    assert text.startswith("the model server exited; it was restarted once and the restart failed: ")
    assert "boom" in text


async def test_a_relaunch_that_times_out_is_named_as_a_failed_restart(models, tmp_path):
    models(_served_while_alive(tmp_path))
    relaunch = RecordingRelaunch(TimeoutError("llama.cpp not ready after 30s."))
    with pytest.raises(RuntimeError) as info:
        await ServerRecovery(URL, tmp_path, relaunch).recover(
            base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert "the restart failed: llama.cpp not ready after 30s." in str(info.value)


@pytest.mark.parametrize("error", [
    FileNotFoundError(2, "No such file or directory", "/opt/llama/llama-server"),
    PermissionError(13, "Permission denied", "/opt/llama/llama-server"),
])
async def test_a_relaunch_that_cannot_start_the_binary_is_named_as_a_failed_restart(
        models, tmp_path, error):
    models(_served_while_alive(tmp_path))
    relaunch = RecordingRelaunch(error)
    with pytest.raises(RuntimeError) as info:
        await ServerRecovery(URL, tmp_path, relaunch).recover(
            base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    text = str(info.value)
    assert text.startswith("the model server exited; it was restarted once and the restart failed: ")
    assert "/opt/llama/llama-server" in text


async def test_a_refused_relaunch_reaches_the_caller_untouched(models, tmp_path):
    models(_served_while_alive(tmp_path))
    refusal = server.LaunchRefused("no llama-server — set server.binary")
    relaunch = RecordingRelaunch(refusal)
    with pytest.raises(server.LaunchRefused) as info:
        await ServerRecovery(URL, tmp_path, relaunch).recover(
            base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert info.value is refusal


async def test_a_dead_process_whose_address_already_answers_is_never_relaunched(models, tmp_path):
    """The launched process is gone, but the person restarted the server by hand at the same address: the
    flight looks at the address before relaunching, sees it answer, and ends the wait with no launch."""
    endpoint = models(ModelsEndpoint())  # answers, although no live process is recorded
    write_dead_pidfile(tmp_path)
    relaunch = RecordingRelaunch()
    await ServerRecovery(URL, tmp_path, relaunch).recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert relaunch.calls == 0
    assert endpoint.calls == 1  # the look at the address; the pidfile check made no request


async def test_a_look_at_the_address_carries_the_key(models, tmp_path):
    endpoint = models(ModelsEndpoint())
    await ServerRecovery(URL, tmp_path, RecordingRelaunch()).recover(
        base_url=URL, api_key=SecretStr(SENTINEL), wait_seconds=5, on_lost=Announce())
    assert endpoint.auth == ["Bearer " + SENTINEL]


async def test_a_relaunch_that_found_another_holder_already_serving_just_keeps_polling(models, tmp_path):
    models(_served_while_alive(tmp_path))
    relaunch = RecordingRelaunch(lambda: write_live_pidfile(tmp_path), returns=None)
    await ServerRecovery(URL, tmp_path, relaunch).recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    assert relaunch.calls == 1


# --- the bound ------------------------------------------------------------------------------- #


async def test_a_relaunch_that_outlasts_the_bound_is_cut_at_the_bound(models, tmp_path):
    """A relaunch waiting on a lock another process holds for its own long launch, or a liveness check
    that hangs, ends at the flight's bound with the bound's sentence; the relaunch itself is cancelled."""
    models(ModelsEndpoint(down=lambda: True))
    cancelled: list[int] = []

    async def stuck():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    relaunch = RecordingRelaunch(stuck)
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError) as info:
        await asyncio.wait_for(ServerRecovery(URL, tmp_path, relaunch).recover(
            base_url=URL, api_key="none", wait_seconds=0.3, on_lost=Announce()), 2.0)
    assert asyncio.get_running_loop().time() - started < 1.0
    assert str(info.value).startswith("the model server did not come back within 0s")
    assert relaunch.calls == 1 and cancelled == [1]


async def test_a_relaunch_under_a_lock_held_elsewhere_ends_at_the_bound_and_lets_go(models, tmp_path):
    models(ModelsEndpoint(down=lambda: True))
    lock = server.lock_path(tmp_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    foreign = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(foreign, fcntl.LOCK_EX)
    try:
        async def under_the_lock():
            async with lifecycle.launch_lock(tmp_path):
                pytest.fail("the lock is held elsewhere for the whole test")

        with pytest.raises(TimeoutError, match="did not come back within"):
            await asyncio.wait_for(ServerRecovery(URL, tmp_path, RecordingRelaunch(under_the_lock)).recover(
                base_url=URL, api_key="none", wait_seconds=0.3, on_lost=Announce()), 2.0)
    finally:
        os.close(foreign)
    assert lifecycle._HELD == {}


async def test_the_bound_with_no_config_dir_has_no_log_tail(models):
    endpoint = models(ModelsEndpoint(down=lambda: True))
    with pytest.raises(TimeoutError) as info:
        await ServerRecovery(None, None, None).recover(
            base_url=URL, api_key="none", wait_seconds=0.1, on_lost=Announce())
    assert str(info.value) == "the model server did not come back within 0s"
    assert endpoint.calls >= 2


async def test_the_bound_with_a_config_dir_carries_the_log_tail(models, tmp_path):
    models(ModelsEndpoint(down=lambda: True))
    write_live_pidfile(tmp_path)
    write_serve_log(tmp_path, "loading weights 40%\n")
    with pytest.raises(TimeoutError) as info:
        await ServerRecovery(URL, tmp_path, None).recover(
            base_url=URL, api_key="none", wait_seconds=0.1, on_lost=Announce())
    assert str(info.value) == "the model server did not come back within 0s. Log tail:\nloading weights 40%"


async def test_a_config_dir_with_no_log_still_says_the_whole_sentence(models, tmp_path):
    models(ModelsEndpoint(down=lambda: True))
    write_live_pidfile(tmp_path)
    with pytest.raises(TimeoutError) as info:
        await ServerRecovery(URL, tmp_path, None).recover(
            base_url=URL, api_key="none", wait_seconds=0.1, on_lost=Announce())
    assert str(info.value).startswith("the model server did not come back within 0s. Log tail:\n")
    assert str(info.value).endswith("(no log)")


@pytest.mark.parametrize(("seconds", "rendered"), [(30.0, "30s"), (1800.0, "1800s"), (0.4, "0s"), (0.6, "1s")])
async def test_the_bound_is_rendered_as_whole_seconds(monkeypatch, seconds, rendered):
    monkeypatch.setattr(server, "wait_ready", WaitReadyRecorder([TimeoutError("vLLM not ready")]))
    with pytest.raises(TimeoutError) as info:
        await ServerRecovery(None, None, None).recover(
            base_url=URL, api_key="none", wait_seconds=seconds, on_lost=Announce())
    assert str(info.value) == f"the model server did not come back within {rendered}"


async def test_a_bound_already_spent_after_the_relaunch_fails_without_polling_again(monkeypatch, tmp_path):
    recorder = WaitReadyRecorder([RuntimeError("exited")], delay=0.15)
    monkeypatch.setattr(server, "wait_ready", recorder)
    relaunch = RecordingRelaunch(returns=None)
    with pytest.raises(TimeoutError) as info:
        await ServerRecovery(URL, None, relaunch).recover(
            base_url=URL, api_key="none", wait_seconds=0.1, on_lost=Announce())
    assert str(info.value) == "the model server did not come back within 0s"
    assert len(recorder.calls) == 1
    assert relaunch.calls == 1


async def test_a_pair_with_no_config_dir_never_reads_a_death_only_the_bound(models):
    """LM Studio and a hand-run server: nothing records a pid, so a server that never answers is
    polled to the bound and fails with the bound text, never the dead-process sentence."""
    models(ModelsEndpoint(down=lambda: True))
    relaunch = RecordingRelaunch()
    with pytest.raises(TimeoutError, match="did not come back within"):
        await ServerRecovery(URL, None, relaunch, "lmstudio").recover(
            base_url=URL, api_key="none", wait_seconds=0.15, on_lost=Announce())
    assert relaunch.calls == 0


# --- failures reach everyone, and are never left unretrieved --------------------------------- #


async def test_every_awaiter_sees_the_same_failure_and_nothing_is_left_unretrieved(models, tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="asyncio")
    models(ModelsEndpoint())
    pair = ServerRecovery(URL, tmp_path, None)
    results = await asyncio.gather(
        *(pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce()) for _ in range(2)),
        return_exceptions=True,
    )
    assert all(isinstance(r, RuntimeError) for r in results)
    assert str(results[0]) == str(results[1]) and str(results[0]).startswith("the model server exited.")
    gc.collect()
    await asyncio.sleep(0)
    assert "never retrieved" not in caplog.text


# --- cancellation (Esc) ---------------------------------------------------------------------- #


async def test_cancelling_one_of_two_awaiters_leaves_the_flight_for_the_other(monkeypatch):
    recorder = WaitReadyRecorder(delay=0.2)
    monkeypatch.setattr(server, "wait_ready", recorder)
    pair = ServerRecovery(None, None, None)
    first = asyncio.create_task(pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce()))
    second = asyncio.create_task(pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce()))
    await asyncio.sleep(0.05)
    flight = pair._flight.current.task
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, 1.0)
    assert not flight.cancelled()
    assert len(recorder.calls) == 1


async def test_cancelling_the_only_awaiter_mid_relaunch_cancels_the_flight_and_frees_the_lock(
    models, tmp_path
):
    models(_served_while_alive(tmp_path))
    inside = asyncio.Event()
    attempts = {"n": 0}

    async def relaunch_under_the_lock():
        attempts["n"] += 1
        if attempts["n"] > 1:
            write_live_pidfile(tmp_path)
            return
        async with lifecycle.launch_lock(tmp_path):
            inside.set()
            await asyncio.Event().wait()  # a launch that never finishes

    relaunch = RecordingRelaunch(relaunch_under_the_lock)
    pair = ServerRecovery(URL, tmp_path, relaunch)
    waiter = asyncio.create_task(pair.recover(base_url=URL, api_key="none", wait_seconds=30, on_lost=Announce()))
    await asyncio.wait_for(inside.wait(), 2.0)
    assert not _lockable(tmp_path)
    flight = pair._flight.current.task
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    # a recover this very tick starts a fresh flight and is never handed the dying one
    again = asyncio.create_task(pair.recover(base_url=URL, api_key="none", wait_seconds=30, on_lost=Announce()))
    await asyncio.wait_for(asyncio.wait([flight]), 2.0)
    assert flight.cancelled()
    await asyncio.wait_for(again, 5.0)  # returns normally, no CancelledError
    assert _lockable(tmp_path)


async def test_a_flight_the_last_awaiter_left_is_kept_alive_until_it_unwinds(monkeypatch):
    recorder = WaitReadyRecorder(delay=0.2)
    monkeypatch.setattr(server, "wait_ready", recorder)
    pair = ServerRecovery(None, None, None)
    waiter = asyncio.create_task(pair.recover(base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce()))
    await asyncio.sleep(0.05)
    flight = pair._flight.current.task
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert flight in lifecycle._UNWINDING
    await asyncio.wait([flight])
    await asyncio.sleep(0)
    assert flight not in lifecycle._UNWINDING


# --- the remaining bound ---------------------------------------------------------------------- #


def test_the_budget_outside_a_flight_is_activates_own_default():
    assert lifecycle.relaunch_budget() == 1800.0


async def test_the_budget_inside_a_flight_is_what_is_left_and_stays_in_the_flight(models, tmp_path):
    models(_served_while_alive(tmp_path))
    relaunch = RecordingRelaunch(lambda: write_live_pidfile(tmp_path))
    await ServerRecovery(URL, tmp_path, relaunch).recover(
        base_url=URL, api_key="none", wait_seconds=5, on_lost=Announce())
    (budget,) = relaunch.budgets
    assert 1.0 < budget <= 5.0
    assert lifecycle.relaunch_budget() == 1800.0  # the caller of recover never saw the deadline


async def test_a_relaunch_that_comes_up_on_the_bounds_edge_is_a_success(monkeypatch, tmp_path):
    """activate returned a live endpoint (its own wait saw /models answer), but the flight's bound passed
    while it did: the server is serving, so the wait ends well instead of "did not come back"."""
    recorder = WaitReadyRecorder([RuntimeError("exited")], delay=0.15)
    monkeypatch.setattr(server, "wait_ready", recorder)
    relaunch = RecordingRelaunch()  # returns a LiveEndpoint
    await ServerRecovery(URL, None, relaunch).recover(
        base_url=URL, api_key="none", wait_seconds=0.1, on_lost=Announce())
    assert relaunch.calls == 1
    assert len(recorder.calls) == 1  # nothing polled after the bound


async def test_the_budget_never_falls_below_one_second(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "wait_ready", WaitReadyRecorder([RuntimeError("exited")], delay=0.1))
    relaunch = RecordingRelaunch(returns=None)
    with pytest.raises(TimeoutError):  # the bound (0.05 s) was spent by the time the server was seen dead
        await ServerRecovery(URL, tmp_path, relaunch).recover(
            base_url=URL, api_key="none", wait_seconds=0.05, on_lost=Announce())
    assert relaunch.budgets == [1.0]


# --- the helper's own chat fake -------------------------------------------------------------- #


async def test_the_dropping_chat_fake_drops_once_then_streams_a_whole_answer():
    chat = DroppingChat([text_answer("hello")])
    async with httpx.AsyncClient(transport=httpx.MockTransport(chat)) as client:
        with pytest.raises(httpx.RemoteProtocolError):
            async with client.stream("POST", URL + "/chat/completions", json={"n": 1}) as response:
                async for _ in response.aiter_bytes():
                    pass
        async with client.stream("POST", URL + "/chat/completions", json={"n": 1}) as response:
            body = b"".join([part async for part in response.aiter_bytes()])
    assert body.endswith(b"data: [DONE]\n\n") and b"hello" in body
    assert len(chat.bodies) == 2 and chat.bodies[0] == chat.bodies[1]
