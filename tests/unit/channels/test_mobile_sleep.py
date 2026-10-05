"""Sleep and wake, the runner's side (0.16.2): the sleep request lands in the REPL's own
`read_input`, a message beats it, the idle watch asks only after a whole quiet stretch, bring-up
turns the builder's answer into the `asleep` stage and drops what the session held, the server's
latch opens again so the next message wakes the thread, a connect never does, and Ctrl-C puts a
live session to sleep instead of killing it.

The thread's trip to disk and back through the real session builder is
tests/integration/test_mobile_sleep_e2e.py.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from localharness.channels.mobile import server as server_mod
from localharness.channels.mobile.channel import MobileChannel
from localharness.channels.mobile.protocol import ASLEEP_STAGE
from localharness.cli import mobile_cmd
from localharness.cli.session_resume import SLEEP_ACTION, Restart, Resume
from localharness.core.bus import EventBus
from tests.unit.channels.test_mobile_server import BEARER, JSON, TOKEN

pytestmark = pytest.mark.asyncio


async def _channel(tmp_path) -> MobileChannel:
    channel = MobileChannel(bus=EventBus(persist_path=tmp_path / "bus-events.jsonl"), config={})
    await channel.start()
    return channel


def _bind(channel: MobileChannel, tmp_path, sid: str = "s1", **kw) -> None:
    channel.bind_runtime(session_id=sid, agent_id="orchestrator", session_dir=tmp_path / "sessions",
                         **kw)


def _slept(sid: str = "s1") -> Restart:
    return Restart(SLEEP_ACTION, Resume(
        action=SLEEP_ACTION, agent_name="orchestrator", conversation=(), prior_context="",
        eviction_store=None, queued=(), gate_mode="auto", previous_sitting_id=sid))


# ---------------------------------------------------------------- the request lands in read_input

async def test_a_sleep_request_ends_read_input_the_way_quit_does(tmp_path):
    channel = await _channel(tmp_path)
    reader = asyncio.ensure_future(channel.read_input())
    await asyncio.sleep(0)
    assert not reader.done()
    channel.request_sleep()
    with pytest.raises(EOFError):
        await asyncio.wait_for(reader, timeout=1)
    assert channel.sleeping, "latched: the builder reads it after the REPL ends"


async def test_the_sleep_latch_survives_a_line_that_arrives_during_the_teardown(tmp_path):
    channel = await _channel(tmp_path)
    reader = asyncio.ensure_future(channel.read_input())
    await asyncio.sleep(0)
    channel.request_sleep()
    with pytest.raises(EOFError):
        await asyncio.wait_for(reader, timeout=1)
    channel.submit("arrived while the session was falling asleep")
    assert channel.sleeping, "the builder must still write the thread"
    assert channel.queued_input == 1, "and the runner wakes it for this line"


async def test_a_message_already_waiting_beats_the_sleep_request(tmp_path):
    channel = await _channel(tmp_path)
    channel.request_sleep()
    channel.submit("hi")
    assert not channel.sleep_requested, "a person spoke: the request is withdrawn"
    assert await asyncio.wait_for(channel.read_input(), timeout=1) == "hi"


async def test_a_message_that_arrives_while_both_are_waiting_wins_and_the_eof_is_not_stale(tmp_path):
    channel = await _channel(tmp_path)
    reader = asyncio.ensure_future(channel.read_input())
    await asyncio.sleep(0)
    channel.submit("late")
    channel.request_sleep()  # both ready in the same tick
    assert await asyncio.wait_for(reader, timeout=1) == "late"
    assert not channel.sleep_requested
    again = asyncio.ensure_future(channel.read_input())
    await asyncio.sleep(0.05)
    assert not again.done(), "the next read is a plain wait, not a leftover EOF"
    again.cancel()


# ---------------------------------------------------------------- what idle means

async def test_idle_means_live_quiet_and_alone(tmp_path):
    channel = await _channel(tmp_path)
    assert not channel.idle(), "no session: nothing to put to sleep"
    _bind(channel, tmp_path, gate=SimpleNamespace(mode="auto", pending={}))
    assert channel.idle()
    client = channel.attach_client()
    assert not channel.idle(), "a phone is attached"
    channel.detach_client(client)
    channel.submit("queued")
    assert not channel.idle(), "a line is waiting"
    assert await channel.read_input() == "queued"
    channel._turn_running = True
    assert not channel.idle(), "a turn is running"
    channel._turn_running = False
    channel._gate.pending["7"] = object()
    assert not channel.idle(), "a parked call is a future only this session's gate can answer"
    channel._gate.pending.clear()
    channel._open_asks["r1"] = object()
    assert not channel.idle(), "a question is open"
    channel._open_asks.clear()
    assert channel.idle()
    assert channel.queued_input == 0


async def test_the_idle_watch_asks_for_sleep_only_after_a_whole_quiet_stretch(tmp_path):
    channel = await _channel(tmp_path)
    _bind(channel, tmp_path)
    watch = asyncio.ensure_future(mobile_cmd._sleep_watch(channel, after_s=0.3, poll_s=0.05))
    try:
        await asyncio.sleep(0.15)
        assert not channel.sleep_requested
        client = channel.attach_client()  # a phone shows up: the stretch starts over
        await asyncio.sleep(0.3)
        assert not channel.sleep_requested, "attached the whole time"
        channel.detach_client(client)
        await asyncio.sleep(0.15)
        assert not channel.sleep_requested, "quiet again, but not for long enough yet"
        await asyncio.sleep(0.5)
        assert channel.sleep_requested
    finally:
        watch.cancel()


# ---------------------------------------------------------------- bring-up's sleep branch

async def test_bring_up_turns_the_builders_sleep_into_the_asleep_stage_and_drops_the_session(
        tmp_path, monkeypatch):
    from localharness.cli import start_cmd

    channel = await _channel(tmp_path)
    seen: dict = {}

    async def builder(*a, **kw):
        seen.update(kw)
        _bind(channel, tmp_path, tool_registry=object(), llm=SimpleNamespace(config=None))
        channel._model_reachable = False  # the last health probe found the model server down
        return _slept()

    monkeypatch.setattr(start_cmd, "_start_async", builder)
    # what the REPL leaves on the channel: its own bound methods, which the next REPL must not find
    channel._pending_resolver = channel._nudge_resolver = channel._cancel_resolver = object()
    phone = channel.attach_client()
    await mobile_cmd._bring_up(channel, config_dir=None, verbose=False, agent=None, fresh_thread=True)

    assert seen["fresh_thread"] is True and seen["mobile_channel"] is channel
    assert channel.asleep and channel.session_id is None and channel.model_state() == "asleep"
    stage = channel.bringup
    assert (stage.stage, stage.detail, stage.failed) == (ASLEEP_STAGE, mobile_cmd.ASLEEP_DETAIL, False)
    assert stage.session_id == "s1", "published while still bound: the client learns which chat slept"
    assert channel._tool_registry is None, "it would keep the memory tools, the engine and the model"
    assert (channel._pending_resolver, channel._nudge_resolver, channel._cancel_resolver) == (None,) * 3, (
        "the next REPL must install its own, or the phone's stop and nudge reach a REPL that is gone")
    assert channel._model_reachable is None, "a stale 'unreachable' must not outrank 'asleep'"
    frames = [f for f in list(phone.queue._queue) if f[0] == "BringUpStage"]
    assert '"stage":"asleep"' in frames[-1][2].replace(" ", "")


async def test_bring_up_still_says_ended_when_the_session_simply_ended(tmp_path, monkeypatch):
    from localharness.cli import start_cmd

    channel = await _channel(tmp_path)

    async def builder(*a, **kw):
        _bind(channel, tmp_path)
        return None

    monkeypatch.setattr(start_cmd, "_start_async", builder)
    await mobile_cmd._bring_up(channel, config_dir=None, verbose=False, agent=None)
    assert channel.bringup.stage == "ended" and not channel.asleep


# ---------------------------------------------------------------- the server: a message wakes, a connect does not

async def test_after_sleep_the_next_message_wakes_the_session_and_a_connect_does_not(tmp_path):
    channel = await _channel(tmp_path)
    started: list = []
    server = server_mod.MobileServer(channel, token=TOKEN,
                                     on_first_message=lambda: started.append("up"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                 base_url="http://web.test") as client:
        first = await client.post("/api/sessions/current/message", json={"text": "hi"}, headers=JSON)
        assert first.json()["bringing_up"] is True and started == ["up"]
        _bind(channel, tmp_path)
        assert await channel.read_input() == "hi"  # the REPL that came up read it
        channel.set_bringup(ASLEEP_STAGE, detail=mobile_cmd.ASLEEP_DETAIL)
        channel.reset_session()
        server.session_over()  # what the runner does once the session task is over

        health = (await client.get("/api/health", headers=BEARER)).json()
        assert (health["model_state"], health["session_live"]) == ("asleep", False)

        async def answering() -> bool:
            return True
        channel.probe_model = answering  # a warm box: a connect would otherwise pre-warm
        await server._maybe_prewarm()
        assert started == ["up"], "a connect is a pocket; a sleeping thread wakes on a message"

        second = await client.post("/api/sessions/current/message", json={"text": "again"}, headers=JSON)
        assert second.json()["bringing_up"] is True and started == ["up", "up"]
        await channel.start()  # the woken REPL's first act
        assert await channel.read_input() == "again", "the line waited for it"


async def _serve_with(monkeypatch, tmp_path, *, scenario, bring_up, sleep_after: int) -> list:
    """The real `_serve` with uvicorn's listen replaced by `scenario(server)` and bring-up by
    `bring_up`; returns the `after_s` each idle watch was started with."""
    import uvicorn

    built: list = []
    watched: list = []

    class _Spy(server_mod.MobileServer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)

    async def fake_watch(channel, *, after_s, poll_s=None):
        watched.append(after_s)
        await asyncio.sleep(3600)

    async def listen(self, *a, **kw):
        await scenario(built[-1])

    monkeypatch.setattr(server_mod, "MobileServer", _Spy)
    monkeypatch.setattr(uvicorn.Server, "serve", listen)
    monkeypatch.setattr(mobile_cmd, "_bring_up", bring_up)
    monkeypatch.setattr(mobile_cmd, "_sleep_watch", fake_watch)
    monkeypatch.setattr(mobile_cmd, "_open_tty", lambda: None, raising=False)
    await mobile_cmd._serve(config_dir=str(tmp_path), host="127.0.0.1", port=0, token=TOKEN,
                            ui_dir=None, replay=None, fixtures=None, speed=1.0, verbose=False,
                            agent=None, sleep_after=sleep_after)
    return watched


async def _until(pred, what: str) -> None:
    """The done-callback collects garbage, which takes what it takes: wait on state, not time."""
    for _ in range(500):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never happened: {what}")


async def test_the_runner_reopens_the_latch_and_wakes_a_message_that_arrived_mid_sleep(
        tmp_path, monkeypatch):
    """The real `_serve` with bring-up faked as a session that sleeps at once: the done-callback
    opens the server's latch, a line that arrived while the session was falling asleep wakes it
    immediately, and the new-chat verb builds with `fresh_thread`."""
    calls: list = []

    async def bring_up(channel, *, fresh_thread=False, **kw):
        calls.append(fresh_thread)
        await channel.start()  # the REPL's first act
        channel.bind_runtime(session_id=f"s{len(calls)}", agent_id="orchestrator",
                             session_dir=tmp_path / "sessions")
        while channel.queued_input:  # the REPL would read these
            await channel.read_input()
        if len(calls) == 1:
            channel.submit("sent while the session was falling asleep")
        channel.set_bringup(ASLEEP_STAGE, detail=mobile_cmd.ASLEEP_DETAIL)
        channel.reset_session()

    out: dict = {}

    async def scenario(server):
        channel = server.channel
        settled = lambda n: len(calls) == n and channel.asleep and not server._bringup_started  # noqa: E731
        assert server._ensure_session() is True
        await _until(lambda: settled(2), "the woken session slept quiet and the latch opened")
        out["after_first_sleep"] = (list(calls), server._bringup_started, channel.asleep)
        assert server._ensure_session() is True  # the latch is open: a message wakes it
        await _until(lambda: settled(3), "the third session slept")
        await server.on_new_session()
        await _until(lambda: settled(4), "the fresh chat was built and slept")
        out["calls"] = list(calls)

    watched = await _serve_with(monkeypatch, tmp_path, scenario=scenario, bring_up=bring_up,
                                sleep_after=7)
    assert out["after_first_sleep"] == ([False, False], False, True), (
        "the first session slept with a line waiting and was woken at once; the second slept quiet")
    assert out["calls"] == [False, False, False, True], "the new-chat verb discards the thread"
    assert watched == [7 * 60.0], "the idle watch runs with the flag's minutes"


async def test_sleep_after_zero_starts_no_idle_watch(tmp_path, monkeypatch):
    async def never(*a, **kw):
        raise AssertionError("no session is built here")

    async def nothing(server):
        return None

    assert await _serve_with(monkeypatch, tmp_path, scenario=nothing, bring_up=never, sleep_after=0) == []


async def test_the_flag_reaches_serve(tmp_path, monkeypatch):
    from tests.unit.channels.test_mobile_artifacts import _invoke_web

    result, seen = await asyncio.to_thread(_invoke_web, tmp_path, monkeypatch, "--sleep-after", "5")
    assert result.exit_code == 0, result.output
    assert seen["sleep_after"] == 5
    result, seen = await asyncio.to_thread(_invoke_web, tmp_path, monkeypatch)
    assert result.exit_code == 0 and seen["sleep_after"] == mobile_cmd.SLEEP_AFTER_DEFAULT_MIN == 30
    result, _ = await asyncio.to_thread(_invoke_web, tmp_path, monkeypatch, "--sleep-after", "-1")
    assert result.exit_code != 0


# ---------------------------------------------------------------- Ctrl-C keeps the thread

async def test_shutdown_puts_a_live_idle_session_to_sleep_instead_of_cancelling_it(tmp_path):
    channel = await _channel(tmp_path)
    _bind(channel, tmp_path)

    async def session() -> str:  # the session task: it ends when its REPL reads EOF
        try:
            await channel.read_input()
        except EOFError:
            return "slept"
        return "read a line?"

    task = asyncio.ensure_future(session())
    await asyncio.sleep(0)
    await mobile_cmd._sleep_or_cancel(channel, task)
    assert task.result() == "slept"


async def test_shutdown_sleeps_even_with_a_line_queued_and_waits_for_the_teardown(tmp_path):
    """A line typed ahead must not turn the stop into another turn (it did: the line won the race
    and cleared the request), and the teardown is waited for however long it takes — memory's own
    shutdown can hold it for fifteen seconds, and cutting it short left its store open."""
    channel = await _channel(tmp_path)
    _bind(channel, tmp_path)
    channel.submit("typed ahead")

    async def session() -> str:
        try:
            got = await channel.read_input()
        except EOFError:
            await asyncio.sleep(0.3)  # the ordered teardown
            return "slept"
        return f"ran a turn for {got!r}"

    task = asyncio.ensure_future(session())
    await asyncio.sleep(0)
    await mobile_cmd._sleep_or_cancel(channel, task)
    assert task.result() == "slept"
    assert channel.queued_input == 1, "the line waits for the wake, it is not lost to the stop"


async def test_shutdown_cancels_a_build_that_has_no_thread_yet(tmp_path):
    channel = await _channel(tmp_path)
    task = asyncio.ensure_future(asyncio.sleep(3600))
    await mobile_cmd._sleep_or_cancel(channel, task)
    assert task.cancelled() and not channel.sleep_requested


# ---------------------------------------------------------------- the stop signal, through the real uvicorn

async def _serve_real_uvicorn(monkeypatch, tmp_path, *, bring_up, after_started) -> None:
    """The real `_serve` on a real ephemeral port: uvicorn captures the stop signal for its graceful
    shutdown and then re-raises it to the handler installed before it — which must be the
    server's own, or the finally that puts the session to sleep is cancelled on its first await
    (seen on the real surface, 2026-10-04). `after_started(server)` runs once uvicorn is up."""
    import uvicorn

    built: list = []
    runners: list = []
    real_startup = uvicorn.Server.startup

    class _Spy(server_mod.MobileServer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)

    async def startup(self, *a, **kw):
        runners.append(self)
        return await real_startup(self, *a, **kw)

    async def driver():
        for _ in range(500):
            if runners and runners[0].started:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("uvicorn never started")
        await after_started(built[-1])

    monkeypatch.setattr(server_mod, "MobileServer", _Spy)
    monkeypatch.setattr(uvicorn.Server, "startup", startup)
    monkeypatch.setattr(mobile_cmd, "_bring_up", bring_up)
    monkeypatch.setattr(mobile_cmd, "_open_tty", lambda: None, raising=False)
    drive = asyncio.ensure_future(driver())
    try:
        await asyncio.wait_for(mobile_cmd._serve(
            config_dir=str(tmp_path), host="127.0.0.1", port=0, token=TOKEN, ui_dir=None,
            replay=None, fixtures=None, speed=1.0, verbose=False, agent=None, sleep_after=0),
            timeout=30)
    finally:
        drive.cancel()


async def test_a_stop_signal_puts_the_live_session_to_sleep_through_the_real_uvicorn(
        tmp_path, monkeypatch):
    import os
    import signal as _signal

    ended: list = []

    async def bring_up(channel, **kw):
        await channel.start()
        channel.bind_runtime(session_id="s1", agent_id="orchestrator", session_dir=tmp_path / "sessions")
        try:
            await channel.read_input()  # the REPL, waiting for the next line
        except EOFError:
            ended.append("slept" if channel.sleep_requested else "eof")

    async def stop(server):
        assert server._ensure_session() is True
        for _ in range(500):
            if server.channel.session_id is not None:
                break
            await asyncio.sleep(0.01)
        os.kill(os.getpid(), _signal.SIGINT)

    before = _signal.getsignal(_signal.SIGINT)
    await _serve_real_uvicorn(monkeypatch, tmp_path, bring_up=bring_up, after_started=stop)
    assert ended == ["slept"], "the session must end through the sleep request, not a cancel"
    assert _signal.getsignal(_signal.SIGINT) == before, "the server's handler is gone with it"


async def test_a_second_stop_signal_gives_up_on_the_sleep(tmp_path, monkeypatch, caplog):
    import os
    import signal as _signal

    ended: list = []

    async def bring_up(channel, **kw):
        await channel.start()
        channel.bind_runtime(session_id="s1", agent_id="orchestrator", session_dir=tmp_path / "sessions")
        try:
            await asyncio.sleep(3600)  # a session that will not fall asleep
        except asyncio.CancelledError:
            ended.append("cancelled")
            raise

    async def stop_twice(server):
        assert server._ensure_session() is True
        for _ in range(500):
            if server.channel.session_id is not None:
                break
            await asyncio.sleep(0.01)
        os.kill(os.getpid(), _signal.SIGINT)
        await asyncio.sleep(0.5)  # into the grace wait
        os.kill(os.getpid(), _signal.SIGINT)

    with caplog.at_level("WARNING"):
        await _serve_real_uvicorn(monkeypatch, tmp_path, bring_up=bring_up, after_started=stop_twice)
    assert ended == ["cancelled"] and "second stop signal" in caplog.text


# ---------------------------------------------------------------- a failed wake tries again

async def test_a_failed_bring_up_reopens_the_latch_so_the_next_message_retries(tmp_path, monkeypatch):
    """A wake that fails before a session exists puts its file back (the builder's job) — and the
    next message must be able to try again, which the server's latch used to prevent."""
    calls: list = []

    async def bring_up(channel, **kw):
        calls.append(1)
        channel.set_bringup("failed", detail="the model server is down", failed=True)

    out: dict = {}

    async def scenario(server):
        assert server._ensure_session() is True
        await _until(lambda: len(calls) == 1 and not server._bringup_started, "the latch reopened")
        assert server._ensure_session() is True, "the next message tries again"
        await _until(lambda: len(calls) == 2, "the retry ran")
        out["calls"] = len(calls)

    await _serve_with(monkeypatch, tmp_path, scenario=scenario, bring_up=bring_up, sleep_after=0)
    assert out["calls"] == 2


# ---------------------------------------------------------------- the screens and the drawer while asleep

async def _asleep(tmp_path, sid: str = "s1") -> MobileChannel:
    channel = await _channel(tmp_path)
    _bind(channel, tmp_path, sid)
    channel.set_bringup(ASLEEP_STAGE, detail=mobile_cmd.ASLEEP_DETAIL)
    channel.reset_session()
    assert channel.asleep
    return channel


async def test_the_memory_screen_says_asleep_not_off(tmp_path):
    channel = await _asleep(tmp_path)
    server = server_mod.MobileServer(channel, token=TOKEN)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                 base_url="http://web.test") as client:
        got = await client.get("/api/memory", headers=BEARER)
        assert got.status_code == 409 and "asleep" in got.json()["error"]
        got = await client.get("/api/memory/fact?name=x", headers=BEARER)
        assert got.status_code == 409
        channel.clear_sleep()
        got = await client.get("/api/memory", headers=BEARER)
        assert got.status_code == 404 and got.json() == server_mod.MEMORY_OFF


async def test_deleting_the_sleeping_chat_from_the_drawer_discards_the_thread(tmp_path):
    import os

    from localharness.cli.session_resume import asleep_path, write_asleep

    channel = await _asleep(tmp_path, "s1")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "s1.jsonl").write_text('{"seq": 1}\n')
    (sessions / "s0.jsonl").write_text('{"seq": 1}\n')
    write_asleep(asleep_path(sessions), _slept("s1").resume, workspace=os.getcwd())
    server = server_mod.MobileServer(channel, token=TOKEN)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                 base_url="http://web.test") as client:
        other = await client.post("/api/sessions/s0/delete", headers=JSON, json={})
        assert other.status_code == 200 and asleep_path(sessions).exists(), "another chat: the thread stays"
        gone = await client.post("/api/sessions/s1/delete", headers=JSON, json={})
        assert gone.status_code == 200
        assert not asleep_path(sessions).exists(), "the sleeping chat's thread goes with its log"
        assert not channel.asleep
        health = (await client.get("/api/health", headers=BEARER)).json()
        assert health["model_state"] == "cold"
