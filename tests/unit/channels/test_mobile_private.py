"""A tap starts a private chat in its own run folder, it sleeps and wakes there, End removes it, and a message lands in the chat on screen."""
import asyncio
import logging
import os
import signal
import time
from types import SimpleNamespace

import pytest

from localharness.channels.mobile.channel import MobileChannel
from localharness.channels.mobile.protocol import ASLEEP_STAGE
from localharness.cli import start_cmd
from localharness.core import run_folder
from localharness.core.bus import EventBus
from tests.unit.channels.test_mobile_server import JSON, _fixed, _read_frames, _stack

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-relative run folders")


@pytest.fixture
def cfg(tmp_path):
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "config.yaml").write_text("{}")
    return d


async def _until(cond, timeout=5):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.02)


def _owner(monkeypatch, cfg, outcomes, script):
    """Drive `_serve` with a fake server and a fake session builder: `script(callbacks, runs, runner)` stands for the phone."""
    import uvicorn

    from localharness.channels.mobile import server as server_mod
    from localharness.cli import mobile_cmd

    seen = {"cb": {}, "runs": [], "cancelled": 0}

    class FakeServer:
        app = None

        def __init__(self, channel, **kw):
            seen["cb"].update(kw, channel=channel)

        def session_over(self):
            seen["cb"]["over"] = seen["cb"].get("over", 0) + 1

    async def fake_start(*args, run=None, mobile_channel=None, **kw):
        seen["runs"].append(run)
        outcome = outcomes.pop(0)
        if outcome == "raise":
            raise RuntimeError("the model server is down")
        private_dir = run.state / "sessions" if run is not None and run.private else None
        mobile_channel.bind_runtime(session_id=f"c{len(seen['runs'])}", agent_id="a",
                                    session_dir=cfg / "agents" / "a" / "sessions", private_dir=private_dir)
        if outcome == "block":
            try:
                while not mobile_channel.sleep_requested:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                seen["cancelled"] += 1
                raise
        return SimpleNamespace(resume=SimpleNamespace(slept=outcome != "end"))

    async def fake_serve(self):
        await script(seen["cb"], seen["runs"], self)

    monkeypatch.setattr(server_mod, "MobileServer", FakeServer)
    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    monkeypatch.setattr(start_cmd, "_start_async", fake_start)
    asyncio.run(mobile_cmd._serve(
        config_dir=str(cfg), host="127.0.0.1", port=1, token="t" * 8, ui_dir=None, replay=None,
        fixtures=None, speed=1.0, verbose=False, agent=None, sleep_after=0))
    return seen


def test_a_tap_on_a_cold_box_begins_a_private_chat(cfg, monkeypatch):
    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=True)
        await _until(lambda: ch.session_id)
        assert runs[0].private and ch.run is runs[0] and ch.private is True
        assert ch.bringup.private and (await ch.hello(None, None)).private

    _owner(monkeypatch, cfg, ["block"], script)


def test_a_tap_on_a_warm_box_replaces_the_normal_chat_with_a_private_one(cfg, monkeypatch):
    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=False)
        await _until(lambda: ch.session_id == "c1")
        assert not runs[0].private and ch.private is False and not (await ch.hello(None, None)).private
        assert await cb["on_new_session"](private=True) == ""
        await _until(lambda: ch.session_id == "c2")
        assert runs[1].private and ch.run is runs[1] and ch.private is True
        assert not runs[0].path.exists()

    _owner(monkeypatch, cfg, ["block", "block"], script)


def test_a_private_chat_sleeps_in_its_run_folder_and_wakes_in_it(cfg, monkeypatch):
    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=True)
        await _until(lambda: ch.asleep)
        hello = await ch.hello(None, None)
        assert runs[0].path.exists() and ch.run is runs[0] and ch.private is True
        assert (hello.asleep_session_id, hello.session_live, hello.private) == ("c1", False, True)
        assert ch.model_state() == "asleep" and ch.bringup.stage == ASLEEP_STAGE
        cb["on_first_message"]()
        await _until(lambda: ch.session_id == "c2")
        assert runs[1] is runs[0] and ch.private is True

    _owner(monkeypatch, cfg, ["asleep", "block"], script)


@pytest.mark.parametrize("private", [True, False])
def test_a_failed_wake_keeps_a_private_chat_until_end_and_ends_a_normal_one(cfg, monkeypatch, private):
    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=private)
        await _until(lambda: ch.asleep)
        run = runs[0]
        thread = run.state / "sessions" / "thread.jsonl"
        if private:
            thread.parent.mkdir(parents=True, exist_ok=True)
            thread.write_text("{}")
        for over in (2, 3):
            cb["on_first_message"]()
            await _until(lambda: cb["over"] == over)
            assert ch.bringup.failed and ch.session_id is None
            assert (ch.run is run, ch.private) == ((True, True) if private else (False, None))
        if private:
            assert runs[1] is runs[2] is run and thread.exists()
        await cb["on_end"]()
        await _until(lambda: not run.path.exists())
        assert ch.run is None and ch.private is None

    _owner(monkeypatch, cfg, ["asleep", "raise", "raise"], script)


@pytest.mark.parametrize("state", ["asleep", "block"])
def test_end_removes_the_private_chat_from_any_state(cfg, monkeypatch, state):
    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=True)
        await _until(lambda: ch.asleep if state == "asleep" else ch.session_id)
        assert await cb["on_end"]() == ""
        assert not runs[0].path.exists()
        assert ch.private is None and ch.run is None and not ch.asleep and ch.session_id is None
        assert ch.model_state() == "cold"

    seen = _owner(monkeypatch, cfg, [state], script)
    assert seen["cancelled"] == (state == "block") and len(seen["runs"]) == 1


def _gone(run):
    return run.path.with_name(f".deleting-{run.path.name}")


def test_a_removal_that_hits_its_bound_is_named_by_end_and_by_a_new_chat(cfg, monkeypatch):
    real = run_folder._finish

    def slow(run):
        time.sleep(0.5)
        return real(run)

    monkeypatch.setattr(run_folder, "_finish", slow)
    monkeypatch.setattr(run_folder, "_SLOW_AFTER_S", 0.05)
    monkeypatch.setattr(run_folder, "PRIVATE_END_WAIT_S", 0.1)

    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=True)
        await _until(lambda: ch.session_id == "c1")
        old = runs[0]
        line = await cb["on_end"]()
        assert line == f"Private session not fully removed: {_gone(old)} stays until localharness next starts"
        cb["on_first_message"](private=True)
        await _until(lambda: ch.session_id == "c2")
        assert await cb["on_new_session"](private=False) == f"Private session not fully removed: {_gone(runs[1])} stays until localharness next starts"

    _owner(monkeypatch, cfg, ["block", "block", "block"], script)


def test_a_new_chat_from_a_normal_chat_says_nothing_was_left(cfg, monkeypatch):
    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=False)
        await _until(lambda: ch.session_id == "c1")
        assert await cb["on_new_session"](private=False) == ""
        await _until(lambda: ch.session_id == "c2")
        assert ch.private is False and not runs[0].path.exists()

    _owner(monkeypatch, cfg, ["block", "block"], script)


def test_a_private_bring_up_on_the_temp_fallback_is_refused_before_any_session(cfg, monkeypatch):
    fallback = []

    def fake_begin(config_dir, private=False):
        fallback.append(run_folder._fallback(private))
        return fallback[0]

    monkeypatch.setattr(run_folder, "begin", fake_begin)

    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=True)
        await _until(lambda: ch.bringup is not None and ch.bringup.failed)
        stage = ch.bringup
        assert stage.private and stage.detail and ch.run is None
        assert runs == [] and not fallback[0].path.exists()

    _owner(monkeypatch, cfg, [], script)


def test_a_normal_chat_after_a_private_one_gets_its_own_run_and_no_private_folder(cfg, monkeypatch):
    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"](private=True)
        await _until(lambda: ch.session_id == "c1")
        await cb["on_end"]()
        cb["on_first_message"](private=False)
        await _until(lambda: ch.session_id == "c2")
        assert not runs[1].private and runs[1].path != runs[0].path and not runs[0].path.exists()
        assert not runs[1].state.exists()
        assert ch.private is False and ch._private_dir is None

    _owner(monkeypatch, cfg, ["block", "block"], script)


def _log_state(*loggers):
    return [(lg, list(lg.handlers), lg.level, lg.propagate) for lg in loggers]


def test_a_private_chat_after_a_normal_one_logs_only_into_its_own_run_and_back(cfg, tmp_path):
    mem, ctx = logging.getLogger("localharness.memory"), logging.getLogger("localharness.agent.context")
    before = _log_state(mem, ctx)
    normal = tmp_path / "normal"
    normal.mkdir()
    run = run_folder.begin(cfg, private=True)
    private = run.state / "agents" / "a"
    private.mkdir(parents=True)

    def flush():
        for lg in (mem, ctx):
            for h in lg.handlers:
                h.flush()

    try:
        start_cmd._route_memory_logs_to_file(normal)
        start_cmd._route_context_logs(normal)
        mem.warning("normal-chat-memory")
        start_cmd._route_memory_logs_to_file(private)
        start_cmd._route_context_logs(private)
        mem.warning("PRIVATE-MEMORY")
        ctx.warning("PRIVATE-CONTEXT")
        flush()
        assert (normal / "memory.log").read_text().count("normal-chat-memory") == 1
        assert "PRIVATE" not in (normal / "memory.log").read_text() + (normal / "context.log").read_text()
        assert "PRIVATE-MEMORY" in (private / "memory.log").read_text()
        assert "PRIVATE-CONTEXT" in (private / "context.log").read_text()
        start_cmd._route_memory_logs_to_file(normal)
        start_cmd._route_context_logs(normal)
        mem.warning("NORMAL-AFTER")
        ctx.warning("NORMAL-AFTER")
        flush()
        assert "NORMAL-AFTER" not in (private / "memory.log").read_text() + (private / "context.log").read_text()
        assert "NORMAL-AFTER" in (normal / "memory.log").read_text()
    finally:
        for lg, handlers, level, propagate in before:
            for h in [h for h in lg.handlers if h not in handlers]:
                lg.removeHandler(h)
                h.close()
            lg.setLevel(level)
            lg.propagate = propagate
        run_folder.end(run)


def _signals(monkeypatch, cfg, ignore):
    seen = {}

    async def script(cb, runs, runner):
        ch = cb["channel"]
        cb["on_first_message"]()
        await _until(lambda: ch.session_id)
        os.kill(os.getpid(), signal.SIGHUP)
        await asyncio.sleep(0.1)
        seen["should_exit"] = runner.should_exit
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0.1)

    def recorder(signum, frame):
        seen["stray"] = True

    original = signal.getsignal(signal.SIGHUP)
    signal.signal(signal.SIGHUP, signal.SIG_IGN if ignore else recorder)
    installed = signal.getsignal(signal.SIGHUP)
    try:
        out = _owner(monkeypatch, cfg, ["block"], script)
        seen["after"], seen["cancelled"] = signal.getsignal(signal.SIGHUP), out["cancelled"]
    finally:
        signal.signal(signal.SIGHUP, original)
    return seen, installed


def test_sighup_stops_mobile_and_a_following_term_is_still_a_first_stop(cfg, monkeypatch):
    seen, installed = _signals(monkeypatch, cfg, ignore=False)
    assert seen["should_exit"] is True and "stray" not in seen
    assert seen["after"] == installed
    assert seen["cancelled"] == 0, "the TERM after the HUP was a first stop, not a second"


def test_sighup_stays_ignored_under_nohup(cfg, monkeypatch):
    seen, _ = _signals(monkeypatch, cfg, ignore=True)
    assert seen["should_exit"] is False
    assert seen["after"] == signal.SIG_IGN


def _channel_with(private, turns=0):
    channel = MobileChannel(bus=EventBus(), config={})
    channel.private, channel._turns = private, turns
    return channel


def test_the_chat_survives_a_sleep_and_forget_chat_clears_it():
    channel = _channel_with(True, 2)
    channel.run, channel._private_dir = SimpleNamespace(private=True), "dir"
    channel.bind_runtime(session_id="s1", agent_id="a", private_dir="dir2")
    assert channel._private_dir == "dir2"
    channel.reset_session()
    assert (channel.private, channel._turns, channel.run.private, channel._private_dir) == (True, 2, True, "dir2")
    channel.forget_chat()
    assert (channel.private, channel._turns, channel.run, channel._private_dir) == (None, 0, None, None)


def test_has_turns_counts_a_root_turn_a_sleeping_chat_and_a_waiting_line():
    assert not _channel_with(True).has_turns()
    assert _channel_with(True, 1).has_turns()
    queued = _channel_with(True)
    queued.submit("typed ahead")
    assert queued.has_turns()
    asleep = _channel_with(True)
    asleep.bind_runtime(session_id="s1", agent_id="a")
    asleep.set_bringup(ASLEEP_STAGE)
    asleep.reset_session()
    assert asleep.has_turns() and asleep.asleep_session_id == "s1"


async def test_a_normal_bind_after_a_private_one_drops_the_private_folder(tmp_path):
    channel = _channel_with(False)
    channel.bind_runtime(session_id="s1", agent_id="a", session_dir=tmp_path / "n", private_dir=tmp_path / "p")
    channel.bind_runtime(session_id="s2", agent_id="a", session_dir=tmp_path / "n")
    assert channel._private_dir is None and channel.session_dir == tmp_path / "n"


async def _private_stack(tmp_path, *, private=True, turns=1, **kw):
    bus, channel, server, client = await _stack(tmp_path, **kw)
    (tmp_path / "sessions").mkdir(exist_ok=True)
    (tmp_path / "held").mkdir(exist_ok=True)
    channel.private, channel._turns, channel._private_dir = private, turns, tmp_path / "held"
    return channel, server, client


async def test_new_carries_the_privacy_and_returns_what_was_left(tmp_path):
    calls = []

    async def restart(*, private=False):
        calls.append(private)
        return "Private session not fully removed: /x stays until localharness next starts" if private is False else ""

    _, _, client = await _private_stack(tmp_path, on_new_session=restart)
    got = await client.post("/api/sessions/new", json={"private": True}, headers=JSON)
    assert got.status_code == 200 and got.json()["private"] is True and got.json()["left"] == ""
    got = await client.post("/api/sessions/new", json={}, headers=JSON)
    assert got.json()["private"] is False and got.json()["left"].startswith("Private session not fully removed")
    got = await client.post("/api/sessions/new", json={"private": "yes"}, headers=JSON)
    assert calls == [True, False, False]


async def test_end_is_refused_in_replay_without_a_handle_with_no_private_chat_and_mid_turn(tmp_path):
    ended = []

    async def on_end():
        ended.append(1)
        return "left over"

    channel, server, client = await _private_stack(tmp_path, on_end=on_end)
    channel._turn_running = True
    got = await client.post("/api/sessions/end", json={}, headers=JSON)
    assert got.status_code == 409 and "stop it first" in got.json()["error"]
    channel._turn_running = False
    channel.private = False
    got = await client.post("/api/sessions/end", json={}, headers=JSON)
    assert got.status_code == 409 and got.json()["error"] == "no private chat is open"
    channel.private = None
    assert (await client.post("/api/sessions/end", json={}, headers=JSON)).status_code == 409
    assert ended == []

    _, _, bare = await _private_stack(tmp_path)
    got = await bare.post("/api/sessions/end", json={}, headers=JSON)
    assert got.status_code == 409 and "cannot restart" in got.json()["error"]

    _, _, replaying = await _private_stack(tmp_path, on_end=on_end, replay=SimpleNamespace(path=tmp_path / "r.jsonl"))
    assert (await replaying.post("/api/sessions/end", json={}, headers=JSON)).status_code == 409
    assert ended == []


async def test_end_ends_the_open_private_chat_and_returns_its_outcome(tmp_path):
    outcomes = ["", "Private session not fully removed: /x stays until localharness next starts"]

    async def on_end():
        return outcomes.pop(0)

    _, _, client = await _private_stack(tmp_path, on_end=on_end)
    first = await client.post("/api/sessions/end", json={}, headers=JSON)
    assert first.status_code == 200 and first.json() == {"status": "ended", "left": ""}
    second = await client.post("/api/sessions/end", json={}, headers=JSON)
    assert second.json()["left"].startswith("Private session not fully removed")


async def test_health_and_hello_say_the_chat_is_private_and_which_one_sleeps(tmp_path):
    channel, _, client = await _private_stack(tmp_path)
    health = (await client.get("/api/health", headers=JSON)).json()
    assert (health["private"], health["asleep_session_id"], health["session_live"]) == (True, None, True)
    channel.set_bringup(ASLEEP_STAGE)
    channel.reset_session()
    health = (await client.get("/api/health", headers=JSON)).json()
    assert (health["private"], health["asleep_session_id"], health["session_live"]) == (True, "s1", False)
    hello = await channel.hello(None, None)
    assert (hello.private, hello.asleep_session_id, hello.session_live) == (True, "s1", False)
    channel.private = False
    assert (await client.get("/api/health", headers=JSON)).json()["private"] is False


def _log(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.jsonl").write_text(
        f'{{"seq": 1, "event_type": "UserMessage", "content": "{text}", "session_id": "{name}"}}\n')


async def test_a_private_chat_is_read_from_its_run_and_never_listed_or_deleted(tmp_path):
    channel, server, client = await _private_stack(tmp_path)
    _log(tmp_path / "held", "p1", "private words")
    _log(tmp_path / "sessions", "n1", "normal words")
    assert server._session_log("p1") == (tmp_path / "held" / "p1.jsonl").resolve()
    assert server._session_log("n1") == (tmp_path / "sessions" / "n1.jsonl").resolve()
    assert server._session_log("p1", private=False) == (tmp_path / "sessions" / "p1.jsonl").resolve()
    assert server._session_log("../p1") is None
    events = await client.get("/api/sessions/p1/events", headers=JSON)
    assert events.status_code == 200 and "private words" in events.text
    listed = (await client.get("/api/sessions", headers=JSON)).json()
    assert [row["session_id"] for row in listed["sessions"]] == ["n1"]
    gone = await client.post("/api/sessions/p1/delete", json={}, headers=JSON)
    assert gone.status_code == 404 and (tmp_path / "held" / "p1.jsonl").exists()
    assert (await client.post("/api/sessions/n1/delete", json={}, headers=JSON)).status_code == 200
    channel._private_dir = None
    assert (await client.get("/api/sessions/p1/events", headers=JSON)).status_code in (200, 404)
    assert "private words" not in (await client.get("/api/sessions/p1/events", headers=JSON)).text


def _fakes(channel, server):
    seen = {"first": [], "new": []}

    def first(*, private=False):
        seen["first"].append(private)

    async def new(*, private=False):
        seen["new"].append(private)
        channel.private, channel._turns = private, 0
        return ""

    server.on_first_message, server.on_new_session = first, new
    return seen


MATRIX = [
    # chat privacy, turns, asleep, queued, request, outcome
    (True, 3, False, False, True, "queued"),
    (False, 3, False, False, False, "queued"),
    (False, 0, False, False, True, "replaced"),
    (False, 3, False, False, True, "changed"),
    (True, 0, False, False, False, "changed"),
    (True, 3, False, False, False, "changed"),
    (False, 0, True, False, True, "changed"),
    (False, 0, False, True, True, "changed"),
]


@pytest.mark.parametrize("chat, turns, asleep, queued, request_private, outcome", MATRIX)
@pytest.mark.parametrize("route", ["message", "command"])
async def test_a_message_or_command_lands_in_the_chat_on_screen(
        tmp_path, route, chat, turns, asleep, queued, request_private, outcome):
    channel, server, client = await _private_stack(tmp_path, private=chat, turns=turns)
    seen = _fakes(channel, server)
    if asleep:
        channel.set_bringup(ASLEEP_STAGE)
        channel.reset_session()
        channel.private = chat
    if queued:
        channel.submit("typed ahead")
    before = channel.queued_input
    body = {"text": "hello" if route == "message" else "/help", "private": request_private}
    got = await client.post(f"/api/sessions/s1/{route}", json=body, headers=JSON)
    if outcome == "changed":
        assert got.status_code == 409
        assert got.json() == {"error": "the chat changed", "private": chat}
        assert seen["new"] == [] and channel.queued_input == before
        return
    assert got.status_code == 200 and channel.queued_input == before + 1
    assert seen["new"] == ([True] if outcome == "replaced" else [])


@pytest.mark.parametrize("route", ["message", "command"])
@pytest.mark.parametrize("request_private", [True, False])
async def test_with_no_chat_a_message_or_command_begins_one_of_its_own_privacy(tmp_path, route, request_private):
    channel, server, client = await _private_stack(tmp_path)
    channel.reset_session()
    channel.forget_chat()
    seen = _fakes(channel, server)
    body = {"text": "hello" if route == "message" else "/help", "private": request_private}
    got = await client.post(f"/api/sessions/s1/{route}", json=body, headers=JSON)
    assert got.status_code == 200 and seen == {"first": [request_private], "new": []}


async def test_a_message_without_a_privacy_field_means_normal(tmp_path):
    channel, server, client = await _private_stack(tmp_path, private=True, turns=1)
    got = await client.post("/api/sessions/s1/message", json={"text": "hi"}, headers=JSON)
    assert got.status_code == 409 and got.json()["private"] is True


async def test_a_connect_prewarms_with_its_privacy_and_never_replaces_a_chat(tmp_path):
    channel, server, _ = await _private_stack(tmp_path)
    channel.reset_session()
    channel.probe_model = _fixed(True)
    seen = _fakes(channel, server)

    channel.forget_chat()
    await server._maybe_prewarm(True)
    assert seen["first"] == [True]
    server.session_over()
    await server._maybe_prewarm(False)
    assert seen["first"] == [True, False]

    server.session_over()
    channel.private = True
    await server._maybe_prewarm(False)
    channel.private = False
    await server._maybe_prewarm(True)
    assert seen == {"first": [True, False], "new": []}


async def test_the_stream_asks_for_the_privacy_in_its_query(tmp_path):
    _, server, _ = await _private_stack(tmp_path)
    asked = []

    async def prewarm(private=False):
        asked.append(private)

    server._maybe_prewarm = prewarm
    await _read_frames(server, 1, path="/api/stream?private=1")
    await _read_frames(server, 1, path="/api/stream")
    assert asked == [True, False]
