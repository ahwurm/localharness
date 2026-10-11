# ruff: noqa: F811  (the imported `sessions` fixture is used by name as a test argument)
"""Reconnect end to end: a real `LLMClient`, a real `AgentLoop`, a real `ServerRecovery` flight and the
REPL's own supplier, with no server, port or profile.

The scripted client is two `httpx.MockTransport`s: `provider.client._TRANSPORT` serves the chat endpoint
(`DroppingChat`: its first call drops mid-stream) and `provider.server._TRANSPORT` serves the `/models`
poll (`ModelsEndpoint`: 503 for a while, then 200). Nothing opens a socket, starts a process or reads
`~/.localharness`; config dirs are `tmp_path`; the global `asyncio.sleep` is never patched in a test that
runs a real flight.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import time
from collections import Counter
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from localharness.agent.loop import AgentLoop
from localharness.config.models import ManagedServerConfig
from localharness.core.events import Observation, ProviderLost, TurnFailed
from localharness.provider import client as client_mod
from localharness.provider import lifecycle
from localharness.provider import server as server_mod
from localharness.provider.client import LLMClient, LLMConfig
from tests.unit.launch_support import stub_liveness
from tests.unit.reconnect_support import (
    DroppingChat,
    ModelsEndpoint,
    assert_no_key_leak,
    text_answer,
    tool_call_answer,
    write_dead_pidfile,
    write_live_pidfile,
    write_serve_log,
)
from tests.unit.test_launch_paths_repl import _no_machine_probes  # noqa: F401  (autouse)
from tests.unit.test_repl_model_cmd import _harness, _heavy_primary, _xrepl
from tests.unit.test_loop_effort import _loop, sessions  # noqa: F401
from tests.unit.test_reconnect_loop import _fail_first_build, _Registry, _tool_loop  # noqa: F401

URL = "http://127.0.0.1:9/v1"
SENTINEL = "KEY-SENTINEL-e2e-hand-run"


@pytest.fixture(autouse=True)
def _scene(monkeypatch):
    """A small visibility threshold (the real poll interval and announce delay), and a clean registry of
    held launch locks and unwinding flights once the test is over."""
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.02)
    yield
    assert lifecycle._HELD == {}
    assert not lifecycle._UNWINDING


def _install(monkeypatch, chat: DroppingChat, models: ModelsEndpoint) -> None:
    monkeypatch.setattr(client_mod, "_TRANSPORT", httpx.MockTransport(chat))
    monkeypatch.setattr(server_mod, "_TRANSPORT", httpx.MockTransport(models))


def _client(*, wait: float, api_key: str | SecretStr = "none", base_url: str = URL) -> LLMClient:
    return LLMClient(LLMConfig(base_url=base_url, model="m", api_key=api_key, is_local=False,
                               reconnect_wait_seconds=wait))


def _supplier():
    pair = lifecycle.ServerRecovery(None, None, None)
    return pair, (lambda: pair)


def _lost(bus) -> list[Any]:
    return bus.history(event_types=[ProviderLost])


def _failed(bus) -> list[Any]:
    return bus.history(event_types=[TurnFailed])


# ---------------------------------------------------------------------------
# Success criterion 1: the scripted client
# ---------------------------------------------------------------------------


async def test_a_drop_then_three_503s_then_serving_completes_with_one_provider_lost(
        bus, sessions, monkeypatch):
    chat, models = DroppingChat(), ModelsEndpoint(down_for=3)
    _install(monkeypatch, chat, models)
    _, supplier = _supplier()

    async with _client(wait=5.0) as llm:
        loop = _loop(llm, bus, server_recovery=supplier)
        assert await loop.run_turn("go") == "ok"

    lost = _lost(bus)
    assert len(lost) == 1 and lost[0].wait_seconds == 5.0
    assert _failed(bus) == []
    assert models.calls == 4  # three 503s, then the 200
    assert set(models.urls) == {URL + "/models"}
    assert len(chat.bodies) == 2
    assert chat.bodies[0] == chat.bodies[1]  # an identical re-send: equal raw request bytes


async def test_a_tool_call_carried_by_the_re_sent_answer_runs_exactly_once(bus, sessions, monkeypatch):
    chat = DroppingChat(answers=[tool_call_answer("record_it", "{}"), text_answer("done")])
    models = ModelsEndpoint(down_for=2)
    _install(monkeypatch, chat, models)
    _, supplier = _supplier()
    registry = _Registry()

    async with _client(wait=5.0) as llm:
        loop = _tool_loop(llm, bus, registry, server_recovery=supplier)
        assert await loop.run_turn("go") == "done"

    assert registry.ran == 1
    assert len([o for o in bus.history(event_types=[Observation]) if o.tool_call_id == "call_1"]) == 1
    assert len(_lost(bus)) == 1 and _failed(bus) == []
    assert chat.bodies[0] == chat.bodies[1]  # the dropped call and its re-send
    assert len(chat.bodies) == 3  # then the call that carries the tool result


async def test_a_build_that_committed_a_summary_then_lost_the_server_is_absorbed(
        bus, sessions, monkeypatch):
    chat, models = DroppingChat(drop_first=False), ModelsEndpoint(down_for=3)
    _install(monkeypatch, chat, models)
    _, supplier = _supplier()

    async with _client(wait=5.0) as llm:
        loop = _loop(llm, bus, server_recovery=supplier)
        builds = _fail_first_build(loop)
        assert await loop.run_turn("go") == "ok"

    assert len(builds) == 2  # the build ran again once the server answered
    assert len(_lost(bus)) == 1 and _failed(bus) == []
    assert sessions[0].turn_start_idx == 1  # the _absorb_commit rule: start(0) + 1
    assert len(chat.bodies) == 1  # the chat call was made once, after the rebuild


async def test_a_blip_that_answers_on_the_first_poll_publishes_nothing(bus, sessions, monkeypatch):
    chat, models = DroppingChat(), ModelsEndpoint(down_for=0)
    _install(monkeypatch, chat, models)
    _, supplier = _supplier()

    async with _client(wait=5.0) as llm:
        assert await _loop(llm, bus, server_recovery=supplier).run_turn("go") == "ok"

    assert models.calls == 1
    assert _lost(bus) == [] and _failed(bus) == []
    assert chat.bodies[0] == chat.bodies[1]


# ---------------------------------------------------------------------------
# Success criterion 2: a hand-run keyed server, and reconnect off
# ---------------------------------------------------------------------------


async def test_a_hand_run_keyed_server_is_polled_with_the_clients_own_key_at_its_own_address(
        bus, sessions, monkeypatch, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    chat, models = DroppingChat(), ModelsEndpoint(down_for=3)
    _install(monkeypatch, chat, models)
    pair, supplier = _supplier()  # a pure poll: no config dir, nothing to relaunch

    async with _client(wait=5.0, api_key=SecretStr(SENTINEL)) as llm:
        loop = _loop(llm, bus, server_recovery=supplier)
        summary = await loop.run_turn("go")

    assert summary == "ok"
    assert models.calls == 4
    assert set(models.auth) == {"Bearer " + SENTINEL}
    assert set(models.urls) == {URL + "/models"}
    (lost,) = _lost(bus)
    argv_like = repr((pair, [e.model_dump() for e in bus.history()]))
    assert_no_key_leak(SENTINEL, summary, lost.detail, repr(lost), argv_like,
                       caplog=caplog, capsys=capsys)


async def test_with_reconnect_off_a_drop_takes_exactly_one_fallback_retry(bus, sessions, monkeypatch):
    chat, models = DroppingChat(), ModelsEndpoint(down=lambda: True)
    _install(monkeypatch, chat, models)
    _, supplier = _supplier()

    async with _client(wait=0.0) as llm:
        assert await _loop(llm, bus, server_recovery=supplier).run_turn("go") == "ok"

    assert models.calls == 0  # no poll at all
    assert _lost(bus) == [] and _failed(bus) == []
    assert len(chat.bodies) == 2 and chat.bodies[0] == chat.bodies[1]


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------


async def test_a_server_that_never_returns_ends_the_turn_at_the_bound_with_llm_error(
        bus, sessions, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    chat, models = DroppingChat(down=lambda: True), ModelsEndpoint(down=lambda: True)
    _install(monkeypatch, chat, models)
    _, supplier = _supplier()

    async with _client(wait=1.0) as llm:
        summary = await _loop(llm, bus, server_recovery=supplier).run_turn("go")

    (failed,) = _failed(bus)
    assert failed.reason == "llm_error"
    assert "did not come back within 1s" in failed.detail
    assert "did not come back within 1s" in summary
    assert not [f for f in _failed(bus) if f.reason == "internal_error"]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(_lost(bus)) == 1
    assert len(chat.bodies) == 1  # one wait per turn: no re-send after the bound


# ---------------------------------------------------------------------------
# Success criteria 3 and 4: the REPL's own supplier, many callers, the lock
# ---------------------------------------------------------------------------

PRIMARY_URL = "http://localhost:8081/v1"


class ToolThenDone(DroppingChat):
    """A chat endpoint that answers by what the request carries, so the answer does not depend on which
    caller arrives first: a request with no tool result yet gets a tool call, one with a result gets
    the final text. Drops (`down`) behave as in `DroppingChat`."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        roles = [m["role"] for m in json.loads(request.content)["messages"]]
        self.answers = [text_answer("done") if "tool" in roles else tool_call_answer("record_it", "{}")]
        self._served = 0
        return super().__call__(request)


class Rig:
    """A REPL over a harness whose primary the harness launched, a real client at that address, and the
    two MockTransports. `up` is whether the server answers (both `/models` and the chat endpoint read it);
    `alive` is what the strategy's liveness reports (a stand-in for `docker inspect`). `start_server`
    is stubbed: it records its argv and keywords, then runs `on_start` (default: write a live pidfile,
    bring the server up)."""

    def __init__(self, tmp_path, monkeypatch, *, server=None, wait: float = 30.0,
                 active: bool = True, chat_class=DroppingChat) -> None:
        self.up = False
        self.alive = False
        self.starts: list[dict[str, Any]] = []
        self.on_start = self._bring_up
        self.chat = chat_class(drop_first=False, down=lambda: not self.up)
        self.models = ModelsEndpoint(down=lambda: not self.up)
        _install(monkeypatch, self.chat, self.models)
        stub_liveness(monkeypatch, lambda: self.alive)
        monkeypatch.setattr(server_mod, "start_server", self._start)

        self.harness = _harness(server=server if server is not None else _heavy_primary())
        self.llm = _client(wait=wait, base_url=PRIMARY_URL)
        self.repl, self.channel, agent, _ = _xrepl(tmp_path, self.harness, {})
        agent._llm = self.llm
        self.repl._active_heavy = (self.harness.server, PRIMARY_URL) if active else None
        self.cfg = self.repl._server_config_dir

    def _start(self, config_dir, cmd, **kw) -> int:
        self.starts.append({"cmd": list(cmd), **kw})
        self.on_start()
        return 4321

    def _bring_up(self) -> None:
        write_live_pidfile(self.cfg)
        self.up = True
        self.alive = True

    def loop(self, **kw) -> AgentLoop:
        return _loop(self.llm, self.bus, server_recovery=self.repl.server_recovery, **kw)

    async def aclose(self) -> None:
        await self.llm.aclose()


@pytest.fixture
def rig(tmp_path, monkeypatch, bus):
    def build(**kw) -> Rig:
        r = Rig(tmp_path, monkeypatch, **kw)
        r.bus = bus
        return r

    return build


async def _until(predicate, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


async def test_three_children_under_one_dead_launched_server_share_one_relaunch(
        sessions, rig, caplog):
    caplog.set_level(logging.DEBUG)
    r = rig(chat_class=ToolThenDone)
    registry = _Registry()
    write_live_pidfile(r.cfg)  # the heavy primary is running, then it dies
    loops = [_tool_loop(r.llm, r.bus, registry, server_recovery=r.repl.server_recovery)
             for _ in range(3)]

    async def the_server_dies_and_nothing_answers():
        await asyncio.sleep(0.1)
        write_dead_pidfile(r.cfg)

    try:
        results = await asyncio.wait_for(asyncio.gather(
            *(loop.run_turn(f"go-{i}") for i, loop in enumerate(loops)),
            the_server_dies_and_nothing_answers()), 20)
    finally:
        await r.aclose()

    assert results[:3] == ["done", "done", "done"]
    assert len(r.starts) == 1  # one relaunch for three callers
    assert r.starts[0]["cmd"]  # the argv itself is pinned by the launch-path tests; only the count is here
    assert len(_lost(r.bus)) == 1
    assert _failed(r.bus) == []
    assert registry.ran == 3  # each child's one tool call ran once; no re-send re-ran a tool
    counts = Counter(r.chat.bodies)
    # the dropped call and its re-send are equal bytes, once per child; the later calls are distinct
    assert sorted(counts.values()) == [1, 1, 1, 2, 2, 2]
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


async def test_a_relaunch_racing_another_holders_launch_starts_nothing(sessions, rig):
    r = rig()
    write_live_pidfile(r.cfg)
    lock = server_mod.lock_path(r.cfg)
    lock.parent.mkdir(parents=True, exist_ok=True)
    foreign = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(foreign, fcntl.LOCK_EX)
    turn = asyncio.ensure_future(r.loop().run_turn("go"))
    try:
        await asyncio.sleep(0.1)
        write_dead_pidfile(r.cfg)  # the server is gone: the flight asks to relaunch and waits for the lock
        await _until(lambda: lifecycle.LAUNCH_LOCK_WAIT_LINE in r.channel.messages)  # waiting on the lock
        write_live_pidfile(r.cfg)  # the other holder recorded its launch ...
        r.up = True  # ... and its server answers
        os.close(foreign)  # ... then released the lock
        foreign = -1
        assert await asyncio.wait_for(turn, 10) == "ok"
    finally:
        if foreign >= 0:
            os.close(foreign)
        await r.aclose()

    assert r.starts == []  # nothing was started: the pidfile re-check under the lock saw a live server
    assert len(_lost(r.bus)) == 1 and _failed(r.bus) == []


async def test_esc_during_a_relaunch_cancels_the_flight_and_frees_the_lock(sessions, rig):
    r = rig()
    write_live_pidfile(r.cfg)
    r.on_start = lambda: write_live_pidfile(r.cfg)  # the process is up, the server never answers
    turn = asyncio.ensure_future(r.loop().run_turn("go"))
    try:
        await asyncio.sleep(0.1)
        write_dead_pidfile(r.cfg)
        await _until(lambda: len(r.starts) == 1)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        await _until(lambda: not lifecycle._UNWINDING and lifecycle._HELD == {})
    finally:
        await r.aclose()

    third = os.open(server_mod.lock_path(r.cfg), os.O_RDWR)
    try:
        fcntl.flock(third, fcntl.LOCK_EX | fcntl.LOCK_NB)  # lockable at once: the flight let go of it
    finally:
        os.close(third)
    assert _failed(r.bus) == []
    assert len(r.starts) == 1


async def test_a_relaunch_that_dies_ends_the_turn_with_the_restart_sentence_and_the_tail(
        sessions, rig, caplog):
    caplog.set_level(logging.DEBUG)
    r = rig()
    write_live_pidfile(r.cfg)

    def dies():
        write_dead_pidfile(r.cfg)
        write_serve_log(r.cfg, "cuda oom tail")

    r.on_start = dies
    turn = asyncio.ensure_future(r.loop().run_turn("go"))
    try:
        await asyncio.sleep(0.1)
        write_dead_pidfile(r.cfg)
        summary = await asyncio.wait_for(turn, 10)
    finally:
        await r.aclose()

    (failed,) = _failed(r.bus)
    assert failed.reason == "llm_error"
    assert "it was restarted once and the restart failed" in failed.detail
    assert "cuda oom tail" in failed.detail
    assert "cuda oom tail" in summary
    assert len(r.starts) == 1
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


async def test_a_bound_that_passes_during_a_relaunch_ends_the_turn_with_one_llm_error(
        sessions, rig, caplog):
    caplog.set_level(logging.DEBUG)
    r = rig(wait=1.0)
    write_live_pidfile(r.cfg)
    r.on_start = lambda: write_live_pidfile(r.cfg)  # up as a process, never answering
    started = time.monotonic()
    turn = asyncio.ensure_future(r.loop().run_turn("go"))
    try:
        await asyncio.sleep(0.1)
        write_dead_pidfile(r.cfg)
        summary = await asyncio.wait_for(turn, 10)
    finally:
        await r.aclose()
    elapsed = time.monotonic() - started

    (failed,) = _failed(r.bus)
    assert failed.reason == "llm_error"
    # the bound ran out inside the relaunch's own readiness wait: the relaunch is held to the flight's
    # bound, so the turn ends at the bound with the bound's sentence, not at activate's later clock
    assert "the model server did not come back within 1s" in failed.detail
    assert "did not come back within 1s" in summary
    assert "the restart failed" not in failed.detail
    assert elapsed < 2.5  # activate's own wait (3 s polls, a 1 s floor from a later start) ran to ~4 s
    assert len(r.starts) == 1
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


async def test_a_gpu_false_managed_primary_that_dies_ends_the_turn_at_once_with_its_log_tail(
        sessions, rig):
    cpu = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf",
                              port=8081, gpu=False)
    r = rig(server=cpu, active=False, wait=30.0)
    write_serve_log(r.cfg, "segfault tail")  # no pidfile: the process is gone

    pair = r.repl.server_recovery()
    assert pair.relaunch is None and pair.config_dir == r.cfg

    started = time.monotonic()
    try:
        summary = await asyncio.wait_for(r.loop().run_turn("go"), 10)
    finally:
        await r.aclose()

    assert time.monotonic() - started < 1.0  # not the 30 s bound
    (failed,) = _failed(r.bus)
    assert failed.reason == "llm_error"
    assert "the model server exited. Log tail:" in failed.detail and "segfault tail" in failed.detail
    assert "segfault tail" in summary
    assert r.starts == []
