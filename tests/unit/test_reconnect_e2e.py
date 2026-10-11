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

    def _start(self, config_dir, cmd, *, runtime="vllm", api_key=None, record_env=None, dropped_out=None,
               xdg_config_home=None, **kw) -> int:
        # start_server's own keywords are named (IN-56); the tail only tolerates one added later
        self.starts.append({"cmd": list(cmd), "runtime": runtime, "api_key": api_key,
                            "record_env": record_env, "dropped_out": dropped_out,
                            "xdg_config_home": xdg_config_home, **kw})
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
    assert "The model server did not come back within 1s" in failed.detail
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

    assert time.monotonic() - started < 5.0  # at once, not the 30 s bound (loose enough for a busy CI box)
    (failed,) = _failed(r.bus)
    assert failed.reason == "llm_error"
    assert "The model server exited. Log tail:" in failed.detail and "segfault tail" in failed.detail
    assert "segfault tail" in summary
    assert r.starts == []


# ---------------------------------------------------------------------------
# One outage per root turn: a root and the child it delegates to share one bound and one relaunch
# ---------------------------------------------------------------------------


class _Delegating:
    """A registry whose one tool is a delegation: the model server dies, then a child loop (on the same
    client and the same supplier) runs its own turn, and its ending is the tool's result."""

    def __init__(self, rig: Rig, child_bound: float | None = None, on_death=None, after_child=None) -> None:
        self.rig = rig
        self.child_bound = child_bound  # a child with this much duration budget left
        self.on_death = on_death
        self.after_child = after_child  # what happens in the root's turn once the child has returned
        self.child_summaries: list[str] = []

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
        from localharness.tools.base import ToolResult

        self.rig.up = False
        self.rig.alive = False
        write_dead_pidfile(self.rig.cfg)  # the launched server dies while the child works
        if self.on_death is not None:
            self.on_death()
        child = _loop(self.rig.llm, self.rig.bus, server_recovery=self.rig.repl.server_recovery)
        if self.child_bound is not None:
            real = child._reconnect_bound
            child._reconnect_bound = (  # type: ignore[method-assign]
                lambda session=None: min(real(), self.child_bound) if session is not None else real())
        self.child_summaries.append(await child.run_turn("look it up"))
        if self.after_child is not None:
            then = self.after_child()
            if then is not None:
                await then
        return ToolResult(output=self.child_summaries[-1], success=True)


def _serving(r: Rig) -> None:
    write_live_pidfile(r.cfg)
    r.up = True
    r.alive = True


async def test_a_root_after_its_child_spent_the_bound_waits_no_second_bound(sessions, rig, caplog):
    """CR-51: the child's flight relaunches the dead server once and spends the whole bound on a server
    that never answers; the root's next call meets the same outage and takes one look, not a second
    bound: one relaunch, one ProviderLost, and the person's whole wait stays within one bound."""
    caplog.set_level(logging.DEBUG)
    r = rig(chat_class=ToolThenDone, wait=1.0)
    _serving(r)

    def up_as_a_process_never_answering():
        write_live_pidfile(r.cfg)
        r.alive = True

    r.on_start = up_as_a_process_never_answering
    delegating = _Delegating(r)
    root = _tool_loop(r.llm, r.bus, delegating, server_recovery=r.repl.server_recovery)
    started = time.monotonic()
    try:
        summary = await asyncio.wait_for(root.run_turn("go"), 10)
    finally:
        await r.aclose()
    elapsed = time.monotonic() - started

    assert len(r.starts) == 1  # one relaunch for the whole root turn
    assert len(_lost(r.bus)) == 1  # one announcement for the one outage
    assert elapsed < 1.6  # one bound (1 s) and one short look, not a second bound
    assert "did not come back within 1s" in delegating.child_summaries[0]
    assert "did not come back within 1s" in summary
    assert [f.reason for f in _failed(r.bus)] == ["llm_error", "llm_error"]  # the child, then the root
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


async def test_a_root_after_its_childs_relaunch_died_never_relaunches_again(sessions, rig):
    """CR-51: the one relaunch already happened (and died) inside the child's turn; the root's next drop
    in the same turn ends with that same sentence, with no second launch of a crash-looping server."""
    r = rig(chat_class=ToolThenDone, wait=5.0)
    _serving(r)

    def dies():
        write_dead_pidfile(r.cfg)
        write_serve_log(r.cfg, "cuda oom tail")

    r.on_start = dies
    delegating = _Delegating(r)
    root = _tool_loop(r.llm, r.bus, delegating, server_recovery=r.repl.server_recovery)
    started = time.monotonic()
    try:
        summary = await asyncio.wait_for(root.run_turn("go"), 10)
    finally:
        await r.aclose()

    assert len(r.starts) == 1
    assert len(_lost(r.bus)) <= 1
    assert time.monotonic() - started < 2.5  # well inside the 5 s bound: nothing waited for it twice
    assert "it was restarted once and the restart failed" in delegating.child_summaries[0]
    assert "it was restarted once and the restart failed" in summary
    assert "cuda oom tail" in summary


async def test_a_new_root_turn_gets_a_whole_bound_and_its_relaunch_again(sessions, rig):
    """The outage record lasts one root turn: the person's next message relaunches the server again."""
    r = rig(chat_class=ToolThenDone, wait=5.0)
    _serving(r)
    r.on_start = lambda: write_dead_pidfile(r.cfg)
    delegating = _Delegating(r)
    try:
        await asyncio.wait_for(
            _tool_loop(r.llm, r.bus, delegating, server_recovery=r.repl.server_recovery).run_turn("go"), 10)
        assert len(r.starts) == 1
        r.on_start = r._bring_up  # this time the relaunch comes up
        summary = await asyncio.wait_for(r.loop().run_turn("again"), 10)
    finally:
        await r.aclose()

    assert len(r.starts) == 2
    assert summary == "done"


async def test_a_call_that_goes_through_ends_the_outage_for_every_loop_on_the_pair(sessions, rig):
    """A drop the child could not end, then the server comes back and a call goes through: the next drop
    in the same root turn is a new one, with its own announcement and its own relaunch."""
    r = rig(wait=5.0)
    _serving(r)
    pair = r.repl.server_recovery()
    from localharness.provider.lifecycle import _Outage

    pair._flight.outage = _Outage(time.monotonic() - 60.0)  # an outage whose bound is long spent ...
    pair._flight.outage.failed = (TimeoutError, "the model server did not come back within 5s")
    lifecycle._OPEN_OUTAGES[pair._flight] = pair.base_url
    from localharness.agent import loop as loop_mod

    depth = loop_mod._TURN_DEPTH.set(1)  # a child's turn: only the call going through can end it
    try:
        # ... and a call that goes through at that address ends it
        assert await asyncio.wait_for(_loop(r.llm, r.bus, server_recovery=r.repl.server_recovery)
                                      .run_turn("go"), 10) == "ok"
    finally:
        loop_mod._TURN_DEPTH.reset(depth)
        await r.aclose()
    assert pair._flight.outage is None
    assert pair._flight not in lifecycle._OPEN_OUTAGES


async def test_a_child_turn_never_ends_the_outage_but_a_root_turn_does(sessions, rig):
    from localharness.agent import loop as loop_mod
    from localharness.provider.lifecycle import _Outage

    r = rig(wait=5.0)
    pair = r.repl.server_recovery()
    seen: list[bool] = []

    async def turn() -> None:
        pair._flight.outage = _Outage(time.monotonic())
        lifecycle._OPEN_OUTAGES[pair._flight] = pair.base_url
        loop = _loop(r.llm, r.bus, server_recovery=r.repl.server_recovery)
        loop._execute_loop = _record_outage  # type: ignore[method-assign]
        await loop.run_turn("go")

    async def _record_outage(*_a, **_k) -> str:
        seen.append(pair._flight.outage is not None)
        return "ok"

    try:
        depth = loop_mod._TURN_DEPTH.set(1)
        try:
            await turn()  # a child's turn: the outage is still on record while it runs
        finally:
            loop_mod._TURN_DEPTH.reset(depth)
        await turn()  # a root turn: it is gone before the loop runs
    finally:
        await r.aclose()
    assert seen == [True, False]
    assert loop_mod._TURN_DEPTH.get() == 0


async def test_a_child_whose_budget_ends_before_the_relaunch_starts_leaves_it_to_the_root(sessions, rig):
    """WR-60: the child (0.3 s of budget left) leaves while the shared relaunch still waits behind another
    process's launch lock. The root, with the whole bound, joins that flight: one server is started and the
    root's turn completes; it never ends at once with a false "restarted once and exited again"."""
    r = rig(chat_class=ToolThenDone, wait=30.0)
    _serving(r)
    lock = server_mod.lock_path(r.cfg)
    lock.parent.mkdir(parents=True, exist_ok=True)
    held: list[int] = []

    def foreign_launch_holds_the_lock():
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        held.append(fd)
        asyncio.get_running_loop().call_later(0.8, lambda: (os.close(fd), held.remove(fd)))

    delegating = _Delegating(r, child_bound=0.3, on_death=foreign_launch_holds_the_lock)
    root = _tool_loop(r.llm, r.bus, delegating, server_recovery=r.repl.server_recovery)
    try:
        summary = await asyncio.wait_for(root.run_turn("go"), 10)
    finally:
        for fd in held:
            os.close(fd)
        await r.aclose()

    assert "did not come back" in delegating.child_summaries[0]  # the child left at its own bound
    assert len(r.starts) == 1  # the one relaunch happened, for the root
    # one line for the one death, stating the configured bound (true for the root's wait), never the
    # child's 0.3 s (WR-23)
    assert [e.wait_seconds for e in _lost(r.bus)] == [30.0]
    assert summary == "done"
    assert [f.reason for f in _failed(r.bus)] == ["llm_error"]  # the child only


def _foreign_lock(r: Rig) -> list[int]:
    """Hold the launch lock from a raw fd, as another LocalHarness's launch would; returns the fd holder."""
    lock = server_mod.lock_path(r.cfg)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return [fd]


def _lockable(r: Rig) -> bool:
    fd = os.open(server_mod.lock_path(r.cfg), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


async def test_esc_after_a_child_left_its_wait_stops_the_shared_flight_before_any_relaunch(sessions, rig):
    """WR-24 / WR-62: the child left at its budget while the shared relaunch waited behind another
    launch's lock; the person presses Esc before the root rejoins. The root turn's end cancels the flight
    nobody awaits: once the lock frees, no server is started and the launch lock is free."""
    r = rig(chat_class=ToolThenDone, wait=30.0)
    _serving(r)
    held: list[int] = []
    turn: list[asyncio.Task] = []

    async def esc() -> None:  # Esc while the root is still handling the child's result
        turn[0].cancel()
        await asyncio.sleep(0)

    delegating = _Delegating(r, child_bound=0.3, on_death=lambda: held.extend(_foreign_lock(r)),
                             after_child=esc)
    root = _tool_loop(r.llm, r.bus, delegating, server_recovery=r.repl.server_recovery)
    try:
        turn.append(asyncio.ensure_future(root.run_turn("go")))
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn[0], 10)
        os.close(held.pop())  # the other launch finishes and lets go of the lock
        await asyncio.sleep(0.5)  # a live flight would now take the lock and relaunch
        await _until(lambda: not lifecycle._UNWINDING and lifecycle._HELD == {})
    finally:
        for fd in held:
            os.close(fd)
        await r.aclose()

    assert "did not come back" in delegating.child_summaries[0]
    assert r.starts == []  # no relaunch with no turn running
    assert _lockable(r)


async def test_a_kill_file_after_a_child_left_its_wait_stops_the_shared_flight(sessions, rig, tmp_path):
    """WR-24 / WR-62: as above, but the root's turn ends at the KILL file: no relaunch after KILL."""
    r = rig(chat_class=ToolThenDone, wait=30.0)
    _serving(r)
    held: list[int] = []
    kill = tmp_path / "KILL"
    delegating = _Delegating(r, child_bound=0.3, on_death=lambda: held.extend(_foreign_lock(r)),
                             after_child=lambda: kill.write_text("stop", encoding="utf-8"))
    root = _tool_loop(r.llm, r.bus, delegating, server_recovery=r.repl.server_recovery,
                      kill_file_path=kill)
    try:
        await asyncio.wait_for(root.run_turn("go"), 10)
        os.close(held.pop())
        await asyncio.sleep(0.5)
        await _until(lambda: not lifecycle._UNWINDING and lifecycle._HELD == {})
    finally:
        for fd in held:
            os.close(fd)
        await r.aclose()

    assert sessions[0].terminated_reason == "kill_file"  # the root's session (the child's is next)
    assert r.starts == []
    assert _lockable(r)
