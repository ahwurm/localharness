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

import logging
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from localharness.core.events import Observation, ProviderLost, TurnFailed
from localharness.provider import client as client_mod
from localharness.provider import lifecycle
from localharness.provider import server as server_mod
from localharness.provider.client import LLMClient, LLMConfig
from tests.unit.reconnect_support import (
    DroppingChat,
    ModelsEndpoint,
    assert_no_key_leak,
    text_answer,
    tool_call_answer,
)
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
