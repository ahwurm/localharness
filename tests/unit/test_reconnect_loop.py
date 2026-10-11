# ruff: noqa: F811  (the imported `sessions` fixture is used by name as a test argument)
"""The loop's half of reconnect (RECON-02/04/06/08): one decision table at both seams, the bounded wait
through the server's own address and key, an identical re-send, and endings that carry the flight's
sentence.

Hermetic: a fake pair or a real `ServerRecovery` over a patched `server.wait_ready` / `httpx.MockTransport`,
`tmp_path` config dirs, `_QUEUE_VISIBILITY_SECONDS` patched small. The global `asyncio.sleep` is patched
only in tests that run no real flight.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from localharness.agent.loop import AgentLoop
from localharness.core.events import TurnFailed
from localharness.provider import client as client_mod
from localharness.provider.client import ProviderConnectionError
from tests.unit.test_loop_effort import _answer, _Scripted, _loop, sessions  # noqa: F401

URL = "http://127.0.0.1:8080/v1"


class FakePair:
    """A ServerRecovery stand-in: records the keywords of every `recover`, optionally raising."""

    def __init__(self, base_url: str | None = None, error: BaseException | None = None) -> None:
        self.base_url = base_url
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def recover(self, **kw: Any) -> None:
        self.calls.append(kw)
        if self.error is not None:
            raise self.error


async def _no_sleep(*_a: Any, **_k: Any) -> None:
    return None


@pytest.fixture
def sleeps(monkeypatch):
    """Patch the loop's asyncio.sleep with a recorder and the fallback delay with a distinctive value."""
    seen: list[float] = []

    async def record(delay: float, *_a: Any, **_k: Any) -> None:
        seen.append(delay)

    monkeypatch.setattr("localharness.agent.loop.asyncio.sleep", record)
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.37)
    return seen


def _scripted(script: list[Any], *, wait: float | None = 30.0, base_url: Any = URL) -> _Scripted:
    llm = _Scripted(script, provider_type="vllm", base_url=base_url if isinstance(base_url, str) else URL)
    llm.config.base_url = base_url
    if wait is not None:
        llm.config.reconnect_wait_seconds = wait
    return llm


def _pce(text: str = "refused") -> ProviderConnectionError:
    return ProviderConnectionError(text)


def _failed(bus) -> list[Any]:
    return bus.history(event_types=[TurnFailed])


# ---------------------------------------------------------------------------
# The decision table, stream seam
# ---------------------------------------------------------------------------


async def test_row_one_waits_once_then_re_sends(bus, sessions, sleeps):
    pair = FakePair()
    llm = _scripted([_pce(), _answer("ok")])
    loop = _loop(llm, bus, server_recovery=lambda: pair)

    assert await loop.run_turn("go") == "ok"

    assert len(pair.calls) == 1
    assert pair.calls[0]["wait_seconds"] == 30.0
    assert pair.calls[0]["base_url"] == URL
    assert sessions[0].reconnect_used is True
    assert sessions[0].connection_retry_used is False  # re-armed by the call that went through
    assert len(llm.calls) == 2
    assert sleeps == []  # the wait is the pair's, not a sleep


class _Registry:
    """A registry with one recording tool: counts how often it ran."""

    def __init__(self) -> None:
        self.ran = 0

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
        from localharness.tools.base import ToolResult

        self.ran += 1
        return ToolResult(output="tool-result", success=True)


def _tool_answer(call_id: str = "tc-1") -> tuple[Any, Any]:
    from tests.conftest import FakeLLMResponse, FakeToolCall

    call = FakeToolCall(id=call_id, name="bash", arguments={"cmd": "ls"})
    return FakeLLMResponse(content=None, tool_calls=[call]), None


def _tool_loop(llm: Any, bus: Any, registry: Any, **kw: Any) -> AgentLoop:
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig, PermissionConfig, SelfCheckConfig

    cfg = AgentConfig(
        name="test-agent", role="Test agent.", self_check=SelfCheckConfig(enabled=False),
        permissions=PermissionConfig(mode="unattended"),
    )
    return AgentLoop(
        config=cfg, llm=llm, bus=bus, context_manager=ContextManager(max_context_tokens=131_072),
        tool_registry=registry, permission_evaluator=PermissionEvaluator(), **kw,
    )


async def test_a_second_outage_after_a_call_went_through_takes_the_fallback(bus, sessions, sleeps):
    pair = FakePair()
    registry = _Registry()
    llm = _scripted([_pce("one"), _tool_answer(), _pce("two"), _answer("ok")])
    loop = _tool_loop(llm, bus, registry, server_recovery=lambda: pair)

    assert await loop.run_turn("go") == "ok"

    assert len(pair.calls) == 1  # one wait per turn
    assert sleeps.count(0.37) == 1  # the second outage took the single fallback retry
    assert sessions[0].reconnect_used is True
    assert registry.ran == 1


async def test_row_three_the_retry_after_a_wait_failing_again_ends_the_turn(bus, sessions, sleeps):
    pair = FakePair()
    llm = _scripted([_pce("one"), _pce("two")])
    loop = _loop(llm, bus, server_recovery=lambda: pair)

    summary = await loop.run_turn("go")

    assert summary.startswith("Agent encountered an error: ProviderConnectionError: two")
    assert len(pair.calls) == 1
    assert sessions[0].terminated_reason == "error"


async def test_reconnect_off_takes_the_fallback_retry(bus, sessions, sleeps):
    pair = FakePair()
    llm = _scripted([_pce(), _answer("ok")], wait=None)  # the stand-in has no knob at all
    loop = _loop(llm, bus, server_recovery=lambda: pair)

    assert await loop.run_turn("go") == "ok"

    assert pair.calls == []
    assert sleeps == [0.37]
    assert sessions[0].reconnect_used is False


async def test_a_knob_of_zero_is_the_fallback_and_a_tiny_one_is_the_wait(bus, sessions, sleeps):
    off = FakePair()
    llm = _scripted([_pce(), _answer("ok")], wait=0.0)
    assert await _loop(llm, bus, server_recovery=lambda: off).run_turn("go") == "ok"
    assert off.calls == [] and sleeps == [0.37]

    on = FakePair()
    llm = _scripted([_pce(), _answer("ok")], wait=0.001)
    assert await _loop(llm, bus, server_recovery=lambda: on).run_turn("go") == "ok"
    assert len(on.calls) == 1 and on.calls[0]["wait_seconds"] == 0.001

    again = FakePair()
    llm = _scripted([_pce(), _pce("again")], wait=0.001)
    summary = await _loop(llm, bus, server_recovery=lambda: again).run_turn("go")
    assert "again" in summary and len(again.calls) == 1


# ---------------------------------------------------------------------------
# The build seam
# ---------------------------------------------------------------------------

_UNREACHABLE = "model server unreachable at http://127.0.0.1:9/tokenize (Connection refused)"


def _fail_first_build(loop: AgentLoop, commit: tuple[int, int] | None = (0, 1)) -> list[int]:
    """Make the loop's first build_messages raise (after recording a committed summary), later ones real."""
    real = loop._ctx.build_messages
    calls: list[int] = []

    async def build(*a: Any, **k: Any):
        calls.append(1)
        if len(calls) == 1:
            loop._ctx.last_commit = commit
            raise ProviderConnectionError(_UNREACHABLE)
        return await real(*a, **k)

    loop._ctx.build_messages = build  # type: ignore[method-assign]
    return calls


async def test_build_seam_waits_absorbs_the_commit_and_rebuilds(bus, sessions, sleeps):
    pair = FakePair()
    llm = _scripted([_answer("ok")])
    loop = _loop(llm, bus, server_recovery=lambda: pair)
    calls = _fail_first_build(loop)

    assert await loop.run_turn("go") == "ok"

    assert len(pair.calls) == 1
    assert len(calls) == 2  # the original build ran again after the wait
    assert sessions[0].turn_start_idx == 1  # start(0) + 1: the commit was absorbed before the retry
    assert _failed(bus) == []


async def test_build_seam_failure_is_one_llm_error_with_the_note(bus, sessions, sleeps):
    pair = FakePair(error=TimeoutError("the model server did not come back within 30s"))
    llm = _scripted([_answer("never reached")])
    loop = _loop(llm, bus, server_recovery=lambda: pair)
    _fail_first_build(loop)

    summary = await loop.run_turn("go")

    failed = _failed(bus)
    assert len(failed) == 1 and failed[0].reason == "llm_error"
    assert "unreachable" in failed[0].detail and "did not come back within 30s" in failed[0].detail
    assert "unreachable" in summary
    assert sessions[0].turn_start_idx == 1  # the commit was absorbed even though the turn ended
    assert llm.calls == []


async def test_build_seam_without_a_note_reraises_the_original(bus, sessions, sleeps):
    llm = _scripted([_answer("never reached")], wait=None)
    loop = _loop(llm, bus)
    builds: list[int] = []

    async def always_fail(*a: Any, **k: Any):
        builds.append(1)
        raise ProviderConnectionError(_UNREACHABLE)

    loop._ctx.build_messages = always_fail  # type: ignore[method-assign]
    await loop.run_turn("go")

    failed = _failed(bus)
    assert len(failed) == 1 and failed[0].detail == _UNREACHABLE
    assert len(builds) == 2  # the fallback retry rebuilt once, the second failure ended the turn
    assert sleeps == [0.37]


# ---------------------------------------------------------------------------
# Endings
# ---------------------------------------------------------------------------


async def test_stream_seam_failure_carries_the_flights_sentence(bus, sessions, sleeps):
    pair = FakePair(error=RuntimeError("the model server exited. Log tail:\nboom"))
    llm = _scripted([_pce("refused")])
    loop = _loop(llm, bus, server_recovery=lambda: pair)

    summary = await loop.run_turn("go")

    assert summary.startswith("Agent encountered an error: ProviderConnectionError: refused")
    assert "Log tail:\nboom" in summary
    assert sessions[0].terminated_reason == "error"
    assert [f.reason for f in _failed(bus)] == ["llm_error"]


async def test_a_failure_that_is_not_one_of_the_flights_own_is_named(bus, sessions, sleeps):
    pair = FakePair(error=ValueError("odd"))
    llm = _scripted([_pce("refused")])
    summary = await _loop(llm, bus, server_recovery=lambda: pair).run_turn("go")
    assert "ValueError: odd" in summary
    assert [f.reason for f in _failed(bus)] == ["llm_error"]


# ---------------------------------------------------------------------------
# Which pair
# ---------------------------------------------------------------------------


class _Observed:
    """Replaces server.wait_ready: records the address and key it was polled with, then returns."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, base_url: str, **kw: Any) -> None:
        self.calls.append({"base_url": base_url, **kw})


@pytest.fixture
def observed(monkeypatch):
    from localharness.provider import server

    seen = _Observed()
    monkeypatch.setattr(server, "wait_ready", seen)
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.02)
    return seen


async def test_a_pair_for_another_address_is_not_used(bus, observed):
    other = FakePair(base_url="http://other:1/v1")
    llm = _scripted([_pce(), _answer("ok")])
    loop = _loop(llm, bus, server_recovery=lambda: other)

    assert await loop.run_turn("go") == "ok"

    assert other.calls == []
    assert [c["base_url"] for c in observed.calls] == [URL]


async def test_the_loop_local_pair_is_kept_for_the_loops_life(bus, observed):
    other = FakePair(base_url="http://other:1/v1")
    llm = _scripted([_pce(), _answer("one"), _pce(), _answer("two")])
    loop = _loop(llm, bus, server_recovery=lambda: other)

    assert await loop.run_turn("a") == "one"
    first = loop._local_recovery
    assert first is not None
    assert await loop.run_turn("b") == "two"

    assert loop._local_recovery is first
    assert len(observed.calls) == 2  # a new turn is a new wait


async def test_a_supplier_returning_none_or_no_supplier_uses_the_local_pair(bus, observed):
    for kw in ({"server_recovery": lambda: None}, {}):
        loop = _loop(_scripted([_pce(), _answer("ok")]), bus, **kw)
        assert await loop.run_turn("go") == "ok"
        assert loop._local_recovery is not None
    assert len(observed.calls) == 2


async def test_a_pair_for_this_address_is_accepted_ignoring_a_trailing_slash(bus, sessions, sleeps):
    mine = FakePair(base_url=URL + "/")
    llm = _scripted([_pce(), _answer("ok")])
    loop = _loop(llm, bus, server_recovery=lambda: mine)
    assert await loop.run_turn("go") == "ok"
    assert len(mine.calls) == 1
    assert loop._local_recovery is None


# ---------------------------------------------------------------------------
# The bound and the signature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(True, 0.0), (-1.0, 0.0), ("30", 0.0), (0, 0.0), (30, 30.0), (0.001, 0.001), (45.5, 45.5)],
)
def test_the_bound_is_read_with_precision(bus, value, expected):
    llm = _scripted([_answer()], wait=value)
    assert _loop(llm, bus)._reconnect_bound() == expected


def test_a_magicmock_attribute_reads_as_off(bus):
    from unittest.mock import MagicMock

    llm = _scripted([_answer()])
    llm.config.reconnect_wait_seconds = MagicMock()
    assert _loop(llm, bus)._reconnect_bound() == 0.0


def test_a_missing_knob_or_a_non_string_address_reads_as_off(bus):
    assert _loop(_scripted([_answer()], wait=None), bus)._reconnect_bound() == 0.0
    from unittest.mock import MagicMock

    assert _loop(_scripted([_answer()], base_url=MagicMock()), bus)._reconnect_bound() == 0.0
    no_config = SimpleNamespace()
    assert _loop(no_config, bus)._reconnect_bound() == 0.0


def test_the_keyword_sits_directly_after_guardrails_path():
    names = list(inspect.signature(AgentLoop.__init__).parameters)
    assert names[names.index("guardrails_path") + 1] == "server_recovery"
    assert inspect.signature(AgentLoop.__init__).parameters["server_recovery"].default is None


# ---------------------------------------------------------------------------
# Through the real ServerRecovery: ProviderLost, the address and the key, endings, Esc
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import logging  # noqa: E402

import httpx  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from localharness.core.events import Observation, ProviderLost  # noqa: E402
from localharness.provider import lifecycle, server  # noqa: E402
from localharness.provider.lifecycle import ServerRecovery  # noqa: E402
from tests.unit.reconnect_support import (  # noqa: E402
    ModelsEndpoint,
    assert_no_key_leak,
    write_serve_log,
)

SENTINEL = "sk-sentinel-7d1e0b52-never-in-a-log"


@pytest.fixture
def models(monkeypatch):
    """Install a `/models` handler as the transport `wait_ready` uses; poll fast."""
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.02)

    def install(endpoint: ModelsEndpoint) -> ModelsEndpoint:
        monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(endpoint))
        return endpoint

    return install


def _real(llm: Any, bus: Any, pair: Any = None, **kw: Any) -> AgentLoop:
    pair = pair if pair is not None else ServerRecovery(None, None, None)
    return _loop(llm, bus, server_recovery=lambda: pair, **kw)


def _lost(bus) -> list[Any]:
    return bus.history(event_types=[ProviderLost])


async def test_one_death_publishes_one_provider_lost(bus, sessions, models):
    models(ModelsEndpoint(down_for=4))
    llm = _scripted([_pce("refused"), _answer("ok")], wait=5.0)

    assert await _real(llm, bus).run_turn("go") == "ok"

    lost = _lost(bus)
    assert len(lost) == 1
    assert lost[0].wait_seconds == 5.0
    assert lost[0].agent_id == sessions[0].agent_id
    assert lost[0].session_id == sessions[0].session_id


async def test_a_blip_publishes_nothing(bus, models):
    models(ModelsEndpoint(down_for=0))
    llm = _scripted([_pce("refused"), _answer("ok")], wait=5.0)
    assert await _real(llm, bus).run_turn("go") == "ok"
    assert _lost(bus) == []


async def test_the_detail_is_redacted_and_capped(bus, models):
    models(ModelsEndpoint(down_for=4))
    text = "unreachable at http://user:secret@127.0.0.1:9/tokenize\x07"
    llm = _scripted([_pce(text), _answer("ok")], wait=5.0)
    await _real(llm, bus).run_turn("go")
    detail = _lost(bus)[0].detail
    assert "http://127.0.0.1:9/tokenize" in detail
    assert "user:secret" not in detail
    assert "\x07" not in detail

    models(ModelsEndpoint(down_for=4))
    bus2 = type(bus)()
    llm = _scripted([_pce("z" * 1000), _answer("ok")], wait=5.0)
    await _real(llm, bus2).run_turn("go")
    assert 0 < len(_lost(bus2)[0].detail) <= 200


async def test_a_publish_failure_never_fails_the_wait(bus, models):
    models(ModelsEndpoint(down_for=4))
    llm = _scripted([_pce("refused"), _answer("ok")], wait=5.0)
    loop = _real(llm, bus)

    async def broken(event: Any) -> None:
        raise RuntimeError("bus is down")

    real_publish = bus.publish

    async def publish(event: Any) -> None:
        if isinstance(event, ProviderLost):
            await broken(event)
        return await real_publish(event)

    bus.publish = publish  # type: ignore[method-assign]
    assert await loop.run_turn("go") == "ok"


async def test_the_poll_uses_the_address_and_key_at_reconnect_time(bus, models, caplog):
    endpoint = models(ModelsEndpoint(down_for=2))
    llm = _scripted([_pce("refused"), _answer("ok")], wait=5.0)
    loop = _real(llm, bus)
    # what rebind_endpoint does after the loop was built
    llm.config.base_url = "http://127.0.0.1:7/v1"
    llm.config.api_key = SecretStr(SENTINEL)

    with caplog.at_level(logging.DEBUG):
        summary = await loop.run_turn("go")

    assert summary == "ok"
    assert endpoint.urls and set(endpoint.urls) == {"http://127.0.0.1:7/v1/models"}
    assert set(endpoint.auth) == {"Bearer " + SENTINEL}
    lost = _lost(bus)
    assert_no_key_leak(SENTINEL, summary, *[e.detail for e in lost], caplog=caplog)


async def test_a_pair_built_for_another_address_is_refused_for_the_key(bus, models):
    endpoint = models(ModelsEndpoint(down_for=1))
    llm = _scripted([_pce("refused"), _answer("ok")], wait=5.0)
    llm.config.api_key = SecretStr(SENTINEL)
    other = ServerRecovery("http://other:1/v1", None, None)

    assert await _real(llm, bus, pair=other).run_turn("go") == "ok"

    assert set(endpoint.urls) == {URL + "/models"}


async def test_the_bound_passing_ends_as_llm_error_with_the_sentence(bus, models, caplog):
    models(ModelsEndpoint(down=lambda: True))
    llm = _scripted([_pce("refused"), _answer("never reached")], wait=0.2)

    with caplog.at_level(logging.DEBUG):
        summary = await _real(llm, bus).run_turn("go")

    failed = _failed(bus)
    assert [f.reason for f in failed] == ["llm_error"]
    assert "did not come back within 0s" in failed[0].detail
    assert "did not come back within 0s" in summary
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_a_dead_server_with_no_relaunch_ends_at_once_with_its_log_tail(bus, models, tmp_path):
    from tests.unit.reconnect_support import write_dead_pidfile

    models(ModelsEndpoint(down=lambda: True))
    write_dead_pidfile(tmp_path)
    write_serve_log(tmp_path, "boom-tail\n")
    llm = _scripted([_pce("refused"), _answer("never reached")], wait=30.0)
    pair = ServerRecovery(None, tmp_path, None)

    summary = await asyncio.wait_for(_real(llm, bus, pair=pair).run_turn("go"), timeout=10)

    failed = _failed(bus)
    assert [f.reason for f in failed] == ["llm_error"]
    assert "the model server exited. Log tail:" in failed[0].detail
    assert "boom-tail" in summary
    assert not [f for f in failed if f.reason == "internal_error"]


async def test_the_note_is_stripped_of_control_characters_and_capped(bus, sessions, sleeps):
    note = "xxx\x1b[31m\nline2" + "y" * 5000
    pair = FakePair(error=RuntimeError(note))
    llm = _scripted([_pce("refused")])

    summary = await _loop(llm, bus, server_recovery=lambda: pair).run_turn("go")

    ending = summary.split("iterations. ", 1)[1]
    assert "\x1b" not in ending
    assert "\n" in ending
    assert len(ending) <= 1500
    assert ending.startswith("xxx[31m\nline2")


async def test_one_failed_reconnect_publishes_exactly_one_llm_error(bus, sessions, sleeps):
    pair = FakePair(error=TimeoutError("the model server did not come back within 30s"))
    llm = _scripted([_pce("refused")])
    await _loop(llm, bus, server_recovery=lambda: pair).run_turn("go")
    assert [f.reason for f in _failed(bus)] == ["llm_error"]


async def test_cancelling_the_turn_during_the_wait_propagates_and_stops_the_flight(bus, models):
    endpoint = models(ModelsEndpoint(down=lambda: True))
    llm = _scripted([_pce("refused"), _answer("never reached")], wait=30.0)
    loop = _real(llm, bus)

    task = asyncio.ensure_future(loop.run_turn("go"))
    for _ in range(200):
        if endpoint.calls:
            break
        await asyncio.sleep(0.01)
    assert endpoint.calls, "the wait never started polling"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _failed(bus) == []
    for _ in range(100):
        if not lifecycle._UNWINDING:
            break
        await asyncio.sleep(0.01)
    assert not lifecycle._UNWINDING
    settled = endpoint.calls
    await asyncio.sleep(0.2)
    assert endpoint.calls == settled


# ---------------------------------------------------------------------------
# Nothing appended, one tool run, a child's stamped event
# ---------------------------------------------------------------------------


async def test_a_failure_on_the_first_iteration_re_sends_the_same_messages(bus, sleeps):
    llm = _scripted([_pce("refused"), _answer("ok")], wait=None)
    assert await _loop(llm, bus).run_turn("go") == "ok"
    assert llm.sent[0] == llm.sent[1]
    assert len(llm.sent[0]) == len(llm.sent[1])


async def test_a_build_failure_on_the_first_iteration_appends_nothing(bus, sessions, sleeps):
    pair = FakePair()
    llm = _scripted([_answer("ok")])
    loop = _loop(llm, bus, server_recovery=lambda: pair)
    real = loop._ctx.build_messages
    lengths: list[int] = []

    async def build(messages, *a: Any, **k: Any):
        lengths.append(len(messages))
        if len(lengths) == 1:
            raise ProviderConnectionError(_UNREACHABLE)
        return await real(messages, *a, **k)

    loop._ctx.build_messages = build  # type: ignore[method-assign]
    assert await loop.run_turn("go") == "ok"
    assert len(lengths) == 2 and lengths[0] == lengths[1]
    assert len(pair.calls) == 1


async def test_a_tool_runs_once_across_a_drop_and_its_re_send(bus, sessions, sleeps):
    pair = FakePair()
    registry = _Registry()
    llm = _scripted([_pce("refused"), _tool_answer(), _answer("done")])
    loop = _tool_loop(llm, bus, registry, server_recovery=lambda: pair)

    assert await loop.run_turn("go") == "done"

    assert registry.ran == 1
    observed = [o for o in bus.history(event_types=[Observation]) if o.tool_call_id == "tc-1"]
    assert len(observed) == 1


async def test_a_child_loop_publishes_a_stamped_provider_lost(bus, models):
    from localharness.agent.subagent import _ParentIdBus

    models(ModelsEndpoint(down_for=4))
    llm = _scripted([_pce("refused"), _answer("ok")], wait=5.0)
    child = _real(llm, _ParentIdBus(bus, "parent-1"))

    assert await child.run_turn("go") == "ok"

    lost = _lost(bus)
    assert len(lost) == 1 and lost[0].parent_id == "parent-1"
