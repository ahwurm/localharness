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
