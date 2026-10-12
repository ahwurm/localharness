"""A superseded failure leaves the model's view, never the record.

The agent loop stamps every tool message it pushes with a private outcome; the context manager
drops a failed-then-succeeded exchange from each request it builds (never a denial, never a
staged call, never a failure of a tool that can touch the host, never a message with other
content), and the session, the bus and ``history.jsonl`` keep both halves.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localharness.agent.context import ContextManager, _drop_superseded_failures
from localharness.agent.gate import DENIED_OBSERVATION_PREFIX
from localharness.agent.gate_types import GateOutcome
from localharness.agent.loop import (
    AgentLoop,
    Session,
    StuckDetector,
    _tool_message,
    compute_signature,
)
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig
from localharness.core.events import Observation
from localharness.core.types import is_harness_message, provider_messages
from localharness.memory.sqlite import MemoryStore
from localharness.tools import Tool, ToolRegistry, ToolResult, ToolSchema
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FlakyProbe(Tool):
    """A safe-host tool that fails on its first call and succeeds from then on."""

    def __init__(self, name: str = "probe", host: str = "safe") -> None:
        super().__init__()
        self._name = name
        self._host = host
        self.calls = 0

    def info(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description="Probe.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
            ingest="none", host=self._host,
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            return self.err("no such file")
        return self.ok("contents")


class _RecordingLLM(MockLLMClient):
    """MockLLMClient that keeps every message list it was asked to complete."""

    def __init__(self, responses: list[FakeLLMResponse]) -> None:
        super().__init__(responses)
        self.seen: list[list[dict[str, Any]]] = []

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kw):
        self.seen.append([dict(m) for m in (messages or [])])
        return await super().stream_complete(messages=messages, tools=tools, on_token=on_token, **kw)


def _loop(llm, bus, registry, *, ctx: ContextManager | None = None, budget: BudgetConfig | None = None,
          gate=None) -> AgentLoop:
    permissions = PermissionConfig(mode="unattended")
    if budget is not None:
        permissions = PermissionConfig(mode="unattended", budget=budget)
    return AgentLoop(
        config=AgentConfig(name="test-agent", role="Test agent.", permissions=permissions),
        llm=llm,
        bus=bus,
        context_manager=ctx or ContextManager(),
        tool_registry=registry,
        permission_evaluator=PermissionEvaluator(),
        gate=gate,
    )


async def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        await registry.register(tool, scope="global")
    return registry


def _tool_messages(loop: AgentLoop) -> list[dict[str, Any]]:
    conversation, _ = loop.resume_state()
    return [m for m in conversation if m.get("role") == "tool"]


def _outcome(call_name: str, args: dict, *, ok: bool, denied: bool = False, staged: bool = False) -> dict:
    return {"outcome": {"call": compute_signature(call_name, args), "ok": ok,
                        "denied": denied, "staged": staged}}


# ---------------------------------------------------------------------------
# Task 1: the stamp
# ---------------------------------------------------------------------------

def test_module_level_signature_matches_the_stuck_detector_method():
    sig = compute_signature("read", {"path": "a"})
    assert sig == StuckDetector().compute_signature("read", {"path": "a"})
    assert len(sig) == 16 and all(c in "0123456789abcdef" for c in sig)
    assert compute_signature("t", {"a": 1, "b": 2}) == compute_signature("t", {"b": 2, "a": 1})


@pytest.mark.asyncio
async def test_a_failed_then_successful_dispatch_is_stamped_with_the_call_signature(bus):
    args = {"path": "a"}
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments=args)]),
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-2", name="probe", arguments=args)]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()))
    await loop.run_turn("task")

    first, second = _tool_messages(loop)
    sig = compute_signature("probe", args)
    assert first["_lh"] == {"outcome": {"call": sig, "ok": False, "denied": False, "staged": False}}
    assert second["_lh"] == {"outcome": {"call": sig, "ok": True, "denied": False, "staged": False}}
    for message in (first, second):
        assert "origin" not in message["_lh"]
        assert "_lh" not in provider_messages([message])[0]
        assert is_harness_message(message) is False


@pytest.mark.asyncio
async def test_a_gate_denial_is_stamped_denied(bus):
    class _Deny:
        mode = "guarded"
        asker = None

        async def check(self, *a, **kw):
            return GateOutcome(allowed=False, reason="not today")

    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()), gate=_Deny())
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["content"].startswith(DENIED_OBSERVATION_PREFIX)
    assert msg["_lh"]["outcome"]["ok"] is False
    assert msg["_lh"]["outcome"]["denied"] is True
    assert msg["_lh"]["outcome"]["staged"] is False


@pytest.mark.asyncio
async def test_a_staged_refusal_is_stamped_denied_and_staged(bus):
    class _Stage:
        mode = "auto"
        asker = None

        async def check(self, *a, **kw):
            return GateOutcome(allowed=False, reason="needs a human (pending #1)", pending=object())  # type: ignore[arg-type]

    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()), gate=_Stage())
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["_lh"]["outcome"]["denied"] is True
    assert msg["_lh"]["outcome"]["staged"] is True


def test_the_helper_stamps_staged_and_leaves_an_unstampable_call_unstamped():
    call = SimpleNamespace(id="tc", name="probe", arguments={"path": "a"})
    staged = _tool_message(call, "x", ok=False, denied=True, staged=True)
    assert staged["_lh"]["outcome"] == {
        "call": compute_signature("probe", {"path": "a"}), "ok": False, "denied": True, "staged": True,
    }
    assert staged["role"] == "tool" and staged["tool_call_id"] == "tc" and staged["content"] == "x"

    class _Unhashable:
        pass

    broken = SimpleNamespace(id="tc", name="probe", arguments={"x": _Unhashable()})
    message = _tool_message(broken, "x", ok=True)
    assert "_lh" not in message and message["content"] == "x"


@pytest.mark.asyncio
async def test_truncated_call_feedback_is_stamped_failed(bus):
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, finish_reason="length",
                        tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()))
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["_lh"]["outcome"] == {
        "call": compute_signature("probe", {"path": "a"}), "ok": False, "denied": False, "staged": False,
    }


@pytest.mark.asyncio
async def test_tool_call_budget_refusal_is_stamped_failed_not_denied(bus):
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()), budget=BudgetConfig(max_tool_calls=0))
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["content"].startswith("Tool call not executed")
    assert msg["_lh"]["outcome"]["ok"] is False
    assert msg["_lh"]["outcome"]["denied"] is False
    assert msg["_lh"]["outcome"]["staged"] is False


@pytest.mark.asyncio
async def test_the_budget_note_lands_in_content_and_the_stamp_survives(bus):
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()), budget=BudgetConfig(max_actions=50))
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["_lh"]["outcome"]["call"] == compute_signature("probe", {"path": "a"})
    assert msg["_lh"]["outcome"]["ok"] is False  # the probe's first call failed


@pytest.mark.asyncio
async def test_step_path_stamps_denial_result_and_truncation(bus):
    # result
    llm = MockLLMClient([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()))
    session = Session(agent_id="test-agent", session_id="s", messages=[])
    session.push({"role": "user", "content": "t"})
    await loop.step(session)
    msg = [m for m in session.messages if m["role"] == "tool"][-1]
    assert msg["_lh"]["outcome"] == {
        "call": compute_signature("probe", {"path": "a"}), "ok": False, "denied": False, "staged": False,
    }

    # denial
    class _Deny:
        mode = "guarded"
        asker = None

        async def check(self, *a, **kw):
            return GateOutcome(allowed=False, reason="no")

    loop = _loop(MockLLMClient([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-2", name="probe", arguments={})]),
    ]), bus, await _registry(_FlakyProbe()), gate=_Deny())
    session = Session(agent_id="test-agent", session_id="s", messages=[])
    session.push({"role": "user", "content": "t"})
    await loop.step(session)
    msg = [m for m in session.messages if m["role"] == "tool"][-1]
    assert msg["_lh"]["outcome"]["denied"] is True and msg["_lh"]["outcome"]["ok"] is False

    # truncation
    loop = _loop(MockLLMClient([
        FakeLLMResponse(content=None, finish_reason="length",
                        tool_calls=[FakeToolCall(id="tc-3", name="probe", arguments={})]),
    ]), bus, await _registry(_FlakyProbe()))
    session = Session(agent_id="test-agent", session_id="s", messages=[])
    session.push({"role": "user", "content": "t"})
    await loop.step(session)
    msg = [m for m in session.messages if m["role"] == "tool"][-1]
    assert msg["_lh"]["outcome"]["ok"] is False and msg["_lh"]["outcome"]["denied"] is False


# ---------------------------------------------------------------------------
# Task 2: the request-view drop
# ---------------------------------------------------------------------------

SAFE = {"read": "safe", "agent": "safe"}


def _host(name: str) -> str:
    return SAFE.get(name, "dangerous")


def _call(tc_id: str, name: str, args: dict, *, as_object: bool = False) -> Any:
    if as_object:
        return SimpleNamespace(
            id=tc_id, type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
    return {"id": tc_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _assistant(tc_id: str, name: str, args: dict, *, content: str | None = None,
               as_object: bool = False) -> dict:
    return {"role": "assistant", "content": content,
            "tool_calls": [_call(tc_id, name, args, as_object=as_object)]}


def _tool(tc_id: str, name: str, args: dict, content: str, *, ok: bool, denied: bool = False,
          staged: bool = False, stamped: bool = True) -> dict:
    msg: dict[str, Any] = {"role": "tool", "tool_call_id": tc_id, "content": content}
    if stamped:
        msg["_lh"] = _outcome(name, args, ok=ok, denied=denied, staged=staged)
    return msg


A = {"path": "a"}
B = {"path": "b"}
USER = {"role": "user", "content": "go"}


def _fail_then_ok(name: str = "read", first_args: dict = A, second_args: dict = A, **first_kw) -> list[dict]:
    first_kw.setdefault("ok", False)
    return [
        USER,
        _assistant("c1", name, first_args),
        _tool("c1", name, first_args, first_kw.pop("content", "[tool error] nope"), **first_kw),
        _assistant("c2", name, second_args),
        _tool("c2", name, second_args, "fine", ok=True),
    ]


def test_a_superseded_failure_is_dropped_and_the_input_is_untouched():
    msgs = _fail_then_ok()
    before = list(msgs)
    out = _drop_superseded_failures(msgs, _host)
    assert out == [msgs[0], msgs[3], msgs[4]]
    assert msgs == before and all(a is b for a, b in zip(msgs, before))
    assert out is not msgs


def test_tool_calls_given_as_objects_match_like_dicts():
    msgs = [
        USER,
        _assistant("c1", "read", A, as_object=True),
        _tool("c1", "read", A, "[tool error] nope", ok=False),
        _assistant("c2", "read", A, as_object=True),
        _tool("c2", "read", A, "fine", ok=True),
    ]
    assert _drop_superseded_failures(msgs, _host) == [msgs[0], msgs[3], msgs[4]]


def _kept(msgs: list[dict], host=_host) -> None:
    out = _drop_superseded_failures(msgs, host)
    assert out == msgs


def test_a_denial_is_kept():
    _kept(_fail_then_ok(denied=True, content=f"{DENIED_OBSERVATION_PREFIX}no"))


def test_a_staged_call_is_kept():
    _kept(_fail_then_ok(denied=True, staged=True, content=f"{DENIED_OBSERVATION_PREFIX}pending #1"))


def test_a_failure_of_a_dangerous_host_tool_is_kept():
    _kept(_fail_then_ok(name="bash_exec", first_args={"command": "x"}, second_args={"command": "x"}))


def test_an_assistant_message_with_other_content_is_kept():
    msgs = _fail_then_ok()
    msgs[1] = _assistant("c1", "read", A, content="Let me try.")
    _kept(msgs)


def test_an_assistant_message_with_two_calls_is_kept():
    msgs = _fail_then_ok()
    msgs[1]["tool_calls"].append(_call("c1b", "read", B))
    msgs.insert(3, _tool("c1b", "read", B, "fine", ok=True))
    _kept(msgs)


def test_a_different_later_call_does_not_supersede():
    _kept(_fail_then_ok(second_args=B))


def test_an_unstamped_failure_is_kept():
    _kept(_fail_then_ok(stamped=False))


def test_a_success_before_the_failure_is_kept():
    msgs = [
        USER,
        _assistant("c0", "read", A), _tool("c0", "read", A, "fine", ok=True),
        _assistant("c1", "read", A), _tool("c1", "read", A, "[tool error] nope", ok=False),
    ]
    _kept(msgs)


def test_a_stamped_non_denied_failure_with_the_denial_prefix_is_kept():
    _kept(_fail_then_ok(content=f"{DENIED_OBSERVATION_PREFIX}looks like one"))


def test_unwired_and_raising_hosts_drop_nothing():
    msgs = _fail_then_ok()
    assert _drop_superseded_failures(msgs, None) == msgs

    def boom(_name: str) -> str:
        raise RuntimeError("lookup broke")

    assert _drop_superseded_failures(msgs, boom) == msgs


def test_pairs_are_matched_by_id_not_adjacency():
    msgs = [
        USER,
        _assistant("c1", "read", A),
        {"role": "user", "content": "interleaved"},
        _tool("c1", "read", A, "[tool error] nope", ok=False),
        _assistant("c2", "read", A),
        _tool("c2", "read", A, "fine", ok=True),
    ]
    out = _drop_superseded_failures(msgs, _host)
    assert [m.get("tool_call_id") for m in out if m["role"] == "tool"] == ["c2"]
    assert all(not (m["role"] == "assistant" and m["tool_calls"][0]["id"] == "c1") for m in out)


@pytest.mark.asyncio
async def test_two_builds_are_byte_identical_and_the_failed_pair_stays_in_the_record():
    ctx = ContextManager(host=_host)
    msgs = _fail_then_ok()
    first, _ = await ctx.build_messages(msgs, None)
    second, _ = await ctx.build_messages(msgs, None)

    def wire(out):
        return json.dumps(provider_messages(out), sort_keys=True)

    assert wire(first) == wire(second)
    assert "c1" not in wire(first) and "c2" in wire(first)
    assert all("_lh" not in m for m in provider_messages(first))
    assert any(m.get("tool_call_id") == "c1" for m in msgs), "the caller's list lost the failed pair"
    assert len(msgs) == 5
    assert ctx.repair_tool_pairing(first) == first


@pytest.mark.asyncio
async def test_an_unwired_manager_drops_nothing():
    msgs = _fail_then_ok()
    out, _ = await ContextManager().build_messages(msgs, None)
    assert any(m.get("tool_call_id") == "c1" for m in out)


@pytest.mark.asyncio
async def test_the_registry_reports_the_declared_host_and_fails_closed():
    registry = await _registry(_FlakyProbe(name="read", host="safe"), _FlakyProbe(name="danger", host="dangerous"))
    assert registry.host("read") == "safe"
    assert registry.host("danger") == "dangerous"
    assert ToolRegistry().host("nope") == "dangerous"


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="obs-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_end_to_end_the_third_request_omits_the_failed_exchange_but_the_record_keeps_it(bus, store):
    registry = await _registry(_FlakyProbe())
    args = {"path": "a"}
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments=args)]),
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-2", name="probe", arguments=args)]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, registry, ctx=ContextManager(host=registry.host))
    await loop.run_turn("task")

    assert len(llm.seen) == 3
    third = llm.seen[2]
    third_ids = {m.get("tool_call_id") for m in third if m.get("role") == "tool"}
    assert third_ids == {"tc-2"}
    call_ids = {tc["id"] if isinstance(tc, dict) else tc.id
                for m in third for tc in (m.get("tool_calls") or [])}
    assert call_ids == {"tc-2"}
    # The second request still carried the failure: it had not yet been superseded.
    assert {m.get("tool_call_id") for m in llm.seen[1] if m.get("role") == "tool"} == {"tc-1"}

    kept = {m.get("tool_call_id") for m in _tool_messages(loop)}
    assert kept == {"tc-1", "tc-2"}
    conversation, _ = loop.resume_state()
    assert sum(1 for m in conversation if m.get("role") == "assistant" and m.get("tool_calls")) == 2

    observations = [e for e in bus.history(event_types=[Observation]) if e.observation_type == "tool_result"]
    assert [o.tool_call_id for o in observations] == ["tc-1", "tc-2"]
    for o in observations:
        await store._on_observation(o)
    rows = [json.loads(line) for line in Path(store._history_path).read_text().splitlines() if line.strip()]
    assert [r["call_id"] for r in rows if r["type"] == "tool_result"] == ["tc-1", "tc-2"]


@pytest.mark.asyncio
async def test_start_passes_the_registry_host_to_the_root_manager(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _stub_start_boundaries

    _stub_start_boundaries(tmp_path, monkeypatch)
    seen: list[dict] = []
    real_init = ContextManager.__init__

    def spy_init(self, *a, **k):
        seen.append(k)
        return real_init(self, *a, **k)

    monkeypatch.setattr(ContextManager, "__init__", spy_init)
    await _start_async(None, False, False, str(tmp_path))

    wired = [k for k in seen if k.get("bus") is not None]
    assert wired
    for k in wired:
        host = k.get("host")
        assert host is not None and getattr(host, "__func__", None) is ToolRegistry.host
        assert host.__self__ is k["result_origin"].__self__, "host and result_origin read different registries"


@pytest.mark.asyncio
async def test_bench_manager_resolves_host_through_its_per_scenario_registry(monkeypatch):
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import LimitsSpec, ScenarioSpec, SuccessCriteria
    from localharness.core.events import BudgetSpec

    captured: dict = {}

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", FakeAgentLoop)
    scen = ScenarioSpec(
        name="t", prompt="x", success_criteria=SuccessCriteria(rubric=["contains:OK"]),
        budget=BudgetSpec(), limits=LimitsSpec(), tools_allowed=["agent"], slice="train",
        category="tool_basics",
    )
    await bench_runner._build_agent_loop(bus=None, llm_client=None, scenario=scen)
    registry = captured["tool_registry"]
    manager = captured["context_manager"]
    assert registry.host("agent") == "safe"
    assert manager._host("agent") == registry.host("agent")
    assert manager._host("not-a-tool") == "dangerous"
