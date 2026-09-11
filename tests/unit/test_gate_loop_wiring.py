"""The gate is REACHABLE: every tool call of every agent goes through it (PRD §3.1).

A green test on `PermissionGate` in isolation proves nothing about the harness — the failure
this phase exists to remove is a gate nothing calls. So these tests drive the real `AgentLoop`
with the suite's fake LLM client and assert on what the MODEL saw: a denied observation it can
re-plan against, an approval that reaches the tool, and a grant that makes the second call
silent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.gate import GATE_ERROR_REASON, PermissionGate
from localharness.agent.gate_types import DEFAULT_MODE, Decision, PermissionRequest
from localharness.agent.loop import AgentLoop, Session
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.grants import GrantStore
from localharness.config.models import AgentConfig
from localharness.core.events import PermissionAsked, PermissionResolved
from localharness.tools import Tool, ToolRegistry, ToolResult, ToolSchema
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient


class _Shell(Tool):
    """Stands in for `bash_exec`: same name, same parameter, so the verdict classifies it the
    way it classifies the real one. Records every command it was actually asked to run."""

    timeout_s: float = 11.0

    def __init__(self) -> None:
        super().__init__()
        self.ran: list[str] = []

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="bash_exec",
            description="Run a shell command.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            group="shell",
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        self.ran.append(kwargs.get("command", ""))
        return self.ok("ran")


async def _registry(tool: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    await registry.register(tool, scope="global")
    return registry


def _gate(tmp_path: Path, workspace: Path, **kw) -> PermissionGate:
    """The gate every test in this file wires into the loop, started in GUARDED.

    ``mode`` is pinned rather than left to :data:`DEFAULT_MODE`: v0.14.1 moved the default to
    ``auto`` (owner ruling 2026-09-11), which asks about almost nothing and remembers nothing.
    These tests are about the GUARDED ask path being reachable from a real ``AgentLoop`` — that
    a question reaches a human, that the answer reaches the tool, that a grant makes the second
    call silent — so the mode that asks is the one they have to run in.
    """
    return PermissionGate(
        boundary=kw.pop("boundary", workspace),
        workspace=workspace,
        grants=kw.pop("grants", GrantStore(tmp_path / "grants.yaml")),
        channel_name="test",
        mode=kw.pop("mode", "guarded"),
        **kw,
    )


def _loop(bus, registry, gate=None, **kw) -> AgentLoop:
    return AgentLoop(
        config=AgentConfig(name="test-agent", role="Test agent."),
        llm=kw.pop("llm"),
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=registry,
        permission_evaluator=PermissionEvaluator(),
        gate=gate,
    )


def _plan(*commands: str) -> list[FakeLLMResponse]:
    """One tool call per turn step, then a final answer."""
    return [
        FakeLLMResponse(
            content=None,
            tool_calls=[FakeToolCall(id=f"tc-{i}", name="bash_exec", arguments={"command": c})],
        )
        for i, c in enumerate(commands)
    ] + [FakeLLMResponse(content="Done.")]


# --------------------------------------------------------------- ask reaches the loop

@pytest.mark.asyncio
async def test_ask_is_awaited_then_remembered_and_the_tool_runs(bus, tmp_path):
    """The end-to-end shape of the spine: ask once, run, never ask again."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    asked: list[PermissionRequest] = []

    async def asker(request: PermissionRequest) -> Decision:
        asked.append(request)
        return Decision(kind="allow_always")

    tool = _Shell()
    gate = _gate(tmp_path, workspace, asker=asker, bus=bus)
    registry = await _registry(tool)

    await _loop(bus, registry, gate, llm=MockLLMClient(_plan("cargo build"))).run_turn("t")
    assert len(asked) == 1 and tool.ran == ["cargo build"]

    await _loop(bus, registry, gate, llm=MockLLMClient(_plan("cargo build"))).run_turn("t")
    assert len(asked) == 1, "the grant was written but the second call still asked"
    assert tool.ran == ["cargo build", "cargo build"]

    assert [e.decision for e in bus.history(event_types=[PermissionResolved])] == ["allow_always"]
    assert len(bus.history(event_types=[PermissionAsked])) == 1


@pytest.mark.asyncio
async def test_the_asker_sees_the_tool_timeout(bus, tmp_path):
    """PRD §3.5: the wait derives from the tool's own timeout, so the loop must supply it."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    seen: list[float | None] = []

    class _Recording(PermissionGate):
        def _timeout_s(self, tool_timeout_s):
            seen.append(tool_timeout_s)
            return super()._timeout_s(tool_timeout_s)

    async def asker(request: PermissionRequest) -> Decision:
        return Decision(kind="allow_once")

    gate = _Recording(
        boundary=workspace, workspace=workspace, grants=GrantStore(tmp_path / "g.yaml"),
        asker=asker, channel_name="test", bus=bus, mode="guarded",
    )
    await _loop(
        bus, await _registry(_Shell()), gate, llm=MockLLMClient(_plan("cargo build"))
    ).run_turn("t")
    assert seen == [11.0]


# ------------------------------------------------------------------- fail closed

@pytest.mark.asyncio
async def test_no_asker_denies_and_the_model_is_told_why(bus, tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    tool = _Shell()
    gate = _gate(tmp_path, workspace, asker=None, bus=bus)
    loop = _loop(bus, await _registry(tool), gate, llm=MockLLMClient(_plan("cargo build")))
    await loop.run_turn("t")

    assert tool.ran == [], "a call nobody approved reached the tool"
    observations = [
        e for e in bus.history() if getattr(e, "observation_type", None) == "tool_result"
    ]
    assert observations and "cannot ask" in (observations[0].error or "")


@pytest.mark.asyncio
async def test_read_only_mode_text_reaches_the_model(bus, tmp_path):
    """PRD §3.4: a soft deny the model can re-plan against, not an error code."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    tool = _Shell()
    gate = _gate(tmp_path, workspace, mode="read-only", asker=None, bus=bus)
    await _loop(bus, await _registry(tool), gate, llm=MockLLMClient(_plan("cargo build"))).run_turn("t")

    assert tool.ran == []
    observations = [
        e for e in bus.history() if getattr(e, "observation_type", None) == "tool_result"
    ]
    assert observations and "not permitted in read-only mode" in (observations[0].error or "")


@pytest.mark.asyncio
async def test_a_loop_built_without_a_gate_still_has_one(bus, tmp_path, monkeypatch):
    """CONTRACTS A4: nothing runs ungated because a call site forgot."""
    monkeypatch.chdir(tmp_path)
    loop = _loop(bus, await _registry(_Shell()), None, llm=MockLLMClient(_plan("x")))
    assert loop.gate is not None
    assert loop.gate.asker is None, "a default gate must never be able to approve anything"
    assert loop.gate.mode == DEFAULT_MODE, "the fallback gate invented a mode of its own"


# --------------------------------------------------------------------- subagents

@pytest.mark.asyncio
async def test_subagent_dispatch_shares_the_parent_gate(bus, tmp_path):
    """PRD §3.4: "subagents inherit the parent session's channel and mode" — one object, so a
    /mode switch and a fresh grant reach a running child."""
    import inspect

    from localharness.agent import subagent

    workspace = tmp_path / "project"
    workspace.mkdir()
    gate = _gate(tmp_path, workspace, asker=None, bus=bus)

    captured: dict[str, Any] = {}

    # The dispatchers import AgentLoop lazily inside the function body, so patch the class in
    # its home module and record what the child was built with.
    import localharness.agent.loop as loop_module

    class _Capture(loop_module.AgentLoop):
        def __init__(self, *a, **kw):
            captured["gate"] = kw.get("gate")
            super().__init__(*a, **kw)

        async def run_turn(self, task, **kw):
            return "child done"

    monkey = pytest.MonkeyPatch()
    monkey.setattr(loop_module, "AgentLoop", _Capture)
    try:
        await subagent.dispatch_explore_subagent(
            "look",
            llm=MockLLMClient([FakeLLMResponse(content="done")]),
            bus=bus,
            base_registry=ToolRegistry(),
            parent_session_id="s",
            permission_evaluator=PermissionEvaluator(),
            gate=gate,
        )
    finally:
        monkey.undo()
    assert captured["gate"] is gate
    assert inspect.signature(subagent.dispatch_explore_subagent).parameters["gate"]


# ------------------------------------------------- one call, one prompt (defect D1)

@pytest.mark.asyncio
async def test_a_two_class_command_asks_once_and_runs_in_the_same_call(bus, tmp_path):
    """Verification A defect D1, at the level the human feels it.

    `mkdir -p <outside>/y && touch <outside>/y/f` raises four asks (two first-exposure commands,
    two directories outside the boundary). PRD §7 wants ONE question, the tool running in that
    same iteration, and the identical call never asking again.
    """
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = (tmp_path / "outside").resolve()
    command = f"mkdir -p {outside}/y ; touch {outside}/y/f"
    asked: list[PermissionRequest] = []

    async def asker(request: PermissionRequest) -> Decision:
        asked.append(request)
        return Decision(kind="allow_always")

    tool = _Shell()
    grants = GrantStore(tmp_path / "grants.yaml")
    gate = _gate(tmp_path, workspace, asker=asker, grants=grants, bus=bus)
    registry = await _registry(tool)

    await _loop(bus, registry, gate, llm=MockLLMClient(_plan(command))).run_turn("t")
    assert len(asked) == 1, f"one call must raise one question, got {[r.display for r in asked]}"
    assert tool.ran == [command], "the approved call must run in the same iteration"
    assert len(asked[0].grant_keys) >= 3
    assert {k for klass, k in asked[0].grant_keys if klass == "shell-unfamiliar"} == {"mkdir", "touch"}
    for klass, key in asked[0].grant_keys:
        assert grants.lookup(workspace, klass, key) is not None, f"no grant written for {klass} {key}"

    await _loop(bus, registry, gate, llm=MockLLMClient(_plan(command))).run_turn("t")
    assert len(asked) == 1, "the grants were written but the identical call asked again"
    assert tool.ran == [command, command]


# ------------------------------------ an unreadable tool schema asks, never allows (R4)

class _Reader(Tool):
    """A read-tier tool by its own schema — so if the schema is readable, it is ALLOWed."""

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="peek",
            description="Read something.",
            parameters={"type": "object", "properties": {}, "required": []},
            group="fs.read",
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return self.ok("peeked")


def _call_peek() -> list[FakeLLMResponse]:
    return [
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-0", name="peek", arguments={})]),
        FakeLLMResponse(content="Done."),
    ]


@pytest.mark.asyncio
async def test_a_readable_schema_keeps_a_read_tier_tool_silent(bus, tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    asked: list[PermissionRequest] = []

    async def asker(request: PermissionRequest) -> Decision:
        asked.append(request)
        return Decision(kind="allow_once")

    gate = _gate(tmp_path, workspace, asker=asker, bus=bus)
    await _loop(bus, await _registry(_Reader()), gate, llm=MockLLMClient(_call_peek())).run_turn("t")
    assert asked == []


@pytest.mark.asyncio
async def test_a_registry_whose_lookup_raises_makes_the_call_ask(bus, tmp_path, caplog):
    """R4: `_tool_facts` swallowed the exception and returned a NEUTRAL `ToolMeta`, which landed
    the call in the ALLOW tier — the one place a tool nobody can describe must not go.

    Same tool as the test above, same schema; only the lookup is broken, and the verdict flips
    from silent to asking.
    """
    import logging

    workspace = tmp_path / "project"
    workspace.mkdir()
    registry = await _registry(_Reader())

    def _boom(*args, **kwargs):
        raise RuntimeError("registry is confused")

    registry.lookup_tool = _boom  # type: ignore[method-assign]

    asked: list[PermissionRequest] = []

    async def asker(request: PermissionRequest) -> Decision:
        asked.append(request)
        return Decision(kind="allow_once")

    gate = _gate(tmp_path, workspace, asker=asker, bus=bus)
    with caplog.at_level(logging.WARNING):
        await _loop(bus, registry, gate, llm=MockLLMClient(_call_peek())).run_turn("t")

    assert len(asked) == 1, "an unreadable schema must reach a human, not the allow tier"
    assert (asked[0].klass, asked[0].key) == ("tool-unfamiliar", "peek")
    assert any("could not read the schema" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_gate_that_raises_produces_a_denied_observation(bus, tmp_path):
    """The gate is injected, so the loop cannot assume it never raises.

    A permission bug must cost one tool call, not the whole turn — so the call site denies with
    the same sentence the model can re-plan against.
    """
    workspace = tmp_path / "project"
    workspace.mkdir()

    class _Exploding(PermissionGate):
        async def check(self, *args, **kwargs):
            raise RuntimeError("boom")

    tool = _Shell()
    gate = _Exploding(
        boundary=workspace, workspace=workspace, grants=GrantStore(tmp_path / "g.yaml"),
        asker=None, channel_name="test", bus=bus,
    )
    summary = await _loop(
        bus, await _registry(tool), gate, llm=MockLLMClient(_plan("cargo build"))
    ).run_turn("t")

    assert tool.ran == [], "a call the gate could not decide reached the tool"
    observations = [
        e for e in bus.history() if getattr(e, "observation_type", None) == "tool_result"
    ]
    assert observations and GATE_ERROR_REASON in (observations[0].error or "")
    assert summary is not None, "the turn survived the permission bug"


@pytest.mark.asyncio
async def test_the_loop_hands_the_gate_the_tool_calls_own_id(bus, tmp_path):
    """`request.call_id` is only useful if the real loop supplies it (the ACP adapter pairs its
    dialog with a `tool_call` the client already knows — `_plan` numbers them `tc-0`, `tc-1`)."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    asked: list[PermissionRequest] = []

    async def asker(request: PermissionRequest) -> Decision:
        asked.append(request)
        return Decision(kind="allow_once")

    gate = _gate(tmp_path, workspace, asker=asker, bus=bus)
    await _loop(
        bus, await _registry(_Shell()), gate, llm=MockLLMClient(_plan("cargo build"))
    ).run_turn("t")

    assert [r.call_id for r in asked] == ["tc-0"]
    assert [r.agent_id for r in asked] == ["test-agent"]


# ------------------------------------------------------- a denied call is not an action

@pytest.mark.asyncio
async def test_a_denied_call_does_not_count_as_an_action_taken(bus, tmp_path):
    """`actions_taken` used to increment BEFORE the gate, so a call nobody approved spent the
    task's action budget and showed up in the "completed N tool calls" line."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    tool = _Shell()
    gate = _gate(tmp_path, workspace, asker=None, bus=bus)  # cannot ask → every ASK denies
    loop = _loop(bus, await _registry(tool), gate, llm=MockLLMClient(_plan("cargo build")))
    session = Session(agent_id="test-agent", session_id="s", messages=[])

    await loop._execute_loop(session, "t", None)
    assert tool.ran == []
    assert session.actions_taken == 0


@pytest.mark.asyncio
async def test_an_allowed_call_still_counts(bus, tmp_path):
    """The other half: the counter has to keep counting the calls that did run."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    tool = _Shell()

    async def asker(request: PermissionRequest) -> Decision:
        return Decision(kind="allow_once")

    gate = _gate(tmp_path, workspace, asker=asker, bus=bus)
    loop = _loop(bus, await _registry(tool), gate, llm=MockLLMClient(_plan("cargo build")))
    session = Session(agent_id="test-agent", session_id="s", messages=[])

    await loop._execute_loop(session, "t", None)
    assert tool.ran == ["cargo build"]
    assert session.actions_taken == 1


@pytest.mark.asyncio
async def test_step_excludes_denied_calls_from_tool_calls_executed(bus, tmp_path):
    """`step()` counted the denied branch as executed, so a step whose every call was refused
    reported the same number as one where every call ran."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    tool = _Shell()
    gate = _gate(tmp_path, workspace, asker=None, bus=bus)
    loop = _loop(bus, await _registry(tool), gate, llm=MockLLMClient(_plan("cargo build")))
    session = Session(agent_id="test-agent", session_id="s", messages=[])
    session.push({"role": "user", "content": "t"})

    result = await loop.step(session)
    assert result.action == "tool_calls"
    assert result.tool_calls_executed == 0
    assert session.actions_taken == 0
    assert tool.ran == []
