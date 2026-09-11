"""Bench pins `unattended` and can never block on a human (PRD §3.4, critic finding 7).

Two properties, and they are the reason the gate can ship without re-baselining every score:
the bench build path hands the loop a gate with NO asker, and in that mode a destructive shell
call — `rm`, `chmod`, the shapes real scenarios use — is still allowed, exactly as before v0.14.
The DENY tier is untouched, so a scenario that was denied before is denied now.
"""
from __future__ import annotations

import pytest

from localharness.agent.gate_types import ToolMeta
from localharness.bench.runner import BENCH_PERMISSION_MODE, _build_agent_loop
from localharness.bench.schema import BudgetSpec, LimitsSpec, ScenarioSpec, SuccessCriteria
from localharness.core.bus import EventBus
from tests.conftest import FakeLLMResponse, MockLLMClient


def _scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="gate-probe",
        slice="train",
        category="tool_basics",
        prompt="do nothing",
        success_criteria=SuccessCriteria(rubric=["contains:ok"]),
        budget=BudgetSpec(),
        limits=LimitsSpec(),
        tools_allowed=["bash_exec"],
    )


async def _bench_loop():
    return await _build_agent_loop(
        bus=EventBus(),
        llm_client=MockLLMClient([FakeLLMResponse(content="ok")]),
        scenario=_scenario(),
        session_id="bench-session",
    )


@pytest.mark.asyncio
async def test_the_bench_gate_has_no_asker_and_is_unattended():
    loop = await _bench_loop()
    assert loop.gate.mode == BENCH_PERMISSION_MODE == "unattended"
    assert loop.gate.asker is None, "a bench run must never be able to block on a human"


@pytest.mark.asyncio
async def test_a_destructive_shell_call_is_allowed_on_the_bench_path():
    """Today's behaviour, preserved: the shapes real scenarios use still run.

    Deliberately NOT `rm -rf`: that is a shipped DENY pattern and was already refused before
    v0.14. These are the calls the gate newly CLASSIFIES (ungrantable destructive, inline
    interpreter, unfamiliar) and which `unattended` must keep allowing.
    """
    loop = await _bench_loop()
    for command in (
        "chmod -R 755 out",
        "rm -r build",
        "python3 -c 'print(1)'",
        "cargo build --release",
    ):
        outcome = await loop.gate.check(
            "bash_exec",
            {"command": command},
            ToolMeta(group="shell"),
            agent_id="bench",
            session_id="bench-session",
            deny=loop._deny_fn,
        )
        assert outcome.allowed, f"{command!r} was refused on the bench path: {outcome.reason}"


@pytest.mark.asyncio
async def test_the_deny_tier_still_holds_on_the_bench_path():
    """`unattended` turns ASK into ALLOW — it does not touch DENY."""
    loop = await _bench_loop()
    outcome = await loop.gate.check(
        "bash_exec",
        {"command": "sudo rm -rf /"},
        ToolMeta(group="shell"),
        agent_id="bench",
        session_id="bench-session",
        deny=loop._deny_fn,
    )
    assert not outcome.allowed
