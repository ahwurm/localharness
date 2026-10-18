"""/effort on Strata: the level rides the request as `reasoning_effort`; no other runtime hears it."""
from __future__ import annotations

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, SelfCheckConfig
from localharness.provider.client import REASONING_CHARS_PER_TOKEN, ProviderReasoningBudgetError
from tests.unit.test_client_effort_keys import MODES, MSGS, THINK_OFF, Server, _bad, make

OTHERS = ("vllm", "llamacpp", "ollama", "lmstudio", "unknown")
LEVELS = {"off": "none", "low": "low", "high": "high", "max": "high"}


def _loop(llm, bus, effort: str) -> AgentLoop:
    return AgentLoop(
        config=AgentConfig(name="t", role="r", self_check=SelfCheckConfig(enabled=False)),
        llm=llm, bus=bus, context_manager=ContextManager(max_context_tokens=131_072),
        tool_registry=None, permission_evaluator=PermissionEvaluator(), effort=(effort, "session"),
    )


async def test_effort_low_through_the_loop_reaches_a_strata_request(bus):
    client, server = make("native", "strata")
    await _loop(client, bus, "low").run_turn("do the thing")
    assert server.calls[0]["extra_body"]["reasoning_effort"] == "low"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("level", LEVELS)
async def test_the_four_levels_map_onto_strata_reasoning_effort(mode, level):
    client, server = make(mode, "strata")
    await client.stream_complete(MSGS, effort=level)
    assert server.calls[0]["extra_body"] == {"reasoning_effort": LEVELS[level]}


async def test_off_sends_the_thinking_off_flag_beside_none():
    client, server = make("native", "strata")
    await client.stream_complete(MSGS, disable_thinking=True, effort="off")
    assert server.calls[0]["extra_body"] == {**THINK_OFF, "reasoning_effort": "none"}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ptype", OTHERS)
@pytest.mark.parametrize("level", LEVELS)
async def test_every_other_runtime_sends_the_request_it_sends_without_an_effort(mode, ptype, level):
    plain, plain_server = make(mode, ptype)
    await plain.stream_complete(MSGS, disable_thinking=True, reasoning_budget=1200)
    leveled, server = make(mode, ptype)
    await leveled.stream_complete(MSGS, disable_thinking=True, reasoning_budget=1200, effort=level)
    assert server.calls == plain_server.calls
    assert "reasoning_effort" not in str(server.calls)


async def test_a_strata_low_turn_with_a_budget_is_still_cut_client_side():
    cut = 10 * REASONING_CHARS_PER_TOKEN
    client, server = make("native", "strata", Server(reasoning=["x" * cut, "y"]))
    with pytest.raises(ProviderReasoningBudgetError):
        await client.stream_complete(MSGS, reasoning_budget=10, effort="low")
    body = server.calls[0]["extra_body"]
    assert body == {"reasoning_effort": "low"}


@pytest.mark.parametrize("mode", ("native", "xml"))
async def test_a_400_naming_reasoning_effort_is_remembered_and_omitted(mode):
    client, server = make(mode, "strata", Server(_bad("unknown field reasoning_effort"), "ok"))
    await client.stream_complete(MSGS, effort="high")
    assert server.calls[0]["extra_body"] == {"reasoning_effort": "high"}
    assert "extra_body" not in server.calls[1]
    assert client.rejected_effort_keys == frozenset({"reasoning_effort"})
    await client.stream_complete(MSGS, effort="high")
    assert "extra_body" not in server.calls[2]
