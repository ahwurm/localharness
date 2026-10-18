"""/effort on Strata: the level rides the request as `reasoning_effort`; no other runtime hears it."""
from __future__ import annotations

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, SelfCheckConfig
from tests.unit.test_client_effort_keys import make


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
