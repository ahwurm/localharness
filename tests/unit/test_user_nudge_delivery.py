"""User nudges from the type-anytime input box ride the #82 session-persisted seam:
push_user_nudge() queues a user-role message that the running turn drains at its NEXT
step boundary (top of the ReAct loop), exactly like the stuck-recovery nudge — but from a
distinct source that NEVER touches the stuck detector's max-nudges-per-turn accounting.
"""
from __future__ import annotations

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, SelfCheckConfig
from localharness.core.events import StuckRecovered, TurnStarted


def _make_loop(mock_llm_client, responses, bus, config=None):
    cfg = config or AgentConfig(name="test-agent", role="Test agent.")
    return AgentLoop(
        config=cfg,
        llm=mock_llm_client(responses),
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
    )


class TestPushUserNudge:
    def test_push_appends(self, mock_llm_client, bus):
        loop = _make_loop(mock_llm_client, [mock_llm_client.Response(content="ok")], bus)
        loop.push_user_nudge("hey do X")
        assert loop._user_nudge_inbox == ["hey do X"]

    def test_push_ignores_blank(self, mock_llm_client, bus):
        loop = _make_loop(mock_llm_client, [mock_llm_client.Response(content="ok")], bus)
        loop.push_user_nudge("   ")
        loop.push_user_nudge("")
        assert loop._user_nudge_inbox == []


class TestDelivery:
    async def test_preloaded_nudge_delivered_this_turn(self, mock_llm_client, bus):
        Response = mock_llm_client.Response
        loop = _make_loop(mock_llm_client, [Response(content="All done.")], bus)
        loop.push_user_nudge("also verify the config file")
        await loop.run_turn("summarize the repo")

        convo = loop._conversation
        users = [m["content"] for m in convo if m.get("role") == "user"]
        assert users[0] == "summarize the repo"
        assert "also verify the config file" in users

    async def test_multiple_nudges_delivered_in_order(self, mock_llm_client, bus):
        Response = mock_llm_client.Response
        loop = _make_loop(mock_llm_client, [Response(content="done")], bus)
        loop.push_user_nudge("first nudge")
        loop.push_user_nudge("second nudge")
        await loop.run_turn("go")

        users = [m["content"] for m in loop._conversation if m.get("role") == "user"]
        assert users.index("first nudge") < users.index("second nudge")

    async def test_nudge_pushed_midturn_lands_at_next_step_boundary(self, mock_llm_client, bus):
        # self_check forces a 2nd iteration with no tools: iter-1 draft -> review -> iter-2.
        Response = mock_llm_client.Response
        cfg = AgentConfig(
            name="test-agent", role="Test.",
            self_check=SelfCheckConfig(enabled=True, max_passes=1),
        )
        loop = _make_loop(
            mock_llm_client,
            [Response(content="draft answer"), Response(content="CONFIRMED")],
            bus, config=cfg,
        )
        orig = loop._llm.stream_complete
        seen = {"n": 0, "call2": None}

        async def wrapper(messages=None, tools=None, on_token=None):
            seen["n"] += 1
            if seen["n"] == 1:
                # user types during iteration 1 (after this iteration's boundary drain)
                loop.push_user_nudge("stop and re-read the spec first")
            if seen["n"] == 2:
                seen["call2"] = list(messages or [])
            return await orig(messages=messages, tools=tools, on_token=on_token)

        loop._llm.stream_complete = wrapper
        await loop.run_turn("do the thing")

        assert seen["n"] == 2, "self_check should force a 2nd iteration"
        contents = [m.get("content") for m in seen["call2"]]
        assert "stop and re-read the spec first" in contents, (
            "nudge must be drained into iteration-2's request at the step boundary"
        )

    async def test_nudges_do_not_touch_stuck_accounting(self, mock_llm_client, bus):
        # Many nudges, no repeated tool calls -> the stuck detector must never fire.
        Response = mock_llm_client.Response
        loop = _make_loop(mock_llm_client, [Response(content="done")], bus)
        for i in range(10):
            loop.push_user_nudge(f"nudge {i}")
        await loop.run_turn("go")

        assert bus.history(event_types=[StuckRecovered]) == []
        assert len(bus.history(event_types=[TurnStarted])) == 1
