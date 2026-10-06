"""run_turn(images=...) → the provider receives OpenAI content parts; the bookkeeping stays text."""
from __future__ import annotations

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, PermissionConfig
from localharness.core.content import image_part
from localharness.core.events import TurnStarted
from tests.conftest import FakeLLMResponse, MockLLMClient
from tests.unit.test_image_content import png_bytes


class _RecordingLLM(MockLLMClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.messages: list[list[dict]] = []

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kwargs):
        self.messages.append(list(messages or []))
        return await super().stream_complete(messages, tools, on_token, **kwargs)


@pytest.mark.asyncio
async def test_run_turn_sends_the_picture_as_parts_and_keeps_the_task_as_text(bus):
    llm = _RecordingLLM([FakeLLMResponse(content="The header overlaps the nav.")])
    loop = AgentLoop(
        config=AgentConfig(name="vision-agent", role="Reviews screenshots.",
                           permissions=PermissionConfig(mode="unattended")),
        llm=llm, bus=bus, context_manager=ContextManager(), tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
    )
    shot = image_part(png_bytes(640, 320), name="shot.png")
    summary = await loop.run_turn("What is wrong with this screen?", images=[shot])
    assert summary == "The header overlaps the nav."

    user_turns = [m for m in llm.messages[0] if m.get("role") == "user"]
    assert len(user_turns) == 1
    body = user_turns[0]["content"]
    assert body[0] == {"type": "text", "text": "What is wrong with this screen?"}
    assert body[1]["type"] == "image_url"
    assert body[1]["image_url"]["url"] == shot["image_url"]["url"]
    # The real client renders the wire shape (provider_messages) at the call; the fake sees the
    # canonical list, so the strip is asserted on the renderer here.
    from localharness.core.types import provider_messages
    assert "_lh" not in provider_messages(user_turns)[0]["content"][1]

    # The session keeps the private record; the event ledger saw only text.
    conversation, _ = loop.resume_state()
    kept = [m for m in conversation if m.get("role") == "user"][0]
    assert kept["content"][1]["_lh"]["width"] == 640
    assert bus.history(event_types=[TurnStarted])[0].task_summary == "What is wrong with this screen?"


@pytest.mark.asyncio
async def test_run_turn_without_images_is_the_plain_string_turn(bus):
    llm = _RecordingLLM([FakeLLMResponse(content="ok")])
    loop = AgentLoop(
        config=AgentConfig(name="plain-agent", role="x", permissions=PermissionConfig(mode="unattended")),
        llm=llm, bus=bus, context_manager=ContextManager(), tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
    )
    await loop.run_turn("hello")
    assert [m for m in llm.messages[0] if m["role"] == "user"][0]["content"] == "hello"
