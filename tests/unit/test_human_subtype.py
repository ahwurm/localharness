"""The person's turns carry the typed or steering subtype and a turn number; every other caller is request."""
from types import SimpleNamespace

import pytest

from localharness.core.bus import EventBus
from localharness.core.types import human_message, provider_messages
from tests.unit.test_task_context import make_loop


def _humans(loop):
    return [m for m in loop.resume_state()[0] if m.get("_lh", {}).get("origin") == "human"]


@pytest.mark.asyncio
async def test_typed_and_default_turns_are_stamped_and_numbered(tmp_path, bus, mock_llm_client):
    llm = mock_llm_client([mock_llm_client.Response(content="one"), mock_llm_client.Response(content="two")])
    loop = make_loop(llm, bus, tmp_path)
    await loop.run_turn("hello", subtype="typed")
    await loop.run_turn("again")
    first, second = _humans(loop)
    assert first["_lh"] == {"origin": "human", "subtype": "typed", "turn": 1}
    assert second["_lh"] == {"origin": "human", "subtype": "request", "turn": 2}
    assert loop.turn == 2
    wire = provider_messages(loop.resume_state()[0])
    assert not any("_lh" in m for m in wire)
    assert [m["content"] for m in wire if m["role"] == "user"] == ["hello", "again"]


def test_human_message_stamps_turn_only_when_given():
    assert "turn" not in human_message("x")["_lh"]
    assert human_message("x", "typed", 3)["_lh"]["turn"] == 3


class _Agent:
    def __init__(self):
        self.calls = []
        self._config = SimpleNamespace(name="orchestrator", context=SimpleNamespace(max_image_tokens=0))
        self.current_session_id = "s1"

    async def run_turn(self, task, on_token=None, **kw):
        self.calls.append((task, kw))
        return "ok"


def _repl_with_agent():
    from localharness.cli.repl import OrchestratorREPL

    repl = OrchestratorREPL.__new__(OrchestratorREPL)
    repl._agent = _Agent()
    repl._channel = SimpleNamespace(channel_id="terminal")
    repl._bus = EventBus()
    repl._staged_images = []
    repl._skills = None
    repl._slash_followup = None
    repl._orchestrator = SimpleNamespace(active_workflow=None)
    repl._detect_creation_intent = lambda text: False
    return repl


async def _run(repl, coro):
    task = await coro
    await task
    return repl._agent.calls[-1]


@pytest.mark.asyncio
async def test_dispatch_input_runs_the_turn_typed_and_start_user_turn_defaults_to_request():
    repl = _repl_with_agent()
    assert await _run(repl, repl._dispatch_input("hello")) == ("hello", {"subtype": "typed"})
    assert await _run(repl, repl._start_user_turn("x")) == ("x", {"subtype": "request"})
