

import pytest


@pytest.mark.asyncio
async def test_child_task_complete_not_posted_to_channel():
    """Child-turn completions (parent_id set) must stay internal — their summaries
    return via the agent tool; posting them reads as a premature final answer."""
    from localharness.channels.base import ChannelAdapter
    from localharness.core.events import TaskComplete

    sent = []

    class _Chan(ChannelAdapter):
        async def read_input(self): return None
        async def send_message(self, content, agent_id=None, **kw): sent.append(content)
        async def send_error(self, error, detail=None, agent_id=None, **kw): pass
        async def send_streaming(self, *a, **kw): pass
        async def send_tool_call(self, *a, **kw): pass
        async def send_tool_result(self, *a, **kw): pass
        async def start(self): pass
        async def stop(self): pass

    chan = _Chan(bus=object(), config=object())
    await chan.on_task_complete(TaskComplete(
        agent_id="youtube-summarizer", session_id="s-child", parent_id="parent-sess",
        success=True, summary="child apology", duration_seconds=1.0, iterations=1))
    await chan.on_task_complete(TaskComplete(
        agent_id="default", session_id="s-parent", parent_id=None,
        success=True, summary="the real answer", duration_seconds=1.0, iterations=1))
    assert sent == ["the real answer"]


def _capture_chan():
    """Stub adapter capturing send_message / send_error calls."""
    from localharness.channels.base import ChannelAdapter

    sent, errors = [], []

    class _Chan(ChannelAdapter):
        async def read_input(self): return None
        async def send_message(self, content, agent_id=None, **kw): sent.append(content)
        async def send_error(self, error, detail=None, agent_id=None, **kw): errors.append((error, detail))
        async def send_streaming(self, *a, **kw): pass
        async def send_tool_call(self, *a, **kw): pass
        async def send_tool_result(self, *a, **kw): pass
        async def start(self): pass
        async def stop(self): pass

    return _Chan(bus=object(), config=object()), sent, errors


@pytest.mark.asyncio
async def test_root_turn_failed_posts_error_to_channel():
    """A failed root turn must surface as an error — silence is indistinguishable
    from progress (observed live: 3 llm_error deaths the user never saw)."""
    from localharness.core.events import TurnFailed

    chan, sent, errors = _capture_chan()
    await chan.on_turn_failed(TurnFailed(
        agent_id="default", session_id="s1", parent_id=None,
        reason="llm_error", detail="61441 input tokens exceed max 61440",
        iterations=6, duration_seconds=12.0))
    assert len(errors) == 1
    assert "llm_error" in errors[0][0]
    assert "61441" in (errors[0][1] or "")
    assert sent == []


@pytest.mark.asyncio
async def test_child_turn_failed_posts_status_note_not_error():
    """Child (delegated) failures get a one-line status note: the parent continues
    and still owes the real answer, so no fatal-looking error — but no silence either."""
    from localharness.core.events import TurnFailed

    chan, sent, errors = _capture_chan()
    await chan.on_turn_failed(TurnFailed(
        agent_id="frontend-designer", session_id="s-child", parent_id="parent-sess",
        reason="budget_exceeded", detail="ran out of time",
        iterations=32, duration_seconds=1200.0))
    assert errors == []
    assert len(sent) == 1
    assert "frontend-designer" in sent[0]
    assert "budget_exceeded" in sent[0]
