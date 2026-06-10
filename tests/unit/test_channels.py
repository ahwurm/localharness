

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
