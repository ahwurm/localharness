"""Scripted mock-LLM replies and a recorder of the messages each call saw."""


def scripted(mock_llm_client, *steps):
    R, C = mock_llm_client.Response, mock_llm_client.ToolCall
    return mock_llm_client([
        R(content=step) if isinstance(step, str)
        else R(content=None, tool_calls=[C(id=f"c{i}", name=step[0], arguments=step[1])])
        for i, step in enumerate(steps)
    ])


def capture(llm):
    seen = []
    original = llm.stream_complete

    async def wrapped(messages, *a, **kw):
        seen.append(list(messages))
        return await original(messages, *a, **kw)
    llm.stream_complete = wrapped
    return seen
