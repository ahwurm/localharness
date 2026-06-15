

def test_local_client_disables_sdk_auto_retries():
    """The SDK's silent default (2 retries) turns one timed-out local generation into
    3x the wait — observed live as 30 min of dead air (3 x 600s). Local endpoints
    must fail fast; remote keeps the default."""
    from localharness.provider.client import LLMClient, LLMConfig

    local = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m",
                                timeout_seconds=600))
    assert local._client.max_retries == 0

    remote = LLMClient(LLMConfig(base_url="https://api.example.com/v1", model="m",
                                 timeout_seconds=120, is_local=False))
    assert remote._client.max_retries == 2


# ---------------------------------------------------------------------------
# True-streaming chunk assembly (_consume_native_stream)
# ---------------------------------------------------------------------------

import pytest
from types import SimpleNamespace as NS


def _chunk(content=None, tool_calls=None, usage=None):
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(usage=usage, choices=[NS(delta=delta)] if (content or tool_calls) or usage is None else [])


async def _aiter(chunks):
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_stream_assembles_content_and_calls_on_token():
    from localharness.provider.client import LLMClient

    seen = []
    async def on_token(piece): seen.append(piece)

    chunks = [_chunk(content="Hel"), _chunk(content="lo "), _chunk(content="world")]
    msg, usage = await LLMClient._consume_native_stream(_aiter(chunks), on_token)
    assert msg.content == "Hello world"
    assert seen == ["Hel", "lo ", "world"]
    assert msg.tool_calls is None
    assert usage is None


@pytest.mark.asyncio
async def test_stream_assembles_fragmented_tool_calls():
    from localharness.provider.client import LLMClient

    chunks = [
        _chunk(tool_calls=[NS(index=0, id="tc-a", function=NS(name="web_search", arguments=""))]),
        _chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments='{"que'))]),
        _chunk(tool_calls=[NS(index=1, id="tc-b", function=NS(name="agent", arguments='{"agent_id"'))]),
        _chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments='ry": "x"}'))]),
        _chunk(tool_calls=[NS(index=1, id=None, function=NS(name=None, arguments=': "explore"}'))]),
    ]
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.content is None
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0] == {"id": "tc-a", "type": "function",
                                 "function": {"name": "web_search", "arguments": '{"query": "x"}'}}
    assert msg.tool_calls[1]["function"]["arguments"] == '{"agent_id": "explore"}'


@pytest.mark.asyncio
async def test_stream_captures_final_usage_chunk():
    from localharness.provider.client import LLMClient

    final = NS(usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15), choices=[])
    chunks = [_chunk(content="ok"), final]
    msg, usage = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.content == "ok"
    assert usage.completion_tokens == 5
