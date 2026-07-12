

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


# ---------------------------------------------------------------------------
# #18 — XML tool-call mode must stream at the transport level. `_complete_xml` /
# `_complete_xml_fallback` accepted a `stream` parameter and IGNORED it, so any
# model whose capability probe falls back to XML mode silently issued whole-
# response requests for the entire agent loop (read timeout races the whole
# generation; a cancel leaves vLLM decoding into the void). No log signal.
# ---------------------------------------------------------------------------

from localharness.provider.client import LLMConfig


def _xml_cfg() -> LLMConfig:
    # is_local=False: skip the inference gate + the local-timeout floor; the subject
    # here is whether `stream=True` reaches the transport, not gating.
    return LLMConfig(base_url="http://127.0.0.1:9/v1", model="m", timeout_seconds=300.0,
                     tool_call_mode="xml", is_local=False)


class _StreamOrResp:
    """A create() return that behaves as BOTH a whole-response object (.choices/.usage)
    AND an async chunk stream — so the RED failure is a clean `stream=True` assertion,
    never an AttributeError from whichever branch the code happens to take."""

    def __init__(self, content: str):
        self._content = content
        self.choices = [NS(message=NS(content=content, tool_calls=None))]
        self.usage = None

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        yield NS(usage=None, choices=[NS(delta=NS(content=self._content, tool_calls=None))])


@pytest.mark.asyncio
async def test_stream_complete_xml_mode_passes_stream_true():
    """RED: stream_complete() in XML mode must request transport streaming. The dead
    `stream` param made XML mode silently non-streaming for the whole loop (#18)."""
    from localharness.provider.client import LLMClient

    client = LLMClient(_xml_cfg())
    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)
        return _StreamOrResp("<tool_call>{}</tool_call>")

    client._client = NS(chat=NS(completions=NS(create=fake_create)))
    msg, _usage = await client.stream_complete([{"role": "user", "content": "hi"}])
    assert captured and captured[0].get("stream") is True
    # Full text is buffered client-side BEFORE the XML parse still sees it.
    assert msg.content == "<tool_call>{}</tool_call>"


@pytest.mark.asyncio
async def test_complete_xml_fallback_streams_when_requested():
    """RED: the system-prompt-injection fallback honors stream too — a BadRequestError
    re-entry must not silently drop back to a whole-response request (#18)."""
    from localharness.provider.client import LLMClient

    client = LLMClient(_xml_cfg())
    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)
        return _StreamOrResp("ok")

    client._client = NS(chat=NS(completions=NS(create=fake_create)))
    msg, _usage = await client._complete_xml_fallback(
        [{"role": "user", "content": "hi"}], None, stream=True
    )
    assert captured and captured[0].get("stream") is True
    assert msg.content == "ok"
