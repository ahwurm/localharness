

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


def _chunk_fr(content=None, tool_calls=None, finish_reason=None):
    """Like _chunk but carries choices[0].finish_reason (the field OpenAI-style streams set
    on the final content chunk: 'stop' | 'length' | 'tool_calls')."""
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(usage=None, choices=[NS(delta=delta, finish_reason=finish_reason)])


@pytest.mark.asyncio
async def test_stream_captures_finish_reason_length_midtoolcall():
    """#77: a completion cut at the output-token ceiling mid-tool-call carries
    finish_reason='length' on its final chunk. The assembled message MUST surface it so the
    loop can refuse to execute the truncated call. Native + xml share this one consumer, so
    capturing here covers both modes."""
    from localharness.provider.client import LLMClient

    chunks = [
        _chunk_fr(tool_calls=[NS(index=0, id="tc-a", function=NS(name="write", arguments='{"pa'))]),
        _chunk_fr(finish_reason="length"),
    ]
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.finish_reason == "length"
    assert msg.tool_calls is not None  # a (truncated) call was still assembled


@pytest.mark.asyncio
async def test_stream_captures_finish_reason_stop():
    """A clean completion surfaces finish_reason='stop' (guard must NOT fire)."""
    from localharness.provider.client import LLMClient

    chunks = [_chunk_fr(content="done"), _chunk_fr(finish_reason="stop")]
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_finish_reason_absent_is_none():
    """Providers that never emit finish_reason (older/partial mocks) yield None, not an
    error — the guard simply stays dormant."""
    from localharness.provider.client import LLMClient

    chunks = [_chunk(content="hi")]  # legacy chunk, no finish_reason attribute
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.finish_reason is None


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


# ---------------------------------------------------------------------------
# rebind_endpoint — cross-endpoint /model swap (0.10.0 model tree)
# ---------------------------------------------------------------------------


def test_rebind_endpoint_rebuilds_client_and_resets_sticky():
    """Re-pointing at a different server REBUILDS the AsyncOpenAI (base_url is baked at
    construction) and resets the per-server sticky `_tools_param_rejected` — the new server may
    accept the `tools` param the old one rejected."""
    from localharness.provider.client import LLMClient, LLMConfig

    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m", timeout_seconds=600))
    old = c._client
    c._tools_param_rejected = True  # a prior server rejected `tools`
    c.rebind_endpoint("http://127.0.0.1:11434/v1", api_key="k2", extra_headers={"X": "1"})
    assert c.config.base_url == "http://127.0.0.1:11434/v1"
    assert c.config.api_key == "k2"
    assert "11434" in str(c._client.base_url)   # rebuilt against the new server
    assert c._client is not old
    assert c._tools_param_rejected is False      # reset for the new server


@pytest.mark.asyncio
async def test_rebind_then_detect_refreshes_fn_converter():
    """After a rebind the caller MUST call detect_capabilities(), which re-derives the converter:
    non-None for an xml server, None for a native one — the single place mode + converter stay in
    lockstep (the SAME path the same-endpoint hot-swap already uses, so no separate fix needed)."""
    from localharness.provider.client import LLMClient, LLMConfig

    # is_local=False → the inference gate yields without a TCP probe (CPU-only, no network).
    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m",
                            timeout_seconds=600, is_local=False, tool_call_mode="native"))
    assert c._fn_converter is None
    c.rebind_endpoint("http://127.0.0.1:11434/v1")

    async def create_xml(**kw):
        return NS(choices=[NS(message=NS(tool_calls=None, content="<tool_call>x</tool_call>"))])

    async def models_empty():
        return NS(data=[])

    c._client = NS(chat=NS(completions=NS(create=create_xml)), models=NS(list=models_empty))
    cap = await c.detect_capabilities()
    assert cap.tool_call_mode == "xml"
    assert c._fn_converter is not None           # created for an xml/text server

    async def create_native(**kw):
        return NS(choices=[NS(message=NS(tool_calls=[NS(id="t")], content=None))])

    c._client = NS(chat=NS(completions=NS(create=create_native)), models=NS(list=models_empty))
    cap = await c.detect_capabilities()
    assert cap.tool_call_mode == "native"
    assert c._fn_converter is None               # cleared for a native server


def test_rebind_endpoint_failure_restores_prior_and_reraises(monkeypatch):
    """If the rebuild raises, the prior client + config are restored and the error re-raised, so a
    failed re-point never strands a half-configured client (mirrors TokenCounter.rebind #30)."""
    from localharness.provider.client import LLMClient, LLMConfig

    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m", timeout_seconds=600))
    old_client = c._client
    old_url = c.config.base_url

    def boom(self):
        raise RuntimeError("cannot build client")

    monkeypatch.setattr(LLMClient, "_build_client", boom)
    with pytest.raises(RuntimeError, match="cannot build client"):
        c.rebind_endpoint("http://127.0.0.1:11434/v1", api_key="k2")
    assert c.config.base_url == old_url          # restored (never left half-applied)
    assert c.config.api_key == "none"            # restored
    assert c._client is old_client               # restored


def test_rebind_to_empty_headers_clears_previous_headers():
    """#3: rebinding to an endpoint with {} headers after one with custom headers CLEARS them —
    'leave unchanged' is only for extra_headers=None. An endpoint's identity is exactly its OWN
    headers; the call site must pass {} (not `ep.extra_headers or None`, which would coerce the
    default {} to None and silently INHERIT the previous endpoint's headers)."""
    from localharness.provider.client import LLMClient, LLMConfig

    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m", timeout_seconds=600,
                            extra_headers={"X-Custom": "secret"}))
    assert "X-Custom" in c._client.default_headers          # baked into the first client
    c.rebind_endpoint("http://127.0.0.1:11434/v1", extra_headers={})
    assert c.config.extra_headers == {}                     # explicitly cleared, not left as-is
    assert "X-Custom" not in c._client.default_headers      # NOT carried over to the new server


def test_rebind_none_headers_leaves_previous_unchanged():
    """The contract the {} case contrasts with: extra_headers=None means LEAVE UNCHANGED (the only
    correct use of None), so an explicit None keeps the prior headers on the rebuilt client."""
    from localharness.provider.client import LLMClient, LLMConfig

    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m", timeout_seconds=600,
                            extra_headers={"X-Custom": "secret"}))
    c.rebind_endpoint("http://127.0.0.1:11434/v1", extra_headers=None)
    assert c.config.extra_headers == {"X-Custom": "secret"}  # unchanged
    assert "X-Custom" in c._client.default_headers
