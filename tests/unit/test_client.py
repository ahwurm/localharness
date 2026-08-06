"""Tests for LLMClient (client.py)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from localharness.provider.client import (
    CapabilityResult,
    LLMClient,
    LLMConfig,
    MalformedResponseError,
    ProviderAPIError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from localharness.provider.fn_call import FnCallConverter


# ---------------------------------------------------------------------------
# LLMConfig defaults
# ---------------------------------------------------------------------------


def test_llm_config_defaults():
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    # #10: default synced to 600s — a 4096-token completion at ~10 tok/s single-stream is
    # ~410s, which the old 300s default killed mid-generation (e.g. bench/orchestrator.py).
    assert config.timeout_seconds == 600.0
    assert config.temperature == 0.6
    assert config.max_tokens == 4096
    assert config.tool_call_mode == "native"
    assert config.api_key == "none"
    assert config.connect_timeout_seconds == 5.0
    from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS
    assert config.context_window == DEFAULT_MAX_CONTEXT_TOKENS  # 131_072 (served truth)
    assert config.is_local is True
    assert config.extra_headers == {}
    assert config.stop_sequences == []


def test_default_timeout_seconds_synced_to_600():
    """#10: the module-level DEFAULT_TIMEOUT_SECONDS must match the LLMConfig/ProviderConfig
    default (600s) so the documented slow-decode math holds everywhere it is referenced."""
    from localharness.config.defaults import DEFAULT_TIMEOUT_SECONDS
    assert DEFAULT_TIMEOUT_SECONDS == 600.0


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------


def test_local_timeout_enforcement():
    """LLMClient with local=True and timeout < 300 raises ValueError containing '300'."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test",
        is_local=True,
        timeout_seconds=60.0,
    )
    with pytest.raises(ValueError, match="300"):
        LLMClient(config)


def test_local_timeout_valid():
    """LLMClient with local=True and timeout >= 300 does NOT raise."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test",
        is_local=True,
        timeout_seconds=300.0,
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        client = LLMClient(config)
    assert client is not None


def test_remote_no_timeout_enforcement():
    """LLMClient with is_local=False and short timeout does NOT raise."""
    config = LLMConfig(
        base_url="http://api.openai.com/v1",
        model="gpt-4",
        is_local=False,
        timeout_seconds=60.0,
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        client = LLMClient(config)
    assert client is not None


# ---------------------------------------------------------------------------
# FnCallConverter instantiation
# ---------------------------------------------------------------------------


def test_fn_converter_set_for_xml():
    """tool_call_mode='xml' -> _fn_converter is FnCallConverter."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test",
        is_local=True,
        timeout_seconds=300.0,
        tool_call_mode="xml",
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        client = LLMClient(config)
    assert isinstance(client._fn_converter, FnCallConverter)


def test_fn_converter_set_for_text():
    """tool_call_mode='text' -> _fn_converter is FnCallConverter."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test",
        is_local=True,
        timeout_seconds=300.0,
        tool_call_mode="text",
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        client = LLMClient(config)
    assert isinstance(client._fn_converter, FnCallConverter)


def test_fn_converter_none_for_native():
    """tool_call_mode='native' -> _fn_converter is None."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test",
        is_local=True,
        timeout_seconds=300.0,
        tool_call_mode="native",
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        client = LLMClient(config)
    assert client._fn_converter is None


# ---------------------------------------------------------------------------
# detect_capabilities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_capabilities_native():
    """Response with non-empty tool_calls -> CapabilityResult(tool_call_mode='native')."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test",
        is_local=True,
        timeout_seconds=300.0,
    )

    mock_tool_call = MagicMock()
    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_openai = MagicMock()
    mock_openai.chat = MagicMock()
    mock_openai.chat.completions = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)

    # Mock /v1/models response
    mock_model_entry = MagicMock()
    mock_model_entry.id = "test"
    # no context_length attribute
    mock_models_response = MagicMock()
    mock_models_response.data = [mock_model_entry]
    mock_openai.models = MagicMock()
    mock_openai.models.list = AsyncMock(return_value=mock_models_response)

    with patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai):
        client = LLMClient(config)
        result = await client.detect_capabilities()

    assert result.tool_call_mode == "native"
    assert result.probe_error is None


@pytest.mark.asyncio
async def test_detect_capabilities_xml():
    """Response content with <tool_call> tag -> CapabilityResult(tool_call_mode='xml')."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test",
        is_local=True,
        timeout_seconds=300.0,
    )

    mock_message = MagicMock()
    mock_message.tool_calls = []
    mock_message.content = "<tool_call><name>list_files</name></tool_call>"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_openai = MagicMock()
    mock_openai.chat = MagicMock()
    mock_openai.chat.completions = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)
    mock_openai.models = MagicMock()
    mock_openai.models.list = AsyncMock(return_value=MagicMock(data=[]))

    with patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai):
        client = LLMClient(config)
        result = await client.detect_capabilities()

    assert result.tool_call_mode == "xml"


@pytest.mark.asyncio
async def test_detect_capabilities_fallback():
    """HTTP 400 response -> CapabilityResult(tool_call_mode='xml', probe_error set)."""
    import openai as openai_module

    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test",
        is_local=True,
        timeout_seconds=300.0,
    )

    mock_openai = MagicMock()
    mock_openai.chat = MagicMock()
    mock_openai.chat.completions = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=openai_module.BadRequestError(
            message="tools not supported",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
    )
    mock_openai.models = MagicMock()
    mock_openai.models.list = AsyncMock(return_value=MagicMock(data=[]))

    with patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai):
        client = LLMClient(config)
        result = await client.detect_capabilities()

    assert result.tool_call_mode == "xml"
    assert result.probe_error is not None


def _probe_client(create_side_effect):
    """LLMClient whose probe generation is driven by `create_side_effect`, /v1/models empty."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1", model="test", is_local=True, timeout_seconds=300.0
    )
    mock_openai = MagicMock()
    mock_openai.chat = MagicMock()
    mock_openai.chat.completions = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    mock_openai.models = MagicMock()
    mock_openai.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return config, mock_openai


def _native_completion():
    msg = MagicMock()
    msg.tool_calls = [MagicMock()]
    msg.content = None
    return MagicMock(choices=[MagicMock(message=msg)])


@pytest.mark.asyncio
async def test_probe_retries_transient_failure_then_confirms_native():
    """THE BUG: one blip condemned the whole session to xml.

    A busy slot / slow first token / connection blip on the single probe request downgraded
    every later turn, and a model whose native syntax is not <tool_call> (DeepSeek emits DSML)
    then had every tool call rendered as chat text — nothing executed, model looked like it
    was lying. A transient failure must be retried, not treated as a verdict.
    """
    config, mock_openai = _probe_client([TimeoutError("slot busy"), _native_completion()])
    with patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai), \
         patch("localharness.provider.client.asyncio.sleep", new=AsyncMock()):
        result = await LLMClient(config).detect_capabilities()

    assert result.tool_call_mode == "native"
    assert result.probe_error is None
    assert mock_openai.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_probe_retries_toolless_response_then_confirms_native():
    """A preamble-prone model narrating instead of calling on ONE sample is not evidence the
    server lacks native tool calling (the observed Qwen3.6 NVFP4 misread). Retry, don't condemn."""
    toolless = MagicMock()
    toolless.tool_calls = None
    toolless.content = "Let me take a look at that directory."
    config, mock_openai = _probe_client(
        [MagicMock(choices=[MagicMock(message=toolless)]), _native_completion()]
    )
    with patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai), \
         patch("localharness.provider.client.asyncio.sleep", new=AsyncMock()):
        result = await LLMClient(config).detect_capabilities()

    assert result.tool_call_mode == "native"
    assert mock_openai.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_probe_does_not_retry_definitive_answers():
    """A 400 is the server ANSWERING that it will not accept `tools`, and confirmed native is
    already the best outcome — neither is retried, so the healthy path costs exactly one call."""
    import openai as openai_module

    bad_request = openai_module.BadRequestError(
        message="tools not supported", response=MagicMock(status_code=400, headers={}), body=None
    )
    config, mock_openai = _probe_client(bad_request)
    with patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai), \
         patch("localharness.provider.client.asyncio.sleep", new=AsyncMock()):
        result = await LLMClient(config).detect_capabilities()
    assert result.tool_call_mode == "xml"
    assert mock_openai.chat.completions.create.await_count == 1

    config, mock_openai = _probe_client([_native_completion()])
    with patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai), \
         patch("localharness.provider.client.asyncio.sleep", new=AsyncMock()):
        result = await LLMClient(config).detect_capabilities()
    assert result.tool_call_mode == "native"
    assert mock_openai.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_probe_give_up_log_reports_actual_attempts_not_the_ceiling(caplog):
    """The give-up warning must count attempts that HAPPENED, not the retry ceiling.

    A definitive HTTP 400 breaks on the first pass, so claiming "after 3 attempts" sends
    whoever is debugging a probe failure hunting two retries that never occurred — in a log
    line whose entire job is to orient them toward the runtime's parser flag."""
    import logging
    import openai as openai_module

    bad_request = openai_module.BadRequestError(
        message="tools not supported", response=MagicMock(status_code=400, headers={}), body=None
    )
    config, mock_openai = _probe_client(bad_request)
    with caplog.at_level(logging.WARNING, logger="localharness.provider.client"), \
         patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai), \
         patch("localharness.provider.client.asyncio.sleep", new=AsyncMock()):
        await LLMClient(config).detect_capabilities()

    give_up = [r.getMessage() for r in caplog.records if "never confirmed native" in r.getMessage()]
    assert len(give_up) == 1, caplog.text
    assert "after 1 attempts" in give_up[0], give_up[0]
    assert mock_openai.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_probe_gives_up_after_exhausting_attempts():
    """A genuinely dead probe still ends in xml with probe_error set — retry widens the window
    for a transient blip, it does NOT invent a capability the server never demonstrated."""
    config, mock_openai = _probe_client(TimeoutError("still busy"))
    with patch("localharness.provider.client.AsyncOpenAI", return_value=mock_openai), \
         patch("localharness.provider.client.asyncio.sleep", new=AsyncMock()):
        result = await LLMClient(config).detect_capabilities()

    assert result.tool_call_mode == "xml"
    assert result.probe_error is not None
    assert mock_openai.chat.completions.create.await_count == 3


# ---------------------------------------------------------------------------
# Error type hierarchy
# ---------------------------------------------------------------------------


def test_error_types_exist():
    """All error types are importable and have correct inheritance."""
    assert issubclass(ProviderConnectionError, ProviderError)
    assert issubclass(ProviderTimeoutError, ProviderError)
    assert issubclass(ProviderRateLimitError, ProviderError)
    assert issubclass(ProviderAPIError, ProviderError)
    assert issubclass(MalformedResponseError, ProviderError)
    assert issubclass(ProviderError, Exception)


def test_provider_error_cause():
    cause = ValueError("inner")
    err = ProviderError("outer", cause=cause)
    assert str(err) == "outer"
    assert err.cause is cause


def test_provider_timeout_error():
    err = ProviderTimeoutError("timed out", tokens_generated=42)
    assert err.tokens_generated == 42
    assert isinstance(err, ProviderError)


def test_provider_rate_limit_error():
    err = ProviderRateLimitError("429", retry_after_seconds=5.0)
    assert err.retry_after_seconds == 5.0


def test_provider_api_error():
    err = ProviderAPIError("bad request", status_code=400)
    assert err.status_code == 400


def test_malformed_response_error():
    err = MalformedResponseError("bad response", raw="garbage")
    assert err.raw == "garbage"


# ---------------------------------------------------------------------------
# _complete_xml always folds the tool-call syntax into the system prompt (not only on the
# BadRequestError fallback) — llama.cpp+Gemma returns HTTP 200 while silently dropping an
# unsupported `tools` param, so waiting for a 400 never injects for that server.
# ---------------------------------------------------------------------------

_PROBE_TOOL = {
    "name": "list_files",
    "description": "List directory contents",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _xml_client() -> LLMClient:
    config = LLMConfig(
        base_url="http://127.0.0.1:0/v1",
        model="m",
        is_local=True,
        timeout_seconds=300.0,
        tool_call_mode="xml",
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        return LLMClient(config)


def _ok_response(**_kwargs):
    msg = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


@pytest.mark.asyncio
async def test_complete_xml_always_injects_tool_syntax_on_200():
    """A 200 response (tools silently dropped, e.g. llama.cpp+Gemma) must still see the taught
    XML syntax — injection must not depend on a BadRequestError to fire."""
    client = _xml_client()
    captured: dict = {}

    async def _spy_create(**kwargs):
        captured.update(kwargs)
        return _ok_response()

    client._client.chat.completions.create = _spy_create

    await client._complete_xml(
        messages=[{"role": "system", "content": "sys"}], tools=[_PROBE_TOOL], stream=False
    )

    sys_msg = captured["messages"][0]
    assert sys_msg["role"] == "system"
    assert "<tool_call>" in sys_msg["content"]
    assert "list_files" in sys_msg["content"]
    assert captured.get("tools")  # kwargs["tools"] is still kept (harmless if supported)


@pytest.mark.asyncio
async def test_complete_xml_inserts_system_message_when_absent():
    """No system message in the conversation -> one is inserted carrying the injection, rather
    than silently sending tools with no XML-mode instructions at all."""
    client = _xml_client()
    captured: dict = {}

    async def _spy_create(**kwargs):
        captured.update(kwargs)
        return _ok_response()

    client._client.chat.completions.create = _spy_create

    await client._complete_xml(
        messages=[{"role": "user", "content": "hi"}], tools=[_PROBE_TOOL], stream=False
    )

    assert captured["messages"][0]["role"] == "system"
    assert "<tool_call>" in captured["messages"][0]["content"]
    assert captured["messages"][1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_complete_xml_fallback_does_not_double_inject_after_400():
    """If the primary XML-mode attempt 400s, the fallback reuses the ALREADY-injected messages —
    the marker guard in _fold_tool_injection must stop it from appending the block twice."""
    import openai

    client = _xml_client()
    calls: list[dict] = []

    async def _spy_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise openai.BadRequestError(
                message="tools not supported",
                response=MagicMock(status_code=400, headers={}),
                body=None,
            )
        return _ok_response()

    client._client.chat.completions.create = _spy_create

    await client._complete_xml(
        messages=[{"role": "system", "content": "sys"}], tools=[_PROBE_TOOL], stream=False
    )

    assert len(calls) == 2
    final_sys_content = calls[1]["messages"][0]["content"]
    assert final_sys_content.count("You have access to the following tools") == 1
    # The 2nd (fallback) attempt must not resend a `tools` param it already 400'd on.
    assert "tools" not in calls[1]


# ---------------------------------------------------------------------------
# Fix E — native-mode 400 retry symmetry. _complete_xml self-heals a rejected `tools` param;
# _complete_native used to hard-error on the same 400. Native must now scope-retry the KNOWN
# optional params (tools, extra_body/chat_template_kwargs), stick the rejection so it isn't
# re-eaten every turn, and STILL surface a genuinely malformed 400 (no blanket retry).
# ---------------------------------------------------------------------------


def _native_client() -> LLMClient:
    # is_local=False → _inference_gate yields without a TCP probe (CPU-only, no network).
    config = LLMConfig(
        base_url="http://127.0.0.1:0/v1",
        model="m",
        is_local=False,
        timeout_seconds=300.0,
        tool_call_mode="native",
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        return LLMClient(config)


def _bad_request(message: str):
    import openai
    return openai.BadRequestError(
        message=message, response=MagicMock(status_code=400, headers={}), body=None
    )


@pytest.mark.asyncio
async def test_complete_native_retries_once_dropping_rejected_tools():
    """A native 400 that names the `tools` param → drop tools, remember it (sticky), retry ONCE,
    succeed. The sticky flag then keeps the NEXT turn from re-sending tools (no repeat 400)."""
    client = _native_client()
    calls: list[dict] = []

    async def _spy_create(**kwargs):  # the server 400s whenever `tools` is present
        calls.append(kwargs)
        if "tools" in kwargs:
            raise _bad_request('"tools" is not supported by this model')
        return _ok_response()

    client._client.chat.completions.create = _spy_create

    msg, _usage = await client._complete_native(
        messages=[{"role": "user", "content": "hi"}], tools=[_PROBE_TOOL], stream=False
    )
    assert len(calls) == 2
    assert "tools" in calls[0]                     # first attempt sent tools
    assert "tools" not in calls[1]                 # retry dropped them
    assert client._tools_param_rejected is True    # remembered
    assert msg.content == "ok"

    # Sticky, across turns: the next call must NOT re-send tools (no second 400 round-trip).
    calls.clear()
    await client._complete_native(
        messages=[{"role": "user", "content": "again"}], tools=[_PROBE_TOOL], stream=False
    )
    assert len(calls) == 1 and "tools" not in calls[0]


@pytest.mark.asyncio
async def test_complete_native_retries_dropping_rejected_extra_body():
    """The other native optional param: a 400 naming chat_template_kwargs/extra_body (sent only on
    internal disable_thinking calls) → drop extra_body, remember it, retry once, succeed."""
    client = _native_client()
    calls: list[dict] = []

    async def _spy_create(**kwargs):  # the server 400s whenever `extra_body` is present
        calls.append(kwargs)
        if "extra_body" in kwargs:
            raise _bad_request("extra_body field 'chat_template_kwargs' is not permitted")
        return _ok_response()

    client._client.chat.completions.create = _spy_create

    msg, _usage = await client._complete_native(
        messages=[{"role": "user", "content": "hi"}], tools=None, stream=False,
        disable_thinking=True,
    )
    assert len(calls) == 2
    assert "extra_body" in calls[0]
    assert "extra_body" not in calls[1]
    assert client._extra_body_rejected is True
    assert msg.content == "ok"


@pytest.mark.asyncio
async def test_complete_native_reraises_genuine_malformed_400():
    """A 400 that is NOT a known rejected-param shape (genuine malformed request) must SURFACE,
    not be swallowed by a blanket retry — the native path scopes its retry. No retry, no sticky."""
    client = _native_client()
    calls: list[dict] = []

    async def _spy_create(**kwargs):
        calls.append(kwargs)
        raise _bad_request("messages: roles must alternate user/assistant")

    client._client.chat.completions.create = _spy_create

    with pytest.raises(ProviderAPIError):
        await client._complete_native(
            messages=[{"role": "user", "content": "hi"}], tools=[_PROBE_TOOL], stream=False
        )
    assert len(calls) == 1                          # surfaced on the first try, never retried
    assert client._tools_param_rejected is False    # sticky NOT set on a genuine 400


@pytest.mark.asyncio
async def test_complete_xml_downgrades_native_tool_history():
    """Iteration 2 in xml mode replays history holding native `role:"tool"` messages and
    assistant `tool_calls` fields. Templates without tool support (Gemma 3) hard-reject that
    shape ("Conversation roles must alternate"), so the outgoing request must carry text-only
    alternating roles: tool results as <tool_response> user turns, tool_calls stripped,
    consecutive same-role turns merged."""
    client = _xml_client()
    captured: dict = {}

    async def _spy_create(**kwargs):
        captured.update(kwargs)
        return _ok_response()

    client._client.chat.completions.create = _spy_create

    await client._complete_xml(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read the file"},
            {
                "role": "assistant",
                "content": '<tool_call>\n<name>read</name>\n<parameters>{"path": "x"}</parameters>\n</tool_call>',
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path": "x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "file says apricot"},
        ],
        tools=[_PROBE_TOOL],
        stream=False,
    )

    roles = [m["role"] for m in captured["messages"]]
    assert "tool" not in roles
    assert all("tool_calls" not in m for m in captured["messages"])
    tool_turn = captured["messages"][-1]
    assert tool_turn["role"] == "user"
    assert "<tool_response>" in tool_turn["content"] and "apricot" in tool_turn["content"]
    body = roles[1:]
    assert all(a != b for a, b in zip(body, body[1:]))


def test_tools_to_api_format_sanitizes_registry_names():
    """`mcp:fetch`-style names violate the OpenAI function-name grammar and make llama.cpp
    400 the whole request, knocking MCP/plugin scenarios off the native path. Wire names
    are sanitized; the unmap restores originals."""
    from localharness.provider.client import _tools_to_api_format

    tools = [
        {"name": "mcp:fetch", "description": "d", "parameters": {}},
        {"name": "plugin:research_tools.exa_search", "description": "d", "parameters": {}},
        {"name": "read", "description": "d", "parameters": {}},
    ]
    payload, unmap = _tools_to_api_format(tools)
    names = [p["function"]["name"] for p in payload]
    assert names == ["mcp_fetch", "plugin_research_tools_exa_search", "read"]
    assert unmap["mcp_fetch"] == "mcp:fetch"
    assert unmap["plugin_research_tools_exa_search"] == "plugin:research_tools.exa_search"
    assert unmap["read"] == "read"
    # collision: two originals sanitizing to the same wire name stay distinct
    payload2, unmap2 = _tools_to_api_format(
        [
            {"name": "a:b", "description": "d", "parameters": {}},
            {"name": "a.b", "description": "d", "parameters": {}},
        ]
    )
    n2 = [p["function"]["name"] for p in payload2]
    assert len(set(n2)) == 2
    assert {unmap2[n] for n in n2} == {"a:b", "a.b"}


def test_unmap_tool_call_names_restores_originals():
    from types import SimpleNamespace
    from localharness.provider.client import LLMClient

    msg = SimpleNamespace(
        content=None,
        tool_calls=[
            {"id": "1", "type": "function", "function": {"name": "mcp_fetch", "arguments": "{}"}},
            # model echoed the original registry name from history — must pass through
            {"id": "2", "type": "function", "function": {"name": "mcp:fetch", "arguments": "{}"}},
        ],
    )
    LLMClient._unmap_tool_call_names(msg, {"mcp_fetch": "mcp:fetch"})
    assert msg.tool_calls[0]["function"]["name"] == "mcp:fetch"
    assert msg.tool_calls[1]["function"]["name"] == "mcp:fetch"


@pytest.mark.asyncio
async def test_complete_xml_tools_rejection_is_sticky():
    """After one BadRequestError on `tools=`, later xml-mode requests must not re-send it —
    observed live as 44 x HTTP 400 in a single bench run, one per iteration."""
    import openai

    client = _xml_client()
    calls: list[dict] = []

    async def _spy_create(**kwargs):
        calls.append(kwargs)
        if "tools" in kwargs:
            raise openai.BadRequestError(
                message="invalid tools", response=MagicMock(status_code=400, headers={}), body=None
            )
        return _ok_response()

    client._client.chat.completions.create = _spy_create

    msgs = [{"role": "system", "content": "sys"}]
    await client._complete_xml(messages=msgs, tools=[_PROBE_TOOL], stream=False)
    assert "tools" in calls[0] and "tools" not in calls[1]  # first try + fallback retry
    await client._complete_xml(messages=msgs, tools=[_PROBE_TOOL], stream=False)
    assert "tools" not in calls[2]  # sticky: no third 400
    assert len(calls) == 3
