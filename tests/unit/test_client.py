"""Tests for LLMClient (client.py)."""
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
    assert config.timeout_seconds == 300.0
    assert config.temperature == 0.6
    assert config.max_tokens == 4096
    assert config.tool_call_mode == "native"
    assert config.api_key == "none"
    assert config.connect_timeout_seconds == 5.0
    assert config.context_window == 128_000
    assert config.is_local is True
    assert config.extra_headers == {}
    assert config.stop_sequences == []


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
