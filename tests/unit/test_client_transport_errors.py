"""A raw httpx transport error raised while a streamed body is read maps to the provider error
types the loop already handles: a drop is a ProviderConnectionError, a hang a ProviderTimeoutError.

The openai library converts only the errors raised while it sends the request or reads headers; a
body read after that surfaces as an httpx exception, which `_wrap_error` used to turn into a base
ProviderError. Hermetic: the SDK client is served by an httpx.MockTransport (no socket, no server)
through `provider.client._TRANSPORT`, and `is_local=False` keeps the inference gate's TCP probe and
flock out of the way."""
from __future__ import annotations

import json

import httpx
import openai
import pytest

from localharness.provider import client as client_mod
from localharness.provider.client import (
    LLMClient,
    LLMConfig,
    ProviderConnectionError,
    ProviderDegenerateError,
    ProviderError,
    ProviderTimeoutError,
)

_DROP = "peer closed connection without sending complete message body: incomplete chunked read"


def _cfg(**kw) -> LLMConfig:
    kw.setdefault("base_url", "http://127.0.0.1:9/v1")
    kw.setdefault("model", "m")
    kw.setdefault("is_local", False)
    kw.setdefault("tool_call_mode", "native")
    return LLMConfig(**kw)


def _sse(delta: dict) -> bytes:
    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


class _DroppingStream(httpx.AsyncByteStream):
    """One good SSE chunk, then the transport error a killed server produces mid-body."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aiter__(self):
        yield _sse({"role": "assistant", "content": "par"})
        raise self._exc


class _CompleteStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield _sse({"role": "assistant", "content": "ok"})
        yield b"data: [DONE]\n\n"


def _serve(monkeypatch, streams: list) -> list:
    """Install a MockTransport answering each request with the next stream; returns the call log."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        stream = streams[min(len(calls), len(streams)) - 1]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    monkeypatch.setattr(client_mod, "_TRANSPORT", httpx.MockTransport(handler))
    return calls


# --- the mapping table ------------------------------------------------------------------------

def _req() -> httpx.Request:
    return httpx.Request("POST", "http://x/v1/chat/completions")


@pytest.mark.parametrize("exc", [
    httpx.RemoteProtocolError(_DROP),
    httpx.ReadError("connection reset"),
    httpx.ConnectError("refused"),
    httpx.WriteError("broken pipe"),
])
def test_raw_transport_error_is_a_connection_error(exc):
    out = LLMClient(_cfg())._wrap_error(exc)
    assert type(out) is ProviderConnectionError
    assert out.cause is exc


@pytest.mark.parametrize("exc", [
    httpx.ReadTimeout("read timed out"),
    httpx.ConnectTimeout("connect timed out"),
    httpx.PoolTimeout("pool timed out"),
])
def test_raw_timeout_is_a_timeout_error_not_a_connection_error(exc):
    # httpx.TimeoutException subclasses TransportError: a hang must not take the reconnect wait.
    assert isinstance(exc, httpx.TransportError)
    out = LLMClient(_cfg())._wrap_error(exc)
    assert type(out) is ProviderTimeoutError
    assert out.cause is exc


def test_provider_error_passes_through_as_the_same_object():
    err = ProviderDegenerateError("loop", unit="ab", repeats=9, chars=18)
    assert LLMClient(_cfg())._wrap_error(err) is err


def test_unknown_exception_is_a_base_provider_error():
    out = LLMClient(_cfg())._wrap_error(ValueError("x"))
    assert type(out) is ProviderError


def test_openai_connection_and_timeout_errors_stay_connection_errors():
    """The openai branches and their order are unchanged: APITimeoutError subclasses
    APIConnectionError, so a connect-phase timeout is a connection error (and takes the reconnect
    wait), while a mid-stream httpx timeout is a timeout."""
    c = LLMClient(_cfg())
    assert type(c._wrap_error(openai.APIConnectionError(request=_req()))) is ProviderConnectionError
    assert type(c._wrap_error(openai.APITimeoutError(request=_req()))) is ProviderConnectionError


# --- a mid-stream drop through the real client ------------------------------------------------

async def test_midstream_drop_reaches_the_caller_as_a_connection_error(monkeypatch):
    drop = httpx.RemoteProtocolError(_DROP)
    _serve(monkeypatch, [_DroppingStream(drop)])
    client = LLMClient(_cfg())
    try:
        with pytest.raises(ProviderConnectionError) as ei:
            await client.stream_complete(messages=[{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()
    cause = ei.value.__cause__ or ei.value.cause
    # The SDK may wrap the raw httpx error on the way; the httpx exception must stay reachable.
    seen = []
    while cause is not None:
        seen.append(cause)
        cause = cause.__cause__ or getattr(cause, "cause", None)
    assert any(isinstance(e, httpx.RemoteProtocolError) for e in seen), seen


async def test_midstream_hang_reaches_the_caller_as_a_timeout_error(monkeypatch):
    _serve(monkeypatch, [_DroppingStream(httpx.ReadTimeout("read timed out"))])
    client = LLMClient(_cfg())
    try:
        with pytest.raises(ProviderTimeoutError):
            await client.stream_complete(messages=[{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()


async def test_the_drop_is_not_sticky(monkeypatch):
    calls = _serve(monkeypatch, [_DroppingStream(httpx.RemoteProtocolError(_DROP)), _CompleteStream()])
    client = LLMClient(_cfg())
    try:
        with pytest.raises(ProviderConnectionError):
            await client.stream_complete(messages=[{"role": "user", "content": "hi"}])
        msg, _usage = await client.stream_complete(messages=[{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()
    assert msg.content == "ok"
    assert len(calls) == 2
