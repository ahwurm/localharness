"""The client half of /effort (spec 16 §7.4; 03-CONTEXT D-17, D-18, D-21, D-29, D-31, D-32).

One `extra_body` assembler for the three request paths (`_complete_native`, `_complete_xml`,
`_complete_xml_fallback`), the per-runtime budget-key table, the per-key sticky-400 rule and the
client-side reasoning cut. Nothing here opens a socket: `is_local=False` skips the inference gate
and the AsyncOpenAI client is replaced by a scripted `create`.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest

from localharness.provider.client import (
    EFFORT_BUDGET_KEY,
    EFFORT_KEYS,
    LLMClient,
    LLMConfig,
)

MSGS = [{"role": "user", "content": "hi"}]
TOOL = {
    "name": "list_files",
    "description": "List directory contents",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
MODES = ("native", "xml", "text")
PROVIDERS = ("llamacpp", "vllm", "ollama", "lmstudio", "unknown", None)


def _bad(message: str) -> openai.BadRequestError:
    return openai.BadRequestError(
        message=message, response=MagicMock(status_code=400, headers={}), body=None
    )


class _Stream:
    """An async chunk stream that records whether it was closed (the cut must close it first)."""

    def __init__(self, reasoning: list[str], content: str = "ok") -> None:
        self._reasoning = reasoning
        self._content = content
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for r in self._reasoning:
            yield SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None, reasoning_content=r),
                    finish_reason=None,
                )],
            )
            await asyncio.sleep(0)
        yield SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=self._content, tool_calls=None, reasoning_content=None),
                finish_reason="stop",
            )],
        )


class Server:
    """A scripted `chat.completions.create`. Each step is an Exception (raised), a callable taking
    the kwargs, or "ok" (a normal answer for the request's stream flag). The last step repeats."""

    def __init__(self, *steps, reasoning: list[str] | None = None) -> None:
        self.steps = list(steps) or ["ok"]
        self.calls: list[dict] = []
        self.streams: list[_Stream] = []
        self.reasoning = reasoning or []

    async def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(kwargs)
        return self._answer(kwargs)

    def _answer(self, kwargs):
        if kwargs.get("stream"):
            s = _Stream(self.reasoning)
            self.streams.append(s)
            return s
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


def make(mode: str = "native", provider_type: str | None = "llamacpp", server: Server | None = None):
    cfg = LLMConfig(
        base_url="http://127.0.0.1:0/v1", model="m", is_local=False, timeout_seconds=300.0,
        tool_call_mode=mode, provider_type=provider_type,
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        client = LLMClient(cfg)
    server = server or Server()
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=server.create))
    )
    return client, server


# ---------------------------------------------------------------------------
# The capability table and the read-only surface
# ---------------------------------------------------------------------------


def test_effort_budget_key_table_is_the_spec_table():
    assert EFFORT_BUDGET_KEY == {
        "llamacpp": "reasoning_budget_tokens", "vllm": None, "ollama": None,
        "lmstudio": None, "unknown": None, None: None,
    }
    assert EFFORT_KEYS == frozenset({"chat_template_kwargs", "reasoning_budget_tokens"})


@pytest.mark.parametrize("ptype", PROVIDERS)
def test_effort_budget_key_property_answers_the_table(ptype):
    client, _ = make(provider_type=ptype)
    assert client.effort_budget_key == EFFORT_BUDGET_KEY[ptype]


def test_effort_sets_are_frozensets_and_empty_on_a_fresh_client():
    client, _ = make()
    assert client.rejected_effort_keys == frozenset()
    assert isinstance(client.rejected_effort_keys, frozenset)
    assert client.accepted_effort_keys == frozenset()
    assert isinstance(client.accepted_effort_keys, frozenset)
    # the old sticky bool is gone; the per-key set replaced it
    assert [n for n in vars(client) if n.endswith("body_rejected")] == []


def test_the_scoped_exceptions_to_11_are_named_in_the_docstrings():
    assert "/effort" in LLMClient.complete.__doc__
    assert "/effort" in LLMClient.stream_complete.__doc__
    assert "/effort" in LLMClient._native_kwargs.__doc__


# ---------------------------------------------------------------------------
# The assembler on the native and XML paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_llamacpp_budget_rides_extra_body(mode):
    client, server = make(mode, "llamacpp")
    await client.stream_complete(MSGS, reasoning_budget=1200)
    assert server.calls[0]["extra_body"] == {"reasoning_budget_tokens": 1200}


@pytest.mark.parametrize("mode", MODES)
async def test_budget_and_thinking_off_share_one_extra_body(mode):
    client, server = make(mode, "llamacpp")
    await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=1200)
    assert server.calls[0]["extra_body"] == {**THINK_OFF, "reasoning_budget_tokens": 1200}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ptype", [p for p in PROVIDERS if p != "llamacpp"])
async def test_a_runtime_without_a_budget_key_sends_no_budget(mode, ptype):
    client, server = make(mode, ptype)
    await client.complete(MSGS, reasoning_budget=1200)
    assert "extra_body" not in server.calls[0]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ptype", [p for p in PROVIDERS if p != "llamacpp"])
async def test_thinking_off_is_sent_on_every_runtime(mode, ptype):
    client, server = make(mode, ptype)
    await client.complete(MSGS, disable_thinking=True, reasoning_budget=1200)
    assert server.calls[0]["extra_body"] == THINK_OFF


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ptype", PROVIDERS)
async def test_no_budget_and_no_thinking_off_is_byte_identical_to_today(mode, ptype):
    client, server = make(mode, ptype)
    await client.stream_complete(MSGS)
    assert set(server.calls[0]) == {"model", "messages", "temperature", "stream", "stream_options"}
    server.calls.clear()
    await client.complete(MSGS)
    assert set(server.calls[0]) == {"model", "messages", "temperature"}


@pytest.mark.parametrize("mode", MODES)
async def test_complete_accepts_reasoning_budget_and_builds_the_same_body(mode):
    client, server = make(mode, "llamacpp")
    await client.complete(MSGS, disable_thinking=True, reasoning_budget=300)
    assert server.calls[0]["extra_body"] == {**THINK_OFF, "reasoning_budget_tokens": 300}
    assert "stream" not in server.calls[0]


@pytest.mark.parametrize("mode", MODES)
async def test_budget_none_sends_no_budget_key_on_llamacpp(mode):
    client, server = make(mode, "llamacpp")
    await client.stream_complete(MSGS, reasoning_budget=None)
    assert "extra_body" not in server.calls[0]


# ---------------------------------------------------------------------------
# The XML fallback path (reached through a tools 400 on the XML request)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("xml", "text"))
@pytest.mark.parametrize(
    ("disable", "budget", "expected"),
    [
        (False, 1200, {"reasoning_budget_tokens": 1200}),
        (True, None, THINK_OFF),
        (True, 1200, {**THINK_OFF, "reasoning_budget_tokens": 1200}),
        (False, None, None),
    ],
)
async def test_fallback_request_carries_the_same_extra_body_and_no_tools(mode, disable, budget, expected):
    client, server = make(mode, "llamacpp", Server(_bad('"tools" is not supported'), "ok"))
    await client.stream_complete(MSGS, [TOOL], disable_thinking=disable, reasoning_budget=budget)
    first, second = server.calls
    assert "tools" in first and "tools" not in second
    assert first.get("extra_body") == expected
    assert second.get("extra_body") == expected
    assert client._tools_param_rejected is True


# ---------------------------------------------------------------------------
# The rejection set suppresses a key on every path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_a_rejected_thinking_off_is_not_sent_but_the_budget_still_is(mode):
    client, server = make(mode, "llamacpp")
    client._rejected_effort_keys = {"chat_template_kwargs"}
    await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=50)
    assert server.calls[0]["extra_body"] == {"reasoning_budget_tokens": 50}


@pytest.mark.parametrize("mode", MODES)
async def test_a_rejected_budget_is_not_sent_but_thinking_off_still_is(mode):
    client, server = make(mode, "llamacpp")
    client._rejected_effort_keys = {"reasoning_budget_tokens"}
    await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=50)
    assert server.calls[0]["extra_body"] == THINK_OFF


async def test_the_fallback_path_honours_the_rejection_set():
    client, server = make("xml", "llamacpp", Server(_bad("tools unsupported"), "ok"))
    client._rejected_effort_keys = {"chat_template_kwargs"}
    await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=7)
    assert [c.get("extra_body") for c in server.calls] == [
        {"reasoning_budget_tokens": 7}, {"reasoning_budget_tokens": 7},
    ]


# ---------------------------------------------------------------------------
# accepted_effort_keys (D-29)
# ---------------------------------------------------------------------------


async def test_accepted_keys_grow_once_create_returns():
    client, _ = make("native", "llamacpp")
    await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=1200)
    assert client.accepted_effort_keys == frozenset({"chat_template_kwargs", "reasoning_budget_tokens"})


async def test_a_request_with_no_effort_key_leaves_the_accepted_set_unchanged():
    client, _ = make("native", "llamacpp")
    client._accepted_effort_keys = {"reasoning_budget_tokens"}
    await client.stream_complete(MSGS)
    assert client.accepted_effort_keys == frozenset({"reasoning_budget_tokens"})


async def test_a_create_that_raises_adds_nothing():
    client, _ = make("native", "llamacpp", Server(ConnectionError("boom")))
    with pytest.raises(Exception):
        await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=9)
    assert client.accepted_effort_keys == frozenset()


async def test_accepted_is_marked_on_the_non_streaming_branch_too():
    client, _ = make("native", "llamacpp")
    await client.complete(MSGS, reasoning_budget=9)
    assert client.accepted_effort_keys == frozenset({"reasoning_budget_tokens"})


async def test_a_later_stream_error_does_not_undo_the_acceptance():
    client, _ = make("native", "llamacpp", Server(lambda kw: _Boom()))
    with pytest.raises(Exception):
        await client.stream_complete(MSGS, reasoning_budget=9)
    assert client.accepted_effort_keys == frozenset({"reasoning_budget_tokens"})


class _Boom:
    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        raise ConnectionError("stream died")
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# rebind_endpoint (D-18, D-29)
# ---------------------------------------------------------------------------


def test_rebind_success_resets_both_sets():
    client, _ = make()
    client._rejected_effort_keys = {"chat_template_kwargs"}
    client._accepted_effort_keys = {"reasoning_budget_tokens"}
    with patch("localharness.provider.client.AsyncOpenAI"):
        client.rebind_endpoint("http://127.0.0.1:1/v1")
    assert client.rejected_effort_keys == frozenset()
    assert client.accepted_effort_keys == frozenset()


def test_a_failed_rebind_restores_both_sets_and_reraises(monkeypatch):
    client, _ = make()
    client._rejected_effort_keys = {"chat_template_kwargs"}
    client._accepted_effort_keys = {"reasoning_budget_tokens"}

    def boom(self):
        raise RuntimeError("cannot build client")

    monkeypatch.setattr(LLMClient, "_build_client", boom)
    with pytest.raises(RuntimeError, match="cannot build client"):
        client.rebind_endpoint("http://127.0.0.1:1/v1")
    assert client.rejected_effort_keys == frozenset({"chat_template_kwargs"})
    assert client.accepted_effort_keys == frozenset({"reasoning_budget_tokens"})
