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
    REASONING_CHARS_PER_TOKEN,
    LLMClient,
    LLMConfig,
    ProviderAPIError,
    ProviderDegenerateError,
    ProviderError,
    ProviderReasoningBudgetError,
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
        "lmstudio": None, "strata": None, "unknown": None, None: None,
    }
    assert EFFORT_KEYS == frozenset(
        {"chat_template_kwargs", "reasoning_budget_tokens", "reasoning_effort"}
    )


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
    assert set(server.calls[0]) == {"model", "messages", "stream", "stream_options"}
    server.calls.clear()
    await client.complete(MSGS)
    assert set(server.calls[0]) == {"model", "messages"}


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


# ---------------------------------------------------------------------------
# The classifier (D-32): pure, one fixed order
# ---------------------------------------------------------------------------

BOTH = {"extra_body": {**THINK_OFF, "reasoning_budget_tokens": 5}}


def _classify(client, text, sent=BOTH, **kw):
    return client._classify_400(_bad(text), sent, **kw)


def test_classifier_order_overflow_then_tools_then_named_then_neither():
    client, _ = make()
    assert _classify(client, "exceed_context_size_error tool reasoning_budget_tokens") == ("overflow", ())
    assert _classify(client, "tool reasoning_budget_tokens") == ("tools", ())
    assert _classify(client, "tool reasoning_budget_tokens", check_tools=False) == (
        "effort", ("reasoning_budget_tokens",),
    )
    assert _classify(client, "bad enable_thinking") == ("effort", ("chat_template_kwargs",))
    assert _classify(client, "unknown field 'foo'") == (
        "neither", ("chat_template_kwargs", "reasoning_budget_tokens"),
    )
    assert _classify(client, "unknown field 'foo'", sent={"model": "m"}) == ("neither", ())


def test_classifier_is_case_insensitive_and_records_nothing():
    client, _ = make()
    assert _classify(client, "EXCEED_CONTEXT_SIZE_ERROR") == ("overflow", ())
    assert _classify(client, "Reasoning_Budget_Tokens is not allowed") == (
        "effort", ("reasoning_budget_tokens",),
    )
    assert client.rejected_effort_keys == frozenset() and client._tools_param_rejected is False


def test_a_key_only_in_the_sent_kwargs_is_not_named_by_the_error_text():
    client, _ = make()
    sent = {"extra_body": dict(THINK_OFF)}   # only thinking-off was sent
    # the text names the budget key, which was NOT sent -> it names no SENT key
    assert _classify(client, "reasoning_budget_tokens is unknown", sent=sent) == (
        "neither", ("chat_template_kwargs",),
    )


# ---------------------------------------------------------------------------
# Native: retry once without the key, record only when the retry succeeds
# ---------------------------------------------------------------------------


async def test_a_400_naming_the_budget_drops_only_the_budget_and_records_it():
    client, server = make("native", "llamacpp", Server(_bad("unknown field reasoning_budget_tokens"), "ok"))
    msg, _ = await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=1200)
    assert len(server.calls) == 2
    assert server.calls[0]["extra_body"] == {**THINK_OFF, "reasoning_budget_tokens": 1200}
    assert server.calls[1]["extra_body"] == THINK_OFF
    assert client.rejected_effort_keys == frozenset({"reasoning_budget_tokens"})
    assert msg.content == "ok"


async def test_a_400_naming_enable_thinking_drops_only_thinking_off():
    client, server = make("native", "llamacpp", Server(_bad("Unexpected key: enable_thinking"), "ok"))
    await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=1200)
    assert server.calls[1]["extra_body"] == {"reasoning_budget_tokens": 1200}
    assert client.rejected_effort_keys == frozenset({"chat_template_kwargs"})


async def test_a_400_naming_neither_key_drops_every_effort_key_sent():
    client, server = make("native", "llamacpp", Server(_bad("unknown field 'foo'"), "ok"))
    await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=1200)
    assert "extra_body" not in server.calls[1]
    assert client.rejected_effort_keys == frozenset({"chat_template_kwargs", "reasoning_budget_tokens"})


async def test_a_400_naming_neither_key_with_no_effort_key_sent_surfaces_at_once():
    client, server = make("native", "llamacpp", Server(_bad("unknown field 'foo'")))
    with pytest.raises(ProviderAPIError):
        await client.stream_complete(MSGS)
    assert len(server.calls) == 1
    assert client.rejected_effort_keys == frozenset()


async def test_the_bare_extra_body_text_no_longer_classifies_a_rejection():
    # Only the budget was sent. The old bare-"extra_body" rule would have flagged thinking-off too.
    client, server = make("native", "llamacpp", Server(_bad("extra_body: invalid"), "ok"))
    await client.stream_complete(MSGS, reasoning_budget=40)
    assert client.rejected_effort_keys == frozenset({"reasoning_budget_tokens"})
    assert "extra_body" not in server.calls[1]


async def test_a_failed_retry_restores_the_set_and_surfaces_the_original_400():
    retry_error = _bad("messages: roles must alternate")   # a 400 of its own: the original wins (D-18)
    client, server = make(
        "native", "llamacpp", Server(_bad("unknown field reasoning_budget_tokens"), retry_error)
    )
    client._rejected_effort_keys = {"chat_template_kwargs"}   # the pre-request value
    with pytest.raises(ProviderAPIError) as info:
        await client.stream_complete(MSGS, reasoning_budget=1200)
    assert "reasoning_budget_tokens" in str(info.value)       # the ORIGINAL 400's text
    assert info.value.status_code == 400
    assert info.value.__cause__ is retry_error
    assert client.rejected_effort_keys == frozenset({"chat_template_kwargs"})
    assert len(server.calls) == 2


async def test_a_retry_that_400s_again_surfaces_the_original_and_records_nothing():
    client, server = make(
        "native", "llamacpp",
        Server(_bad("unknown field reasoning_budget_tokens"), _bad("messages: roles must alternate")),
    )
    with pytest.raises(ProviderAPIError) as info:
        await client.stream_complete(MSGS, reasoning_budget=1200)
    assert "reasoning_budget_tokens" in str(info.value)
    assert client.rejected_effort_keys == frozenset()
    assert len(server.calls) == 2


async def test_a_cancelled_retry_leaves_the_set_as_it_was():
    def cancel(_kw):
        raise asyncio.CancelledError()

    client, _ = make(
        "native", "llamacpp", Server(_bad("unknown field reasoning_budget_tokens"), cancel)
    )
    with pytest.raises(asyncio.CancelledError):
        await client.stream_complete(MSGS, reasoning_budget=1200)
    assert client.rejected_effort_keys == frozenset()


def _request() -> MagicMock:
    return MagicMock()


@pytest.mark.parametrize("mode", MODES)
async def test_a_retry_that_fails_on_the_connection_surfaces_that_failure_not_the_400(mode):
    """IN-05: a dropped connection on the retry is a connection failure (the loop's retry handler
    keys on it), and the rejection set is still put back."""
    from localharness.provider.client import ProviderConnectionError

    dropped = openai.APIConnectionError(request=_request())
    client, server = make(
        mode, "llamacpp", Server(_bad("unknown field reasoning_budget_tokens"), dropped)
    )
    with pytest.raises(ProviderConnectionError) as info:
        await client.stream_complete(MSGS, reasoning_budget=1200)
    assert info.value.__cause__ is dropped
    assert client.rejected_effort_keys == frozenset()
    assert len(server.calls) == 2


async def test_a_retry_that_times_out_surfaces_the_timeout_not_the_400():
    from localharness.provider.client import ProviderTimeoutError

    slow = ProviderTimeoutError("generation timed out")   # what _consume_bounded raises past gen_timeout
    client, _ = make(
        "native", "llamacpp", Server(_bad("unknown field reasoning_budget_tokens"), slow)
    )
    client._rejected_effort_keys = {"chat_template_kwargs"}
    with pytest.raises(ProviderTimeoutError) as info:
        await client.stream_complete(MSGS, reasoning_budget=1200)
    assert info.value is slow
    assert client.rejected_effort_keys == frozenset({"chat_template_kwargs"})


async def test_a_failed_retry_takes_back_only_the_keys_it_added():
    """IN-05: the client is shared by concurrent children. A key a sibling recorded while this
    retry was in flight survives this retry's failure; the snapshot assignment used to wipe it."""
    def sibling_records_a_key_then_400s(_kw):
        client._rejected_effort_keys.add("chat_template_kwargs")   # a sibling's evidence, mid-retry
        raise _bad("messages: roles must alternate")

    client, _ = make(
        "native", "llamacpp",
        Server(_bad("unknown field reasoning_budget_tokens"), sibling_records_a_key_then_400s),
    )
    with pytest.raises(ProviderAPIError):
        await client.stream_complete(MSGS, reasoning_budget=1200)
    assert client.rejected_effort_keys == frozenset({"chat_template_kwargs"})


@pytest.mark.parametrize("text", [
    "exceed_context_size_error: prompt is too long",
    "Error: EXCEED_CONTEXT_SIZE_ERROR (n_ctx=4096)",
    "exceed_context_size_error, tools and reasoning_budget_tokens and enable_thinking are fine",
])
async def test_an_overflow_surfaces_at_once_with_the_set_untouched(text):
    client, server = make("native", "llamacpp", Server(_bad(text), "ok"))
    with pytest.raises(ProviderAPIError):
        await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=1200)
    assert len(server.calls) == 1
    assert client.rejected_effort_keys == frozenset()
    assert client._tools_param_rejected is False
    assert client.accepted_effort_keys == frozenset()


async def test_a_tools_400_retries_without_tools_and_keeps_every_effort_key():
    client, server = make("native", "llamacpp", Server(_bad('"tools" is not supported'), "ok"))
    await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=1200)
    first, second = server.calls
    assert "tools" in first and "tools" not in second
    assert first["extra_body"] == second["extra_body"] == {**THINK_OFF, "reasoning_budget_tokens": 1200}
    assert client.rejected_effort_keys == frozenset()
    assert client._tools_param_rejected is True


async def test_a_key_error_text_naming_a_tool_word_is_a_tools_error_first():
    # D-32 rule 2 before rule 3: the "tool" word wins even beside a key name.
    client, server = make("native", "llamacpp", Server(_bad("tool_choice with reasoning_budget_tokens"), "ok"))
    await client.stream_complete(MSGS, [TOOL], reasoning_budget=9)
    assert "tools" not in server.calls[1] and server.calls[1]["extra_body"] == {"reasoning_budget_tokens": 9}
    assert client.rejected_effort_keys == frozenset()


async def test_a_rejection_after_a_tools_400_without_tools_sent_falls_through_to_the_keys():
    client, server = make("native", "llamacpp", Server(_bad("tool parser: reasoning_budget_tokens"), "ok"))
    await client.stream_complete(MSGS, reasoning_budget=9)    # no tools were sent
    assert client.rejected_effort_keys == frozenset({"reasoning_budget_tokens"})
    assert client._tools_param_rejected is False


async def test_a_recorded_rejection_leaves_the_accepted_set():
    client, _ = make("native", "llamacpp", Server(_bad("unknown field reasoning_budget_tokens"), "ok"))
    client._accepted_effort_keys = {"chat_template_kwargs", "reasoning_budget_tokens"}
    await client.stream_complete(MSGS, reasoning_budget=1200)   # the retry sends no effort key
    assert client.rejected_effort_keys == frozenset({"reasoning_budget_tokens"})
    assert client.accepted_effort_keys == frozenset({"chat_template_kwargs"})
    assert not (client.accepted_effort_keys & client.rejected_effort_keys)


async def test_a_second_request_after_a_recorded_rejection_carries_no_such_key():
    client, server = make("native", "llamacpp", Server(_bad("unknown field reasoning_budget_tokens"), "ok"))
    await client.stream_complete(MSGS, reasoning_budget=1200)
    server.calls.clear()
    await client.stream_complete(MSGS, reasoning_budget=1200)
    assert len(server.calls) == 1 and "extra_body" not in server.calls[0]


# ---------------------------------------------------------------------------
# XML path: tools fallback as today; a named key retries the XML request
# ---------------------------------------------------------------------------


class _Spy:
    def __init__(self, client):
        self.n = 0
        real = client._complete_xml_fallback

        async def wrapped(*a, **kw):
            self.n += 1
            return await real(*a, **kw)

        client._complete_xml_fallback = wrapped


@pytest.mark.parametrize("mode", ("xml", "text"))
async def test_xml_overflow_surfaces_at_once_without_the_fallback(mode):
    client, server = make(mode, "llamacpp", Server(_bad("exceed_context_size_error tools")))
    spy = _Spy(client)
    with pytest.raises(ProviderAPIError):
        await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=9)
    assert len(server.calls) == 1 and spy.n == 0
    assert client.rejected_effort_keys == frozenset() and client.accepted_effort_keys == frozenset()
    assert client._tools_param_rejected is False


@pytest.mark.parametrize("mode", ("xml", "text"))
async def test_xml_tools_400_runs_the_fallback_and_keeps_the_effort_keys(mode):
    client, server = make(mode, "llamacpp", Server(_bad("tools not supported"), "ok"))
    spy = _Spy(client)
    await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=9)
    assert spy.n == 1 and client._tools_param_rejected is True
    assert "tools" not in server.calls[1]
    assert server.calls[1]["extra_body"] == {**THINK_OFF, "reasoning_budget_tokens": 9}
    assert client.rejected_effort_keys == frozenset()


@pytest.mark.parametrize("mode", ("xml", "text"))
async def test_xml_named_key_retries_the_xml_request_with_tools_kept(mode):
    client, server = make(mode, "llamacpp", Server(_bad("enable_thinking is not allowed"), "ok"))
    spy = _Spy(client)
    await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=9)
    assert spy.n == 0 and len(server.calls) == 2
    assert "tools" in server.calls[1]
    assert server.calls[1]["extra_body"] == {"reasoning_budget_tokens": 9}
    assert client.rejected_effort_keys == frozenset({"chat_template_kwargs"})
    assert client._tools_param_rejected is False


async def test_xml_named_key_failed_retry_restores_and_surfaces_the_original_400():
    client, server = make("xml", "llamacpp", Server(_bad("enable_thinking is not allowed"), _bad("bad roles")))
    with pytest.raises(ProviderAPIError) as info:
        await client.stream_complete(MSGS, [TOOL], disable_thinking=True)
    assert "enable_thinking" in str(info.value) and info.value.status_code == 400
    assert client.rejected_effort_keys == frozenset() and len(server.calls) == 2


@pytest.mark.parametrize("mode", ("xml", "text"))
async def test_xml_unnamed_400_runs_the_fallback_and_records_no_effort_key(mode):
    client, server = make(mode, "llamacpp", Server(_bad("unknown field 'foo'"), "ok"))
    spy = _Spy(client)
    await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=9)
    assert spy.n == 1 and "tools" not in server.calls[1]
    assert server.calls[1]["extra_body"] == {**THINK_OFF, "reasoning_budget_tokens": 9}
    assert client.rejected_effort_keys == frozenset()


async def test_xml_bare_extra_body_text_records_nothing():
    client, _ = make("xml", "llamacpp", Server(_bad("extra_body: invalid"), "ok"))
    await client.stream_complete(MSGS, [TOOL], disable_thinking=True)
    assert client.rejected_effort_keys == frozenset()


# ---------------------------------------------------------------------------
# XML fallback path: overflow and named keys only
# ---------------------------------------------------------------------------


async def test_fallback_400_naming_the_budget_retries_once_and_records_on_success():
    client, server = make(
        "xml", "llamacpp", Server(_bad("tools unsupported"), _bad("reasoning_budget_tokens unknown"), "ok")
    )
    await client.stream_complete(MSGS, [TOOL], reasoning_budget=9)
    assert len(server.calls) == 3
    assert "tools" not in server.calls[1] and "tools" not in server.calls[2]
    assert "extra_body" not in server.calls[2]
    assert client.rejected_effort_keys == frozenset({"reasoning_budget_tokens"})


@pytest.mark.parametrize("text", ["exceed_context_size_error", "unknown field 'foo'", "messages: bad roles"])
async def test_fallback_400_overflow_or_unnamed_surfaces_at_once(text):
    client, server = make("xml", "llamacpp", Server(_bad("tools unsupported"), _bad(text), "ok"))
    with pytest.raises(ProviderAPIError):
        await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=9)
    assert len(server.calls) == 2
    assert client.rejected_effort_keys == frozenset()


async def test_fallback_failed_retry_restores_and_surfaces_the_original():
    client, _ = make(
        "xml", "llamacpp",
        Server(_bad("tools unsupported"), _bad("reasoning_budget_tokens unknown"), _bad("bad roles")),
    )
    with pytest.raises(ProviderAPIError) as info:
        await client.stream_complete(MSGS, [TOOL], reasoning_budget=9)
    assert "reasoning_budget_tokens" in str(info.value)
    assert client.rejected_effort_keys == frozenset()


# ---------------------------------------------------------------------------
# The client-side cut (D-21, D-31): only where the runtime has no budget key
# ---------------------------------------------------------------------------

OVER = ["a" * 20, "b" * 21]          # 41 reasoning chars: past a 10-token budget (40 chars)
AT_LIMIT = ["a" * 20, "b" * 20]       # exactly 40: not past it


def test_the_cut_error_and_its_constant():
    assert REASONING_CHARS_PER_TOKEN == 4
    err = ProviderReasoningBudgetError(10, 41)
    assert issubclass(ProviderReasoningBudgetError, ProviderError)
    assert (err.budget, err.spent) == (10, 41)
    assert "10" in str(err) and "41" in str(err)


async def test_consumer_raises_with_budget_and_spent_and_closes_the_response_first():
    stream = _Stream(OVER)
    with pytest.raises(ProviderReasoningBudgetError) as info:
        await LLMClient._consume_native_stream(
            stream, None, reasoning_budget_chars=40, reasoning_budget_tokens=10,
        )
    assert info.value.budget == 10 and info.value.spent == 41
    assert "41" in str(info.value) and "10" in str(info.value)
    assert stream.closed is True


@pytest.mark.parametrize("mode", MODES)
async def test_vllm_stream_over_budget_is_cut(mode):
    client, server = make(mode, "vllm", Server(reasoning=OVER))
    with pytest.raises(ProviderReasoningBudgetError) as info:
        await client.stream_complete(MSGS, reasoning_budget=10)
    assert (info.value.budget, info.value.spent) == (10, 41)
    assert server.streams[0].closed is True


@pytest.mark.parametrize("mode", MODES)
async def test_vllm_stream_within_budget_completes(mode):
    client, _ = make(mode, "vllm", Server(reasoning=AT_LIMIT))
    msg, _usage = await client.stream_complete(MSGS, reasoning_budget=10)
    assert msg.content == "ok"


@pytest.mark.parametrize("mode", MODES)
async def test_llamacpp_is_never_cut_by_the_client(mode):
    client, _ = make(mode, "llamacpp", Server(reasoning=OVER))
    msg, _usage = await client.stream_complete(MSGS, reasoning_budget=10)
    assert msg.content == "ok"


@pytest.mark.parametrize("ptype", [p for p in PROVIDERS if p != "llamacpp"])
async def test_no_budget_never_cuts_however_long_the_reasoning(ptype):
    client, _ = make("native", ptype, Server(reasoning=[f"step {i}: consider item {i * 7 % 13}. " for i in range(300)]))
    msg, _usage = await client.stream_complete(MSGS, reasoning_budget=None)
    assert msg.content == "ok"


async def test_complete_non_stream_never_cuts():
    client, _ = make("native", "vllm", Server(reasoning=OVER))
    msg, _usage = await client.complete(MSGS, reasoning_budget=10)
    assert msg.content == "ok"


@pytest.mark.parametrize("mode", ("xml", "text"))
async def test_the_fallback_path_stream_passes_the_cut(mode):
    client, server = make(mode, "vllm", Server(_bad("tools unsupported"), "ok", reasoning=OVER))
    with pytest.raises(ProviderReasoningBudgetError):
        await client.stream_complete(MSGS, [TOOL], reasoning_budget=10)
    assert len(server.calls) == 2 and "tools" not in server.calls[1]


async def test_the_native_retry_send_passes_the_cut_and_the_rejection_stands():
    client, server = make(
        "native", "vllm", Server(_bad("enable_thinking is not allowed"), "ok", reasoning=OVER)
    )
    with pytest.raises(ProviderReasoningBudgetError):
        await client.stream_complete(MSGS, disable_thinking=True, reasoning_budget=10)
    assert len(server.calls) == 2 and "extra_body" not in server.calls[1]
    # the retry reached the server and streamed: the rest of the request is fine, the key is out
    assert client.rejected_effort_keys == frozenset({"chat_template_kwargs"})


async def test_the_native_tools_retry_passes_the_cut():
    client, _ = make("native", "vllm", Server(_bad("tools unsupported"), "ok", reasoning=OVER))
    with pytest.raises(ProviderReasoningBudgetError):
        await client.stream_complete(MSGS, [TOOL], reasoning_budget=10)


async def test_the_xml_retry_send_passes_the_cut():
    client, _ = make(
        "xml", "vllm", Server(_bad("enable_thinking is not allowed"), "ok", reasoning=OVER)
    )
    with pytest.raises(ProviderReasoningBudgetError):
        await client.stream_complete(MSGS, [TOOL], disable_thinking=True, reasoning_budget=10)


async def test_two_concurrent_streams_each_count_only_their_own_reasoning():
    def long_run(_kw):
        return _Stream(["a" * 20] * 3)     # 60 chars: crosses 40

    def short_run(_kw):
        return _Stream(["b" * 10] * 3)     # 30 chars: never crosses on its own

    client, _ = make("native", "vllm", Server(long_run, short_run))
    results = await asyncio.gather(
        client.stream_complete(MSGS, reasoning_budget=10),
        client.stream_complete(MSGS, reasoning_budget=10),
        return_exceptions=True,
    )
    cut = [r for r in results if isinstance(r, ProviderReasoningBudgetError)]
    done = [r for r in results if isinstance(r, tuple)]
    assert len(cut) == 1 and len(done) == 1
    assert cut[0].spent == 60   # its own three 20-char chunks, not the other stream's


async def test_the_degenerate_watch_still_works_beside_the_budget_counter():
    client, _ = make("native", "vllm", Server(reasoning=["413, 313, 213, "] * 500))
    with pytest.raises(ProviderDegenerateError):
        await client.stream_complete(MSGS, reasoning_budget=1_000_000)
