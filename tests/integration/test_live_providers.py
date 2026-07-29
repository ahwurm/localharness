"""0.10.0 provider certification: live round-trip tests for the non-vLLM providers.

Mirrors live_vllm (tests/conftest.py, tests/integration/test_spine_real_e2e.py) for the three
provider families the harness also speaks OpenAI-compat to: Ollama, llama.cpp, LM Studio. Each
gets its own opt-in marker (live_ollama / live_llamacpp / live_lmstudio) with its own boolean gate
env var (LOCALHARNESS_LIVE_OLLAMA / _LLAMACPP / _LMSTUDIO); all three resolve the actual target
through the SAME channel live_vllm uses (LOCALHARNESS_LIVE_MODEL / LOCALHARNESS_LIVE_BASE_URL) —
only one live provider is validated per invocation, so one pinned target is enough.

Skipped by default (autouse _skip_live_providers, conftest.py): the CPU-only suite never opens a
socket here. An opted-in run against a dead/unserved endpoint HARD-FAILS (_preflight, below)
rather than silently skipping or leaking a bare connection exception — same two-gate contract as
conftest.live_endpoint (env-gate skip first, reachability-gate hard-fail second).

The round trip drives the REAL provider-client path end to end — no mocks, no reimplemented
native/xml branching:
  1. LLMClient(LLMConfig(...)) + detect_capabilities() — the real capability probe; asserts a
     tool_call_mode came back.
  2. probe_served_window(base_url, model, provider_type) — the provider-aware served-window probe
     (agent/context.py's llama.cpp /props, Ollama /api/ps, LM Studio /api/v0/models branches);
     asserts an int (this runtime exposes it) or None (honest "unknowable"), never raises.
  3. A short real completion (max_tokens capped small) — proves basic generation works.
  4. A tool-call fidelity check: one real tool schema, a prompt that should invoke it, the
     response run through agent.loop._extract_tool_calls (the SAME mode-aware extractor the real
     AgentLoop uses downstream of LLMClient) — proves native/xml tool-calling round-trips for
     real, not just that a probe endpoint answered.
  5. Optional: if LOCALHARNESS_LIVE_BASE_URL_2 is also set, rebind_endpoint to a second live
     target, complete, then rebind back — exercises the cross-endpoint swap mechanism (the
     0.10.0 provider-tree feature this certification exists for) against a real second server.
     LOCALHARNESS_LIVE_MODEL_2 optionally names a different model on that endpoint; unset, it
     reuses LOCALHARNESS_LIVE_MODEL (same-model, different-server rebind). This leg is a plain
     `if`, not a skip: legs 1-4 are the required certification and still report a full PASS when
     no second endpoint is configured.

CPU-only guardrail: nothing in this module touches a socket unless its LOCALHARNESS_LIVE_* gate
is set. Never set one of those vars for the default/CI suite.
"""
from __future__ import annotations

import os

import pytest

from localharness.agent.context import probe_served_window
from localharness.agent.loop import _extract_tool_calls
from localharness.provider.client import LLMClient, LLMConfig

# One real, minimal tool: a fidelity check needs a schema a capable model will actually invoke,
# not a zero-arg no-op it might just answer in prose. Deliberately NOT the ToolRegistry (its
# dispatch/permissions machinery is out of scope here — this is a provider-client-level probe of
# "does the model call the tool", not an AgentLoop e2e; test_spine_real_e2e.py already covers the
# full-registry live path for vLLM).
_SIMPLE_TOOL = {
    "name": "get_current_time",
    "description": "Return the current wall-clock time. Call this whenever asked what time it is.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
_TOOL_PROMPT = [
    {
        "role": "user",
        "content": (
            "What time is it right now? Use the get_current_time tool to find out, then tell "
            "me the result."
        ),
    }
]
_SHORT_PROMPT = [{"role": "user", "content": "Reply with a single short word."}]


def _preflight(base_url: str, model: str, gate_env: str) -> None:
    """Reachability hard-fail, mirroring conftest.live_endpoint's vLLM-only contract: an
    opted-in-but-dead/unserved target is a REAL failure of an explicit request (pytest.fail),
    never a skip, never a leaked bare exception. The env-gate skip (autouse _skip_live_providers,
    conftest.py) always runs first, so this only executes once the caller has actually opted in.
    """
    import httpx

    try:
        resp = httpx.get(base_url.rstrip("/") + "/models", timeout=3.0)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001 — any failure to reach an opted-in endpoint is a hard fail
        pytest.fail(f"{gate_env} is set but the endpoint is unreachable at {base_url}: {e}")
    served = {m.get("id") for m in resp.json().get("data", [])}
    if model not in served:
        pytest.fail(
            f"model {model!r} is not served at {base_url} (served: {sorted(served)}). "
            "Set LOCALHARNESS_LIVE_MODEL (or _MODEL_2) to a model the endpoint actually serves."
        )


def _assert_generated(message) -> None:
    """Basic-generation proof that is correct for THINKING models. A thinking-capable local model
    can spend a small max_tokens budget entirely inside reasoning_content and return empty
    `content` with finish_reason='length' — that is a live generation, not a failure. Accept
    non-empty output in EITHER channel (content or reasoning_content)."""
    produced = (message.content or "").strip() or (getattr(message, "reasoning_content", "") or "").strip()
    assert produced != "", (
        f"no content or reasoning_content produced "
        f"(finish_reason={getattr(message, 'finish_reason', None)!r})"
    )


@pytest.mark.parametrize(
    "provider_type",
    [
        pytest.param("ollama", marks=pytest.mark.live_ollama, id="live_ollama"),
        pytest.param("llamacpp", marks=pytest.mark.live_llamacpp, id="live_llamacpp"),
        pytest.param("lmstudio", marks=pytest.mark.live_lmstudio, id="live_lmstudio"),
    ],
)
async def test_provider_round_trip(provider_type):
    """The 0.10.0 provider certification round trip — see module docstring for the 5 legs.

    One shared body, parametrized by provider_type; each case carries its OWN marker so the
    autouse _skip_live_providers gate (conftest.py) skips it independently of the other two.
    """
    from tests.conftest import LIVE_PROVIDER_GATES, live_target

    gate_env = LIVE_PROVIDER_GATES[f"live_{provider_type}"]
    model, base_url = live_target()
    _preflight(base_url, model, gate_env)

    # 1. Real client + real capability probe.
    client = LLMClient(LLMConfig(base_url=base_url, model=model, is_local=True))
    cap = await client.detect_capabilities()
    assert cap.tool_call_mode in ("native", "xml", "text")

    # 2. Provider-aware served-window probe — int (this runtime exposes it) or None (honest
    # "unknowable", never a fabricated default). Never raises.
    window = probe_served_window(base_url, model, provider_type)
    assert window is None or isinstance(window, int)

    # 3. Short real completion — proves basic generation works. Budget must clear a thinking
    # model's reasoning preamble (same reasoning as leg 4): a thinking-capable local model can
    # spend the whole cap inside reasoning_content and emit empty `content` (finish_reason=
    # 'length'). That is a live generation, so _assert_generated accepts either channel.
    client.config.max_tokens = 256
    message, _usage = await client.complete(_SHORT_PROMPT)
    _assert_generated(message)

    # 4. Tool-call fidelity check. Generous budget (mirrors detect_capabilities' own 256-token
    # probe cap, client.py): a thinking-capable local model spends real tokens on preamble/
    # reasoning before the call, and user-facing turns never set disable_thinking (#11 —
    # thinking stays on for subject/user-facing turns).
    client.config.max_tokens = 512
    tool_message, _usage2 = await client.complete(_TOOL_PROMPT, tools=[_SIMPLE_TOOL])
    calls = _extract_tool_calls(tool_message, client.config.tool_call_mode)
    assert any(c.name == "get_current_time" for c in calls), (
        f"model did not invoke get_current_time (mode={client.config.tool_call_mode}, "
        f"content={getattr(tool_message, 'content', None)!r})"
    )

    # 5. Optional: cross-endpoint rebind — the 0.10.0 provider-tree feature this certification
    # exists for. A plain `if`, not a skip: legs 1-4 above are the required round trip and still
    # count as a full pass when no second endpoint is configured.
    base_url_2 = os.environ.get("LOCALHARNESS_LIVE_BASE_URL_2")
    if base_url_2:
        model_2 = os.environ.get("LOCALHARNESS_LIVE_MODEL_2", model)
        _preflight(base_url_2, model_2, gate_env)

        client.rebind_endpoint(base_url_2)
        client.config.model = model_2
        await client.detect_capabilities()
        client.config.max_tokens = 256
        message_2, _usage3 = await client.complete(_SHORT_PROMPT)
        _assert_generated(message_2)

        # ... and back — proves the SAME client instance round-trips both directions, not just
        # a fresh one per endpoint.
        client.rebind_endpoint(base_url)
        client.config.model = model
        await client.detect_capabilities()
        message_3, _usage4 = await client.complete(_SHORT_PROMPT)
        _assert_generated(message_3)


@pytest.mark.parametrize(
    "provider_type",
    [
        pytest.param("vllm", marks=pytest.mark.live_vllm, id="live_vllm"),
        pytest.param("llamacpp", marks=pytest.mark.live_llamacpp, id="live_llamacpp"),
    ],
)
def test_count_messages_matches_usage_live(provider_type):
    """Codifies the message-level exactness bar as repeatable certification: the harness's
    whole-request count — chat template applied, tools block included — must equal the
    server's own usage.prompt_tokens for the SAME request. The original verification was a
    one-off manual script; this pins it so a tokenize-contract drift (server upgrade, payload
    regression) fails a live run instead of silently skewing every count."""
    import httpx

    from tests.conftest import LIVE_PROVIDER_GATES, live_target
    from localharness.agent.context import TokenCounter

    gate_env = LIVE_PROVIDER_GATES.get(f"live_{provider_type}", "LOCALHARNESS_LIVE_VLLM")
    model, base_url = live_target()
    _preflight(base_url, model, gate_env)

    tc = TokenCounter(base_url=base_url, model=model, provider_type=provider_type)
    assert tc._messages_exact, (
        f"{provider_type} at {base_url} answered /tokenize but not message-level counting "
        f"(vLLM messages-mode / llama.cpp /apply-template) — exactness certification failed"
    )
    tools = [{"type": "function", "function": _SIMPLE_TOOL}]
    for msgs, use_tools in ((_SHORT_PROMPT, None), (_TOOL_PROMPT, tools)):
        harness = tc.count_messages(msgs, tools=use_tools)
        payload = {"model": model, "messages": msgs, "max_tokens": 1}
        if use_tools:
            payload["tools"] = use_tools
        resp = httpx.post(base_url.rstrip("/") + "/chat/completions", json=payload, timeout=120.0)
        resp.raise_for_status()
        server = resp.json()["usage"]["prompt_tokens"]
        assert harness == server, (
            f"{provider_type}: harness count {harness} != usage.prompt_tokens {server} "
            f"(tools={bool(use_tools)})"
        )
