"""Tests for ContextManager.repair_tool_pairing and build_messages."""
import pytest
from localharness.agent.context import ContextManager


def _make_assistant_with_tool_call(tool_call_id: str, tool_name: str = "bash") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": tool_call_id, "function": {"name": tool_name, "arguments": "{}"}}],
    }


def _make_tool_result(tool_call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ---------------------------------------------------------------------------
# repair_tool_pairing
# ---------------------------------------------------------------------------

def test_repair_removes_orphaned_tool_result():
    """Tool result with no matching assistant tool_calls entry is removed."""
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "do something"},
        _make_tool_result("orphan-id"),  # no preceding assistant with tool_calls
    ]
    repaired = cm.repair_tool_pairing(messages)
    assert all(m.get("role") != "tool" for m in repaired)


def test_repair_keeps_valid_pairs():
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "go"},
        _make_assistant_with_tool_call("tc-1"),
        _make_tool_result("tc-1"),
    ]
    repaired = cm.repair_tool_pairing(messages)
    tool_msgs = [m for m in repaired if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "tc-1"


def test_repair_handles_empty_message_list():
    cm = ContextManager()
    assert cm.repair_tool_pairing([]) == []


@pytest.mark.asyncio
async def test_build_messages_returns_copy_not_original():
    cm = ContextManager()
    original = [{"role": "user", "content": "hello"}]
    result, budget = await cm.build_messages(original)
    assert result is not original
    assert result == original


@pytest.mark.asyncio
async def test_build_messages_calls_repair_internally():
    """build_messages should strip orphaned tool results via repair_tool_pairing."""
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "go"},
        _make_tool_result("orphan"),
    ]
    result, budget = await cm.build_messages(messages)
    assert all(m.get("role") != "tool" for m in result)


@pytest.mark.asyncio
async def test_build_messages_normalizes_none_content_to_empty_string():
    """Wire safety: vLLM's request validation rejects content:None ('Input should
    be a valid string' -> HTTP 400). Reasoning-parser tool turns legitimately
    produce assistant content=None, and persisted/legacy history may carry it for
    any role — build_messages must never let None reach the payload (symmetric:
    assistant/tool/user), while string content and tool_calls pass untouched."""
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "go"},
        _make_assistant_with_tool_call("tc-1"),                      # content: None
        {"role": "tool", "tool_call_id": "tc-1", "content": None},   # symmetric case
        {"role": "assistant", "content": "plain reply"},
    ]
    result, _budget = await cm.build_messages(messages)
    assert all(m.get("content") is not None for m in result)
    asst = next(m for m in result if m.get("tool_calls"))
    assert asst["content"] == ""                                     # normalized
    assert asst["tool_calls"][0]["id"] == "tc-1"                     # tool_calls intact
    assert result[-1]["content"] == "plain reply"                    # strings untouched


# --- Phase 4: TokenCounter + TokenBudget ---

def test_token_counter_tiktoken():
    """TokenCounter produces non-zero count for non-empty text."""
    from localharness.agent.context import TokenCounter
    tc = TokenCounter()
    count = tc.count("Hello, world! This is a test sentence.")
    assert count > 0
    assert count < 100  # sanity


def test_token_counter_messages():
    from localharness.agent.context import TokenCounter
    tc = TokenCounter()
    msgs = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    count = tc.count_messages(msgs)
    assert count > 10  # at least the text content
    assert count < 200  # sanity


def test_token_counter_remote_unreachable_falls_back_to_approximate():
    """#8: base_url+model whose /tokenize is unreachable must NOT hard-fail construction.
    Capability detection: an absent/foreign /tokenize disables the exact remote path and
    falls back to the approximate cl100k meter, so `start` works against Ollama/LM Studio/
    llama.cpp instead of exiting 1. The exact path is only for runtimes that actually serve
    vLLM's /tokenize contract."""
    from localharness.agent.context import TokenCounter
    tc = TokenCounter(base_url="http://127.0.0.1:1/v1", model="nope")  # nothing listening
    assert tc._tokenize_url is None  # remote exact path disabled by capability detection
    assert tc.count("hello world this is approximate") > 0  # tiktoken fallback still counts


def test_token_counter_remote_path_and_cache(monkeypatch):
    """When /tokenize works, count() uses the server count and caches by content hash."""
    from localharness.agent import context as ctxmod
    calls = {"n": 0}

    def fake_remote(self, text):
        calls["n"] += 1
        return 999  # server-truth count, distinct from any tiktoken value

    monkeypatch.setattr(ctxmod.TokenCounter, "_remote_count", fake_remote)
    monkeypatch.setattr(ctxmod.TokenCounter, "_remote_count_messages", lambda self, m, t: 7)
    tc = ctxmod.TokenCounter(base_url="http://localhost:8000/v1", model="qwen")
    assert tc._tokenize_url == "http://localhost:8000/tokenize"  # /v1 stripped, root path
    n1 = calls["n"]  # probe consumed one call
    assert tc.count("some digits 123456") == 999
    assert tc.count("some digits 123456") == 999  # cached
    assert calls["n"] == n1 + 1  # exactly one new remote call (second was cached)


def test_token_counter_remote_count_strips_v1_and_reads_count(monkeypatch):
    """_remote_count POSTs to {root}/tokenize and reads the 'count' field."""
    from localharness.agent import context as ctxmod

    captured = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"count": 42}'

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = req.data
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    tc = ctxmod.TokenCounter(base_url="http://localhost:8000/v1", model="qwen")
    assert captured["url"] == "http://localhost:8000/tokenize"
    assert tc.count("x") == 42


# --- #8: per-runtime /tokenize capability detection (fakes mirror real API shapes) ---


def _patch_urlopen(monkeypatch, handler):
    """Route TokenCounter's urllib probe through `handler(url) -> bytes` (200) or a raised error."""
    import urllib.request
    from localharness.agent import context as ctxmod  # noqa: F401

    class _Resp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload

    def fake_urlopen(req, timeout=0):
        return _Resp(handler(req.full_url))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_token_counter_vllm_ok_uses_exact_no_warning(monkeypatch, caplog):
    """vLLM serves /tokenize -> {count}. Capability detection keeps the exact path; no fallback log."""
    import logging
    from localharness.agent.context import TokenCounter
    _patch_urlopen(monkeypatch, lambda url: b'{"count": 3}')
    with caplog.at_level(logging.WARNING, logger="localharness.agent.context"):
        tc = TokenCounter(base_url="http://localhost:8000/v1", model="qwen")
    assert tc._tokenize_url == "http://localhost:8000/tokenize"  # exact path live
    assert tc.count("abc") == 3
    assert not [r for r in caplog.records if "approximate" in r.getMessage().lower()]


def test_token_counter_lmstudio_404_falls_back_to_approximate(monkeypatch):
    """LM Studio (OpenAI-compatible) 404s on /tokenize -> approximate fallback, no raise."""
    import urllib.error
    import urllib.request
    from localharness.agent.context import TokenCounter

    def raise_404(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_404)
    tc = TokenCounter(base_url="http://localhost:1234/v1", model="lmstudio-model")
    assert tc._tokenize_url is None  # foreign/absent -> exact path disabled
    assert tc.count("some tokens here") > 0  # tiktoken meter


def test_token_counter_ollama_no_tokenize_falls_back_to_approximate(monkeypatch):
    """Ollama has no tokenize API (/api/* only) -> the /tokenize probe errors -> approximate."""
    import urllib.error
    import urllib.request
    from localharness.agent.context import TokenCounter

    def refuse(req, timeout=0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    tc = TokenCounter(base_url="http://localhost:11434/v1", model="qwen2.5")
    assert tc._tokenize_url is None
    assert tc.count("ollama runs approximate") > 0


def test_token_counter_unknown_provider_falls_through_to_llamacpp_shape(monkeypatch):
    """#8 harvest: a 200 with llama.cpp's {tokens:[...]} body (no `count`) is a VALID exact
    shape, NOT a foreign one. With no provider_type the counter tries the vLLM shape, finds no
    `count`, then falls THROUGH to the llama.cpp shape and counts EXACTLY (len tokens) — it must
    NOT degrade to the approximate meter (doctor already reports llama.cpp /tokenize as exact)."""
    from localharness.agent.context import TokenCounter
    _patch_urlopen(monkeypatch, lambda url: b'{"tokens": [7, 8]}')
    tc = TokenCounter(base_url="http://localhost:8080/v1", model="llamacpp-model")
    assert tc._mode == "llamacpp"        # locked onto the exact tokens-shape
    assert tc._tokenize_url is not None  # exact path LIVE, not disabled
    assert tc.count("xy") == 2           # len(tokens), not a cl100k estimate
    assert not tc.approximate


def _patch_llama_server(monkeypatch, captured):
    """Fake a full llama-server: /tokenize {content}->{tokens}, /apply-template
    {messages}->{prompt}. Records every (url, body) in `captured` (a list)."""
    import json
    import urllib.request

    class _Resp:
        def __init__(self, payload): self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload

    def fake_urlopen(req, timeout=0):
        captured.append((req.full_url, req.data))
        if req.full_url.endswith("/apply-template"):
            body = json.loads(req.data.decode())
            rendered = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                               for m in body["messages"])
            if body.get("tools"):
                rendered = "TOOLS-BLOCK\n" + rendered
            return _Resp(json.dumps({"prompt": rendered}).encode())
        return _Resp(b'{"tokens": [1, 2, 3]}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_token_counter_llamacpp_shape(monkeypatch):
    """provider_type=llamacpp POSTs {"content"} and counts len(tokens) — llama-server returns
    {"tokens": [...]} with NO "count" field (verified against llama.cpp's /tokenize contract).
    Message-level capability locks on via /apply-template."""
    from localharness.agent import context as ctxmod

    captured: list = []
    _patch_llama_server(monkeypatch, captured)
    tc = ctxmod.TokenCounter(
        base_url="http://localhost:8080/v1", model="qwen", provider_type="llamacpp"
    )
    url0, body0 = captured[0]  # content-shape probe
    assert url0 == "http://localhost:8080/tokenize"
    assert b"content" in body0 and b"prompt" not in body0
    assert tc._messages_exact  # /apply-template answered the capability probe
    assert tc.count("hello world") == 3
    assert not tc.approximate


def test_token_counter_dead_server_raises_connection_error_with_cause(monkeypatch):
    """Server killed mid-session (live 2026-08-10: llama-server took two SIGINTs mid-turn):
    count() must fail loud as ProviderConnectionError NAMING the transport cause, so the agent
    loop renders "model server unreachable" calmly instead of an internal-error traceback —
    and never substitutes an approximate count. A server that ANSWERS badly (HTTP 500) is a
    contract failure, not unreachability: that stays RuntimeError."""
    import urllib.error
    import urllib.request
    from localharness.agent import context as ctxmod
    from localharness.provider.client import ProviderConnectionError

    captured: list = []
    _patch_llama_server(monkeypatch, captured)
    tc = ctxmod.TokenCounter(
        base_url="http://localhost:8080/v1", model="qwen", provider_type="llamacpp"
    )

    def refused(req, timeout=0):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises(ProviderConnectionError) as ei:
        tc.count("fresh text the probe never cached")
    assert "unreachable" in str(ei.value)
    assert "Connection refused" in str(ei.value)

    def erroring(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", erroring)
    with pytest.raises(RuntimeError) as ei2:  # would NOT catch ProviderConnectionError
        tc.count("another fresh text, also uncached")
    assert "HTTPError" in str(ei2.value)


def test_token_counter_llamacpp_count_messages_two_hop(monkeypatch):
    """llamacpp count_messages = /apply-template (renders the server's own template, tools
    included) then /tokenize {content, add_special:true} — the chat path tokenizes its render
    with add_special=true (server-context.cpp tokenize_input_prompts(..., true, true)), so
    BOS-adding models count identically by construction (parity with usage.prompt_tokens
    verified live: 20==20 plain / 160==160 with tools). Result is whole-list cached."""
    import json
    from localharness.agent import context as ctxmod

    captured: list = []
    _patch_llama_server(monkeypatch, captured)
    tc = ctxmod.TokenCounter(
        base_url="http://localhost:8080/v1", model="qwen", provider_type="llamacpp"
    )
    captured.clear()
    msgs = [{"role": "user", "content": "Say pong."}]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    assert tc.count_messages(msgs, tools=tools) == 3  # len(tokens) of the rendered prompt
    apply_url, apply_body = captured[0]
    tok_url, tok_body = captured[1]
    assert apply_url.endswith("/apply-template")
    sent = json.loads(apply_body.decode())
    assert sent["messages"] == msgs and sent["tools"] == tools
    tok_sent = json.loads(tok_body.decode())
    assert tok_sent["add_special"] is True  # mirrors the chat path's tokenize flags
    assert "TOOLS-BLOCK" in tok_sent["content"]  # tools rendered into the counted prompt
    n_calls = len(captured)
    assert tc.count_messages(msgs, tools=tools) == 3  # cached — no new network
    assert len(captured) == n_calls


def _patch_vllm_server(monkeypatch, captured, messages_count=50, fail_messages=False):
    """Fake a vLLM /tokenize that answers BOTH contracts: {model,prompt}->{count:3} and
    messages-mode {model,messages,...}->{count:messages_count} (400 when fail_messages)."""
    import json
    import urllib.error
    import urllib.request

    class _Resp:
        def __init__(self, payload): self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload

    def fake_urlopen(req, timeout=0):
        captured.append((req.full_url, req.data))
        body = json.loads(req.data.decode())
        if "messages" in body:
            if fail_messages:
                raise urllib.error.HTTPError(req.full_url, 400, "no messages mode", {}, None)
            return _Resp(json.dumps({"count": messages_count}).encode())
        return _Resp(b'{"count": 3}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_token_counter_vllm_count_messages_exact(monkeypatch):
    """vLLM count_messages POSTs the messages-mode /tokenize payload — model, messages,
    add_generation_prompt, and the SAME wire-format tools the real call carries — and returns
    the server's template-rendered count (verified live == usage.prompt_tokens: 22/282/24
    incl. the tools block). Whole-list cached."""
    import json
    from localharness.agent import context as ctxmod

    captured: list = []
    _patch_vllm_server(monkeypatch, captured)
    tc = ctxmod.TokenCounter(base_url="http://localhost:8000/v1", model="qwen", provider_type="vllm")
    assert tc._messages_exact
    captured.clear()
    msgs = [{"role": "user", "content": "Say pong."}]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    assert tc.count_messages(msgs, tools=tools) == 50
    sent = json.loads(captured[0][1].decode())
    assert sent["model"] == "qwen" and sent["messages"] == msgs
    assert sent["add_generation_prompt"] is True and sent["tools"] == tools
    assert tc.count_messages(msgs, tools=tools) == 50  # cached
    assert len(captured) == 1
    # A DIFFERENT tools list is a different request — never served from the tools-less cache.
    assert tc.count_messages(msgs) == 50 and len(captured) == 2
    # An empty list never dials (a template render of [] is a server-side error, not a count).
    assert tc.count_messages([]) == 0 and len(captured) == 2


def test_token_counter_vllm_messages_probe_failure_degrades_to_summed(monkeypatch, caplog):
    """A vLLM build without messages-mode /tokenize keeps EXACT content counting but degrades
    count_messages to the summed per-message estimate (+ serialized tools), warned once —
    never a hard failure at rebind for a content-exact server."""
    import logging
    from localharness.agent import context as ctxmod

    captured: list = []
    _patch_vllm_server(monkeypatch, captured, fail_messages=True)
    with caplog.at_level(logging.WARNING, logger="localharness.agent.context"):
        tc = ctxmod.TokenCounter(
            base_url="http://localhost:8000/v1", model="qwen", provider_type="vllm"
        )
    assert not tc._messages_exact
    assert [r for r in caplog.records if "message-level" in r.getMessage()]
    assert not tc.approximate  # content counting is still exact
    msgs = [{"role": "user", "content": "hi"}]
    base = tc.count_messages(msgs)
    assert base == 4 + 3  # +4 overhead + server content count (3)
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    assert tc.count_messages(msgs, tools=tools) > base  # tools estimated into the sum


def test_token_counter_llamacpp_apply_template_absent_degrades(monkeypatch, caplog):
    """An older llama-server without /apply-template (404) keeps EXACT content counting but
    degrades count_messages to the summed estimate, warned once — the llamacpp shape's OWN
    probe-failure path, independent of the vLLM one."""
    import json
    import logging
    import urllib.error
    import urllib.request
    from localharness.agent import context as ctxmod

    class _Resp:
        def __init__(self, payload): self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload

    def fake_urlopen(req, timeout=0):
        if req.full_url.endswith("/apply-template"):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        return _Resp(b'{"tokens": [1, 2, 3]}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with caplog.at_level(logging.WARNING, logger="localharness.agent.context"):
        tc = ctxmod.TokenCounter(
            base_url="http://localhost:8080/v1", model="qwen", provider_type="llamacpp"
        )
    assert tc._mode == "llamacpp" and not tc._messages_exact
    assert [r for r in caplog.records if "message-level" in r.getMessage()]
    assert not tc.approximate  # content path still exact
    assert tc.count("abc") == 3
    assert tc.count_messages([{"role": "user", "content": "hi"}]) == 4 + 3  # summed estimate


def test_successful_rebind_clears_message_cache(monkeypatch):
    """A /model swap must invalidate the whole-list count_messages cache — a count computed
    under the OLD model's tokenizer is wrong for the new one. (The list-level cache is new in
    the exact-mode commit; count()'s content cache was already covered.)"""
    import json
    import urllib.request
    from localharness.agent import context as ctxmod

    counts = {"messages": 50}

    class _Resp:
        def __init__(self, payload): self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data.decode())
        n = counts["messages"] if "messages" in body else 3
        return _Resp(json.dumps({"count": n}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    tc = ctxmod.TokenCounter(base_url="http://localhost:8000/v1", model="model-a", provider_type="vllm")
    msgs = [{"role": "user", "content": "hi"}]
    assert tc.count_messages(msgs) == 50
    counts["messages"] = 60
    assert tc.count_messages(msgs) == 50  # cached under model-a
    tc.rebind(base_url="http://localhost:8000/v1", model="model-b", provider_type="vllm")
    assert tc.count_messages(msgs) == 60  # cache cleared — recounted under model-b


def test_token_counter_count_messages_mid_session_failure_raises(monkeypatch):
    """Once message-level exact is locked on, a mid-session miss fails LOUD (same contract as
    count()) — never a silent substitution of an approximate number. A transport-level miss
    ("server went away") is typed ProviderConnectionError so the loop surfaces it as the calm
    llm_error path, naming the cause — still loud, better addressed."""
    import pytest
    import urllib.error
    import urllib.request
    from localharness.agent import context as ctxmod
    from localharness.provider.client import ProviderConnectionError

    captured: list = []
    _patch_vllm_server(monkeypatch, captured)
    tc = ctxmod.TokenCounter(base_url="http://localhost:8000/v1", model="qwen", provider_type="vllm")
    assert tc._messages_exact

    def refuse(req, timeout=0):
        raise urllib.error.URLError("server went away")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(ProviderConnectionError, match="unreachable"):
        tc.count_messages([{"role": "user", "content": "fresh, uncached"}])


def test_token_counter_malformed_count_degrades_not_crashes(monkeypatch):
    """A non-conformant server returning 200 with a non-numeric 'count' must read as "no
    exact source" — the probe fails with the CONTRACTED RuntimeError (start_cmd catches it) —
    never a raw TypeError from int() that skips the `except RuntimeError` handler."""
    from localharness.agent.context import TokenCounter

    _patch_urlopen(monkeypatch, lambda url: b'{"count": ["not", "a", "number"]}')
    with pytest.raises(RuntimeError, match="exact token counting unavailable"):
        TokenCounter(base_url="http://localhost:8000/v1", model="m", provider_type="vllm")


def test_count_messages_offline_estimate_includes_tools():
    """The offline/off-mode estimator adds the serialized tools block — the loop's no-usage
    fallback passes the wire tools, and ignoring them would undercount by the largest
    structural cost in the request."""
    from localharness.agent.context import TokenCounter

    tc = TokenCounter()
    msgs = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "get_weather",
              "description": "Get weather for a city", "parameters": {"type": "object"}}}]
    assert tc.count_messages(msgs, tools=tools) > tc.count_messages(msgs)


@pytest.mark.asyncio
async def test_build_messages_never_routes_fragments_through_exact_renderer(monkeypatch):
    """LIVE-CAUGHT regression (the real-turn smoke): build_messages counts tool schemas as a
    system-only pseudo-message, and compaction slices history into fragments — chat templates
    REJECT non-conversations (Qwen3.6 raises 'No user query found' → vLLM 400), and
    count_messages' fail-loud contract turned that into a crashed turn on the FIRST question.
    The budget spine must run on estimate_messages (fragment-safe, never renders); the exact
    renderer is for complete wire requests only."""
    from localharness.agent import context as ctxmod
    from localharness.tools.base import ToolSchema

    captured: list = []
    _patch_vllm_server(monkeypatch, captured)
    tc = ctxmod.TokenCounter(base_url="http://localhost:8000/v1", model="qwen", provider_type="vllm")
    assert tc._messages_exact

    def boom(self, messages, tools):
        raise AssertionError("budget spine dialed the exact message renderer on a fragment")
    monkeypatch.setattr(ctxmod.TokenCounter, "_remote_count_messages", boom)

    cm = ctxmod.ContextManager(token_counter=tc)
    msgs = [{"role": "user", "content": "What is the capital of South Korea?"}]
    schemas = [ToolSchema(name="get_weather", description="d", parameters={"type": "object"})]
    request, budget = await cm.build_messages(msgs, schemas)
    assert request and request[0]["role"] == "user"
    assert budget.tool_schema_tokens > 0  # schemas were still costed (via the estimator)


def test_estimate_messages_fragment_safe_in_exact_mode(monkeypatch):
    """estimate_messages never dials the message renderer even when message-level exact is
    live — system-only and tool-only fragments (no template accepts them) count structurally,
    with per-content counts still server-exact."""
    import json as _j
    from localharness.agent import context as ctxmod

    captured: list = []
    _patch_vllm_server(monkeypatch, captured)
    tc = ctxmod.TokenCounter(base_url="http://localhost:8000/v1", model="qwen", provider_type="vllm")
    captured.clear()
    assert tc.estimate_messages([{"role": "system", "content": "tool schema blob"}]) == 4 + 3
    assert tc.estimate_messages([{"role": "tool", "content": "orphan result"}]) == 4 + 3
    assert captured, "content counts still go through the exact /tokenize"
    assert all("messages" not in _j.loads(body.decode()) for _, body in captured)


def test_token_counter_exact_local_count_messages_ignores_tools(monkeypatch):
    """exact_local (ollama/lmstudio GGUF path) renders message structure exactly but does NOT
    render a tools block (per-runtime tool templating is unverified — Ollama re-renders with
    its own Go template); `tools` must not change the count NOR poison the cache."""
    from localharness.agent import gguf_tokenizer as gguf_mod
    from localharness.agent.context import TokenCounter

    monkeypatch.setattr(gguf_mod, "load_gguf_tokenizer", lambda *a, **k: _FakeGguf())
    tc = TokenCounter(base_url="http://127.0.0.1:11434/v1", model="q", provider_type="ollama")
    msgs = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    assert tc.count_messages(msgs) == 999
    assert tc.count_messages(msgs, tools=tools) == 999
    assert tc.estimate_messages(msgs) == 999  # spine delegates to the GGUF render too
    assert tc._gguf.message_calls == 3  # each call really delegated (no stale cache serving)


def test_token_counter_ollama_approximate_mode_never_dials(monkeypatch):
    """ollama serves no tokenize API: explicit approximate mode, ZERO network. Port 1 would
    hard-fail any probe — constructing against it proves nothing is dialed. Counts are the
    plain cl100k estimate inflated by the safety factor (over-count = the safe direction).
    load_gguf_tokenizer is stubbed to None so this stays hermetic on a box with a real Ollama
    install (no ~/.ollama filesystem scan) and exercises the approximate FALLBACK path."""
    import math

    from localharness.agent import gguf_tokenizer as gguf_mod
    from localharness.agent.context import APPROX_TOKENIZE_SAFETY_FACTOR, TokenCounter
    monkeypatch.setattr(gguf_mod, "load_gguf_tokenizer", lambda *a, **k: None)
    tc = TokenCounter(base_url="http://127.0.0.1:1/v1", model="m", provider_type="ollama")
    assert tc.approximate
    plain = TokenCounter()  # offline estimator baseline (no inflation)
    text = "some digits 123456 and code()"
    assert tc.count(text) == math.ceil(plain.count(text) * APPROX_TOKENIZE_SAFETY_FACTOR)


def test_token_counter_lmstudio_approximate_mode(monkeypatch):
    """LM Studio documents no tokenize endpoint on either API surface — same policy as ollama.
    load_gguf_tokenizer is stubbed to None so this stays hermetic on a box with a real LM
    Studio install (no `lms ls` shell-out / ~/.lmstudio scan) and exercises the approximate
    FALLBACK path only."""
    from localharness.agent import gguf_tokenizer as gguf_mod
    from localharness.agent.context import TokenCounter
    monkeypatch.setattr(gguf_mod, "load_gguf_tokenizer", lambda *a, **k: None)
    tc = TokenCounter(base_url="http://127.0.0.1:1/v1", model="m", provider_type="lmstudio")
    assert tc.approximate
    assert tc.count("hello") >= TokenCounter().count("hello")


class _FakeGguf:
    """Deterministic stand-in for GgufTokenizer — never touches llama_cpp."""
    def __init__(self):
        self.message_calls = 0

    def count(self, text):
        return len(text.split())

    def count_messages(self, messages):
        self.message_calls += 1
        return 999


def test_token_counter_exact_local_when_gguf_resolves_ollama(monkeypatch):
    """When load_gguf_tokenizer resolves a GGUF for an ollama-served model, TokenCounter locks
    onto mode 'exact_local' and delegates count()/count_messages() to it — not approximate."""
    from localharness.agent import gguf_tokenizer as gguf_mod
    from localharness.agent.context import TokenCounter

    monkeypatch.setattr(gguf_mod, "load_gguf_tokenizer", lambda *a, **k: _FakeGguf())
    tc = TokenCounter(base_url="http://127.0.0.1:11434/v1", model="qwen2.5:7b", provider_type="ollama")
    assert tc._mode == "exact_local"
    assert tc.approximate is False
    assert tc.count("a b c") == 3
    assert tc.count_messages([{"role": "user", "content": "hi there"}]) == 999


def test_token_counter_exact_local_when_gguf_resolves_lmstudio(monkeypatch):
    """Same exact_local lock-on for LM Studio."""
    from localharness.agent import gguf_tokenizer as gguf_mod
    from localharness.agent.context import TokenCounter

    monkeypatch.setattr(gguf_mod, "load_gguf_tokenizer", lambda *a, **k: _FakeGguf())
    tc = TokenCounter(
        base_url="http://127.0.0.1:1234/v1", model="qwen2.5-0.5b-instruct", provider_type="lmstudio"
    )
    assert tc._mode == "exact_local"
    assert tc.approximate is False
    assert tc.count("a b c") == 3
    assert tc.count_messages([{"role": "user", "content": "hi there"}]) == 999


def test_token_counter_exact_local_falls_back_when_no_gguf(monkeypatch):
    """load_gguf_tokenizer returning None (no local GGUF reachable) degrades to the labeled
    approximate estimator — never a silent guess, and never exact_local."""
    from localharness.agent import gguf_tokenizer as gguf_mod
    from localharness.agent.context import TokenCounter

    monkeypatch.setattr(gguf_mod, "load_gguf_tokenizer", lambda *a, **k: None)
    tc = TokenCounter(base_url="http://127.0.0.1:11434/v1", model="qwen2.5:7b", provider_type="ollama")
    assert tc._mode == "approximate"
    assert tc.approximate is True


def test_token_counter_exact_local_survives_rebind(monkeypatch):
    """A /model swap between two ollama-served models both backed by a local GGUF stays
    exact_local across rebind (the fake gguf identity is preserved, not re-derived from stale
    state); a swap to a model with no resolvable GGUF correctly falls to approximate — proving
    _rebind_probe's unconditional `self._gguf = None` reset doesn't leak the prior tokenizer."""
    from localharness.agent import gguf_tokenizer as gguf_mod
    from localharness.agent.context import TokenCounter

    fake = _FakeGguf()
    monkeypatch.setattr(gguf_mod, "load_gguf_tokenizer", lambda *a, **k: fake)
    tc = TokenCounter(base_url="http://127.0.0.1:11434/v1", model="qwen2.5:7b", provider_type="ollama")
    assert tc._mode == "exact_local"

    tc.rebind(base_url="http://127.0.0.1:11434/v1", model="other:tag", provider_type="ollama")
    assert tc._mode == "exact_local"
    assert tc._gguf is fake

    monkeypatch.setattr(gguf_mod, "load_gguf_tokenizer", lambda *a, **k: None)
    tc.rebind(base_url="http://127.0.0.1:11434/v1", model="no-gguf:tag", provider_type="ollama")
    assert tc._mode == "approximate"
    assert tc._gguf is None


def test_token_counter_llamacpp_unreachable_fails_loud():
    """llama.cpp HAS a tokenize endpoint — an unreachable one stays a HARD error (doctor
    reports the same failure), never a silent approximate meter for a should-be-exact runtime."""
    import pytest

    from localharness.agent.context import TokenCounter
    with pytest.raises(RuntimeError, match="exact token counting unavailable"):
        TokenCounter(base_url="http://127.0.0.1:1/v1", model="m", provider_type="llamacpp")


def test_token_counter_fallback_logs_single_approximate_warning(monkeypatch, caplog):
    """The fallback emits exactly ONE clear log line stating approximate counting is in effect."""
    import logging
    import urllib.error
    import urllib.request
    from localharness.agent.context import TokenCounter

    def refuse(req, timeout=0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with caplog.at_level(logging.WARNING, logger="localharness.agent.context"):
        TokenCounter(base_url="http://localhost:11434/v1", model="qwen2.5")
    approx = [r for r in caplog.records if "approximate" in r.getMessage().lower()]
    assert len(approx) == 1, [r.getMessage() for r in caplog.records]


def test_token_budget_usage_fraction():
    from localharness.agent.context import TokenBudget
    budget = TokenBudget(total_limit=100_000, current_usage=70_000, tool_schema_tokens=10_000)
    assert budget.usage_fraction == pytest.approx(0.80, abs=0.01)
    assert budget.needs_summary_compact is True
    assert budget.needs_full_compact is False


def test_token_budget_below_threshold():
    from localharness.agent.context import TokenBudget
    budget = TokenBudget(total_limit=100_000, current_usage=50_000, tool_schema_tokens=5_000)
    assert budget.usage_fraction == pytest.approx(0.55, abs=0.01)
    assert budget.needs_summary_compact is False


# --- Phase 4: CompactionPipeline ---

def test_tool_result_cap_keeps_head_and_tail():
    # Oversized result: head+tail keep must preserve BOTH ends — the tail (where exit codes /
    # errors live) is exactly what the old head-only truncation discarded.
    from localharness.agent.context import ToolResultCapStage, TokenBudget, TokenCounter
    stage = ToolResultCapStage(max_chars=100)
    content = "H" * 100 + "T" * 100  # head-only would drop every T
    messages = [{"role": "tool", "tool_call_id": "tc-1", "content": content}]
    budget = TokenBudget(total_limit=128000, current_usage=1000, tool_schema_tokens=0)
    result, modified = stage.apply(messages, budget, TokenCounter())
    out = result[0]["content"]
    assert modified is True
    assert out.startswith("H" * 60)   # 60% head
    assert out.endswith("T" * 40)     # 40% tail preserved
    assert "elided" in out


def test_tool_result_cap_strips_ansi_and_whitespace():
    from localharness.agent.context import ToolResultCapStage, TokenBudget, TokenCounter
    stage = ToolResultCapStage(max_chars=10_000)
    content = "\x1b[31mERROR\x1b[0m   \n\n\n\ndone"
    messages = [{"role": "tool", "tool_call_id": "tc-1", "content": content}]
    budget = TokenBudget(total_limit=128000, current_usage=1000, tool_schema_tokens=0)
    result, modified = stage.apply(messages, budget, TokenCounter())
    out = result[0]["content"]
    assert modified is True
    assert "\x1b" not in out             # ANSI stripped
    assert out == "ERROR\n\ndone"        # trailing ws dropped, blank-line run → one blank line


def test_tool_result_cap_clean_can_avoid_truncation():
    # A result over cap only because of ANSI/whitespace cleans back under cap → no elision.
    from localharness.agent.context import ToolResultCapStage, TokenBudget, TokenCounter
    stage = ToolResultCapStage(max_chars=20)
    content = "abc" + "\x1b[0m" * 20 + "xyz"  # >20 raw, ~6 once stripped
    messages = [{"role": "tool", "tool_call_id": "tc-1", "content": content}]
    budget = TokenBudget(total_limit=128000, current_usage=1000, tool_schema_tokens=0)
    result, modified = stage.apply(messages, budget, TokenCounter())
    out = result[0]["content"]
    assert modified is True
    assert out == "abcxyz"
    assert "elided" not in out


def test_tool_result_cap_no_op_when_short():
    from localharness.agent.context import ToolResultCapStage, TokenBudget, TokenCounter
    stage = ToolResultCapStage(max_chars=100)
    messages = [{"role": "tool", "tool_call_id": "tc-1", "content": "short"}]
    budget = TokenBudget(total_limit=128000, current_usage=1000, tool_schema_tokens=0)
    result, modified = stage.apply(messages, budget, TokenCounter())
    assert modified is False


@pytest.mark.asyncio
async def test_summary_compaction_fires_at_80_pct():
    from localharness.agent.context import SummaryCompactionStage, TokenBudget, TokenCounter
    async def mock_summarize(msgs):
        return "Summary of middle messages"
    stage = SummaryCompactionStage(preserve_first_n=2, preserve_last_n=2, llm_summarize_fn=mock_summarize)
    # Build messages: 2 preserved first + 6 middle + 2 preserved last = 10
    messages = [{"role": "system", "content": "sys"}]
    messages.append({"role": "user", "content": "task"})
    for i in range(6):
        messages.append({"role": "assistant", "content": f"response {i}"})
    messages.append({"role": "user", "content": "recent"})
    messages.append({"role": "assistant", "content": "latest"})
    budget = TokenBudget(total_limit=100_000, current_usage=82_000, tool_schema_tokens=0)
    result, modified = await stage.apply(messages, budget, TokenCounter())
    assert modified is True
    assert len(result) < len(messages)
    # Summary message should be present
    assert any("[Context Summary]" in (m.get("content") or "") for m in result)


@pytest.mark.asyncio
async def test_summary_compaction_skips_below_80_pct():
    from localharness.agent.context import SummaryCompactionStage, TokenBudget, TokenCounter
    async def mock_summarize(msgs):
        return "Should not be called"
    stage = SummaryCompactionStage(preserve_first_n=2, preserve_last_n=2, llm_summarize_fn=mock_summarize)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    budget = TokenBudget(total_limit=100_000, current_usage=50_000, tool_schema_tokens=0)
    result, modified = await stage.apply(messages, budget, TokenCounter())
    assert modified is False
    assert result == messages


@pytest.mark.asyncio
async def test_compaction_pipeline_preserves_tool_pairs():
    from localharness.agent.context import CompactionPipeline, TokenBudget, TokenCounter
    async def mock_summarize(msgs):
        return "Summarized"
    tc = TokenCounter()
    pipeline = CompactionPipeline(token_counter=tc, preserve_first_n=2, preserve_last_n=2, llm_summarize_fn=mock_summarize)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc-1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc-1", "content": "result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "latest"},
    ]
    budget = TokenBudget(total_limit=100_000, current_usage=82_000, tool_schema_tokens=0)
    result, modified = await pipeline.run(messages, budget)
    # No orphaned tool messages in result
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    for tm in tool_msgs:
        tc_id = tm.get("tool_call_id")
        assert any(
            tc_id in [tc.get("id") for tc in (m.get("tool_calls") or [])]
            for m in result if m.get("role") == "assistant"
        )


# --- MOVE 0c (coordinator ruling 2026-07-06): compact-to-target + per-turn fire cap +
# emergency hard floor. The earlier must-shrink LATCH is REMOVED: shrink-per-fire is often tiny
# near the trigger (SEMA-05 sawtooth 82→70→80→81→73), so the latch switched compaction OFF
# mid-long-turn and a live run overflowed to 101.1% utilization (designed-20260706T143144Z,
# killed). Overflow must be impossible; compaction is never latched off while over budget. ---

def _compactible_msgs():
    """system + 4 bulky middle messages + 1 recent — a compactible middle of 3 (preserve 1/1)."""
    big = "word " * 200
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "recent short tail"},
    ]


def _guard_cm(summarize_fn):
    """A ContextManager whose usage sits at ~85% (in [0.80, 0.95): summary-compaction fires,
    the 95% full-auto path does NOT — so exactly one summarize call per fire)."""
    from localharness.agent.context import CompactionPipeline, ContextManager, TokenCounter
    tc = TokenCounter()
    msgs = _compactible_msgs()
    n = tc.count_messages(msgs)
    pipeline = CompactionPipeline(tc, preserve_first_n=1, preserve_last_n=1, llm_summarize_fn=summarize_fn)
    cm = ContextManager(max_context_tokens=int(n / 0.85), pipeline=pipeline, token_counter=tc)
    return cm, msgs


@pytest.mark.asyncio
async def test_long_turn_with_bloating_compaction_never_exceeds_100_pct():
    """Ruling test (a): overflow must be IMPOSSIBLE. Worst case — the summarizer BLOATS (a fire
    never shrinks; the live-stall shape) while the session grows a big tool exchange every
    iteration of ONE long turn (loop.py keeps the full session list and rebuilds each iteration).
    Utilization must stay <= 100% at EVERY iteration: the fire cap bounds summarizer runs and the
    emergency floor hard-truncates the oldest non-system messages at safe-cut boundaries."""
    from localharness.agent.context import CompactionPipeline, ContextManager, TokenCounter

    async def bloating(middle):
        return "X " * 3000  # a 'summary' larger than the whole budget — worst-case fire

    tc = TokenCounter()
    pipeline = CompactionPipeline(tc, preserve_first_n=1, preserve_last_n=1, llm_summarize_fn=bloating)
    cm = ContextManager(max_context_tokens=2000, pipeline=pipeline, token_counter=tc)
    session = [{"role": "system", "content": "sys"}, {"role": "user", "content": "the task"}]
    for i in range(12):  # one long turn: the session list only ever grows
        session.append({"role": "assistant", "content": None,
                        "tool_calls": [{"id": f"tc{i}", "function": {"name": "bash", "arguments": "{}"}}]})
        session.append({"role": "tool", "tool_call_id": f"tc{i}", "content": "result words " * 120})
        _req, budget = await cm.build_messages(list(session))
        assert budget.usage_fraction <= 1.0, f"context overflow at iteration {i}: {budget.usage_fraction:.2%}"


@pytest.mark.asyncio
async def test_compaction_reduces_to_target_when_safe_cuts_exist():
    """Ruling test (b): a fire must land utilization AT TARGET (default 0.60 — well below the
    0.80 trigger), not merely 'shrink a little' (the sawtooth). Head/tail sized so ONE summarize
    pass at the configured preserve width CANNOT reach the target: the stage must WIDEN the
    summarized span (>= 2 summarize calls) and end <= target."""
    from localharness.agent.context import (
        COMPACTION_TARGET_USAGE_FRACTION, CompactionPipeline, ContextManager, TokenCounter,
    )

    calls = {"n": 0}

    async def tiny(middle):
        calls["n"] += 1
        return "a compact summary"

    big = "word " * 150
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(16):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": big})
    msgs.append({"role": "user", "content": "recent tail"})
    tc = TokenCounter()
    n = tc.count_messages(msgs)
    # preserve 7/7 over 18 messages: the first summarize removes only 4 of 16 big messages ->
    # ~0.75 of the input retained, ABOVE the target (0.6 of limit = ~0.706 of the 0.85-start
    # input) -> the stage MUST widen (a second summarize) to land under target.
    pipeline = CompactionPipeline(tc, preserve_first_n=7, preserve_last_n=7, llm_summarize_fn=tiny)
    cm = ContextManager(max_context_tokens=int(n / 0.85), pipeline=pipeline, token_counter=tc)

    _req, budget = await cm.build_messages(list(msgs))
    assert budget.usage_fraction <= COMPACTION_TARGET_USAGE_FRACTION, (
        f"compaction landed at {budget.usage_fraction:.2%}, above the "
        f"{COMPACTION_TARGET_USAGE_FRACTION:.0%} target"
    )
    assert calls["n"] >= 2, "one narrow pass met the target — widen-to-target was never exercised"


@pytest.mark.asyncio
async def test_compaction_guard_caps_fires_per_turn():
    """Ruling test (c): the per-turn fire cap is the BACKSTOP — even when each fire genuinely
    shrinks (context regrows next iteration), the expensive summarizer runs at most
    MAX_COMPACTION_FIRES_PER_TURN times per turn; reset_compaction_guard re-arms a new turn."""
    calls = {"n": 0}

    async def shrinking(middle):
        calls["n"] += 1
        return "s"  # a tiny summary -> each fire genuinely shrinks and meets target in one call

    cm, msgs = _guard_cm(shrinking)
    for _ in range(6):  # same oversized input each iteration -> would fire every time, uncapped
        await cm.build_messages(list(msgs))
    assert calls["n"] == 3, f"per-turn fire cap not enforced: summarizer ran {calls['n']}x (cap 3)"

    cm.reset_compaction_guard()  # a new turn re-arms the cap
    await cm.build_messages(list(msgs))
    assert calls["n"] == 4, "reset_compaction_guard must re-arm compaction for the next turn"


# --- Phase 4: compact.md load path ---

def test_load_compact_md_returns_message_when_file_exists(tmp_path):
    """load_compact_md returns a system message when compact.md exists with content."""
    from localharness.agent.context import load_compact_md
    compact_file = tmp_path / "compact.md"
    compact_file.write_text("Previous session summary: user was building a research agent.")
    msg = load_compact_md(compact_file)
    assert msg is not None
    assert msg["role"] == "system"
    assert "[Prior Session Context]" in msg["content"]
    assert "research agent" in msg["content"]


def test_load_compact_md_returns_none_when_missing(tmp_path):
    """load_compact_md returns None when compact.md does not exist."""
    from localharness.agent.context import load_compact_md
    compact_file = tmp_path / "compact.md"
    msg = load_compact_md(compact_file)
    assert msg is None


def test_load_compact_md_returns_none_when_empty(tmp_path):
    """load_compact_md returns None when compact.md is empty."""
    from localharness.agent.context import load_compact_md
    compact_file = tmp_path / "compact.md"
    compact_file.write_text("")
    msg = load_compact_md(compact_file)
    assert msg is None


# ---------------------------------------------------------------------------
# SCEN-04 plumbing: CompactionTriggered publication (Plan 12-01 Task 2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compaction_publishes_event():
    """When pipeline modifies messages, CompactionTriggered is published once.

    The stub shrinks to a tiny message (not just messages[:1], which at "x"*5000 is still
    hugely over the 100-token budget) so the emergency floor does NOT also fire afterward and
    publish a second event — this test isolates the LLM-stage publish path specifically. The
    emergency-floor-on-top-of-an-unmodified-pipeline case is its own test below."""
    from localharness.agent.context import ContextManager
    from localharness.core.events import CompactionTriggered

    published: list = []

    class FakeBus:
        async def publish(self, event):
            published.append(event)

    # Stub pipeline that always reports modified=True on the LLM stage (FIX 2 split interface)
    class StubPipeline:
        async def run(self, messages, budget):
            return [{"role": "user", "content": "y"}], True
        async def run_deterministic(self, messages, budget):
            return messages, False
        async def run_llm(self, messages, budget):
            return [{"role": "user", "content": "y"}], True   # shrinks well under budget

    # Use a tiny budget so needs_summary_compact triggers
    cm = ContextManager(
        max_context_tokens=100,
        pipeline=StubPipeline(),
        bus=FakeBus(),
        agent_id="agent-1",
        session_id="session-1",
    )
    cm.set_iteration(7)

    # Build messages large enough to exceed needs_summary_compact threshold
    big = [{"role": "user", "content": "x" * 5000}] * 5
    await cm.build_messages(big, tool_schemas=None)

    assert len(published) == 1
    ev = published[0]
    assert isinstance(ev, CompactionTriggered)
    assert ev.agent_id == "agent-1"
    assert ev.session_id == "session-1"
    assert ev.iteration == 7
    assert ev.pre_usage_fraction >= 0.0
    assert ev.post_usage_fraction >= 0.0


@pytest.mark.asyncio
async def test_compaction_no_publish_when_unchanged():
    """When pipeline reports any_modified=False AND the (unmodified) content already fits under
    budget, no event is published. Sized (dynamically, via the same TokenCounter the manager
    uses) to land inside needs_summary_compact (>=80% usage) but under the emergency floor's
    hard limit — isolating the LLM-stage-only path. A genuinely oversized-and-unmodified case IS
    expected to publish now, via the emergency floor — see the next test."""
    from localharness.agent.context import ContextManager, TokenCounter

    published: list = []

    class FakeBus:
        async def publish(self, event):
            published.append(event)

    class NoOpPipeline:
        async def run(self, messages, budget):
            return messages, False
        async def run_deterministic(self, messages, budget):
            return messages, False
        async def run_llm(self, messages, budget):
            return messages, False

    tc = TokenCounter()
    messages = [{"role": "user", "content": "word " * 40}]
    usage = tc.count_messages(messages)
    max_ctx = max(10, int(usage / 0.85))  # usage_fraction ~0.85: >= needs_summary_compact(0.80),
                                           # still < the emergency floor's effective_limit (== max_ctx here)
    cm = ContextManager(
        max_context_tokens=max_ctx,
        pipeline=NoOpPipeline(),
        bus=FakeBus(),
        agent_id="a",
        session_id="s",
        token_counter=tc,
    )
    await cm.build_messages(list(messages), tool_schemas=None)
    assert published == []


@pytest.mark.asyncio
async def test_emergency_floor_publishes_when_pipeline_leaves_it_unchanged():
    """Regression: when the LLM pipeline reports any_modified=False but the content is still
    genuinely oversized (e.g. a single first-turn message gives SummaryCompactionStage no safe
    'middle' to summarize — see near_compaction), the emergency floor is the only thing that
    shrinks the request. It must publish CompactionTriggered too — a silent cut is exactly what
    the event exists to record (this is the exact fixture test_compaction_no_publish_when_unchanged
    used before it was narrowed to isolate the LLM-only path above)."""
    from localharness.agent.context import ContextManager
    from localharness.core.events import CompactionTriggered

    published: list = []

    class FakeBus:
        async def publish(self, event):
            published.append(event)

    class NoOpPipeline:
        async def run(self, messages, budget):
            return messages, False
        async def run_deterministic(self, messages, budget):
            return messages, False
        async def run_llm(self, messages, budget):
            return messages, False

    cm = ContextManager(
        max_context_tokens=100,
        pipeline=NoOpPipeline(),
        bus=FakeBus(),
        agent_id="a",
        session_id="s",
    )
    big = [{"role": "user", "content": "x" * 5000}] * 5
    await cm.build_messages(big, tool_schemas=None)
    assert len(published) == 1
    assert isinstance(published[0], CompactionTriggered)
    assert published[0].agent_id == "a"
    assert published[0].session_id == "s"
    assert published[0].stages_modified == []


@pytest.mark.asyncio
async def test_compaction_no_bus_no_publish():
    """Back-compat — ContextManager without bus must not raise."""
    from localharness.agent.context import ContextManager

    class StubPipeline:
        async def run(self, messages, budget):
            return messages[:1], True
        async def run_deterministic(self, messages, budget):
            return messages, False
        async def run_llm(self, messages, budget):
            return messages[:1], True

    cm = ContextManager(max_context_tokens=100, pipeline=StubPipeline())
    big = [{"role": "user", "content": "x" * 5000}] * 5
    out, _budget = await cm.build_messages(big, tool_schemas=None)
    # Should not raise. No bus, no publication. The stub truncates to one message; the emergency
    # floor then head+tail shrinks that oversized lone message to fit (FIX 3) — so assert structure,
    # not byte-for-byte equality (the pre-FIX pass-through of an over-budget message was the bug).
    assert len(out) == 1 and out[0]["role"] == "user"


def test_default_deny_patterns_use_bash_exec():
    """Default deny_patterns reference bash_exec (the actual tool name), not legacy bash."""
    from localharness.config.models import PermissionConfig
    patterns = PermissionConfig().deny_patterns
    # issue #15: `sudo:*` (needed a literal colon, matched no real sudo cmd) -> `*sudo *`.
    assert "bash_exec(*sudo *)" in patterns
    assert "bash_exec(rm -rf *)" in patterns
    assert "bash_exec(chmod 777 *)" in patterns
    assert "bash_exec(sudo:*)" not in patterns
    assert "bash(sudo:*)" not in patterns
    assert "bash(rm -rf *)" not in patterns


# ---------------------------------------------------------------------------
# Stale web-result eviction (_evict_stale_web_results + build_messages gate)
# ---------------------------------------------------------------------------

import json

from localharness.agent.context import (
    WEB_EVICT_KEEP_LAST,
    _evict_stale_web_results,
)


def _web_exchange(i: int, tool: str = "web_fetch", body_chars: int = 3000):
    """One assistant tool-call + tool-result pair for a web tool."""
    hint = {"url": f"https://example.test/p{i}"} if tool == "web_fetch" else {"query": f"q{i}"}
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"wc-{i}", "type": "function",
            "function": {"name": tool, "arguments": json.dumps(hint)},
        }]},
        {"role": "tool", "tool_call_id": f"wc-{i}", "content": "x" * body_chars},
    ]


def test_evict_stubs_all_but_newest_keeping_hint():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(4):
        msgs += _web_exchange(i)
    out, evicted = _evict_stale_web_results(msgs, keep_last=2)
    assert evicted == 2
    stubbed = [m for m in out if m.get("role") == "tool" and "omitted" in m["content"]]
    assert len(stubbed) == 2
    # oldest two stubbed, URL hint preserved, newest two intact
    assert "https://example.test/p0" in stubbed[0]["content"]
    assert out[-1]["content"] == "x" * 3000
    # original list untouched (no mutation)
    assert msgs[3]["content"] == "x" * 3000


def test_evict_skips_small_and_non_web_results():
    msgs = [
        *_web_exchange(0, body_chars=100),               # small web result — skip
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "rc-1", "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "rc-1", "content": "y" * 5000},  # non-web — skip
        *_web_exchange(1),
        *_web_exchange(2),
        *_web_exchange(3),
    ]
    out, evicted = _evict_stale_web_results(msgs, keep_last=2)
    assert evicted == 1  # only exchange 1 (oldest big web beyond keep-last)
    assert out[1]["content"] == "x" * 100          # small survives
    assert any(m.get("content") == "y" * 5000 for m in out)  # read result survives


def test_evict_idempotent_on_stubs():
    msgs = []
    for i in range(4):
        msgs += _web_exchange(i)
    once, n1 = _evict_stale_web_results(msgs, keep_last=1)
    twice, n2 = _evict_stale_web_results(once, keep_last=1)
    assert n1 == 3 and n2 == 0
    assert once == twice


@pytest.mark.asyncio
async def test_build_messages_evicts_only_over_threshold():
    from localharness.agent.context import ContextManager

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(4):
        msgs += _web_exchange(i, body_chars=4000)
    # tiny window -> usage fraction far over 0.50 -> eviction fires
    cm_small = ContextManager(max_context_tokens=2000)
    built, _ = await cm_small.build_messages(list(msgs), None)
    assert sum("omitted" in (m.get("content") or "") for m in built) == 4 - WEB_EVICT_KEEP_LAST
    # huge window -> under threshold -> untouched
    cm_big = ContextManager(max_context_tokens=1_000_000)
    built2, _ = await cm_big.build_messages(list(msgs), None)
    assert all("omitted" not in (m.get("content") or "") for m in built2)


# ---------------------------------------------------------------------------
# Emergency-floor compaction floors (FIX WAVE): the floor must preserve turn
# structure (FIX 1), deterministic capping must survive the LLM fire cap (FIX 2),
# and the floor must actually FIT the budget incl. the reply reserve (FIX 3).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emergency_floor_drops_whole_exchange_not_lone_user():
    """FIX 1 (root cause B): the emergency floor's deletion unit is a FULL user-turn exchange
    (a user message through everything before the next user message), never a lone user message.
    Dropping only the older user would leave [system, assistant, tool, ...] — no leading user turn —
    which the OpenAI-compatible server 400s. Sized so dropping ONLY the old user satisfies budget."""
    from localharness.agent.context import _hard_truncate_to_budget, TokenCounter
    tc = TokenCounter()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "OLD task " + "word " * 2000},   # big: dropping THIS alone fits
        _make_assistant_with_tool_call("tc-1"),
        _make_tool_result("tc-1", "ok-old"),
        {"role": "user", "content": "NEW task"},
        _make_assistant_with_tool_call("tc-2"),
        _make_tool_result("tc-2", "ok-new"),
    ]
    # Budget fits everything EXCEPT the big old user message -> old code drops only that user and
    # stops at [system, assistant, tool, ...]; the fix must drop the WHOLE old exchange instead.
    budget = tc.count_messages([messages[0]] + messages[2:]) + 20
    out, _dropped = _hard_truncate_to_budget(messages, budget, tc)
    non_system = [m for m in out if m.get("role") != "system"]
    assert non_system and non_system[0]["role"] == "user"          # FIX 1: leading user preserved
    assert non_system[0]["content"] == "NEW task"                  # the OLD exchange dropped WHOLE
    # invariant: truncation never leaves an assistant directly after the system message
    assert not (len(out) >= 2 and out[0]["role"] == "system" and out[1]["role"] == "assistant")


@pytest.mark.asyncio
async def test_tool_result_cap_survives_exhausted_fire_budget():
    """FIX 2 (root cause A): the per-turn LLM fire cap must gate ONLY the LLM stages. The cheap
    deterministic ToolResultCapStage must still run when the cap is exhausted, so an oversized tool
    result is capped instead of riding raw (52-72K chars) into the request."""
    from localharness.agent.context import (
        CompactionPipeline, ContextManager, TokenCounter, MAX_COMPACTION_FIRES_PER_TURN,
    )

    async def never_called(middle):
        raise AssertionError("LLM summarizer must NOT run once the per-turn fire cap is exhausted")

    tc = TokenCounter()
    pipeline = CompactionPipeline(
        token_counter=tc, tool_result_cap=1_000, llm_summarize_fn=never_called,
        preserve_first_n=1, preserve_last_n=1,
    )
    cm = ContextManager(max_context_tokens=4_000, pipeline=pipeline, token_counter=tc)
    cm._compaction_fires = MAX_COMPACTION_FIRES_PER_TURN  # fire budget already spent this turn
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _make_assistant_with_tool_call("tc-1"),
        _make_tool_result("tc-1", "R" * 50_000),  # oversized: over the 0.80 window AND over the cap
    ]
    out, _budget = await cm.build_messages(messages)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 1, "the tool result must survive (capped), not be dropped by the floor"
    assert len(tool_msgs[0]["content"]) <= 1_100, "oversized tool result must be capped to tool_result_cap"
    assert "elided" in tool_msgs[0]["content"], "cap uses the head+tail keep marker"


@pytest.mark.asyncio
async def test_floor_shrinks_undroppable_final_message_with_marker():
    """FIX 3 (root cause C): when the un-droppable remnant (system + the final message) itself
    exceeds budget, dropping whole exchanges cannot help. The floor must head+tail SHRINK the
    content as a true last resort so the request actually fits UNDER (budget - reply reserve)."""
    from localharness.agent.context import ContextManager, TokenCounter, RESPONSE_RESERVE_TOKENS
    tc = TokenCounter()
    cm = ContextManager(max_context_tokens=8_000, token_counter=tc)  # no pipeline: floor is the only lever
    huge = " ".join(f"tok{i}" for i in range(40_000))  # ~250K chars / tens of K tokens >> budget
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _make_assistant_with_tool_call("tc-1"),
        _make_tool_result("tc-1", huge),  # final tool message larger than the whole budget
    ]
    out, budget = await cm.build_messages(messages)
    # (1) the built request fits UNDER (budget - reserve) — room for the model's reply
    assert budget.current_usage + budget.tool_schema_tokens <= cm.max_context_tokens - RESPONSE_RESERVE_TOKENS
    # (2) content was visibly truncated with the head+tail marker (not silently dropped whole)
    assert any("elided" in (m.get("content") or "") for m in out), "content must be shrunk with the marker"
    assert out[-1].get("role") == "tool", "the final tool message must survive (shrunk), not be deleted"


@pytest.mark.asyncio
async def test_floor_reserves_response_headroom():
    """FIX 3 (reserve): the floor must compare against (max_context_tokens - RESPONSE_RESERVE_TOKENS),
    not the full budget — so a request that 'fits' still leaves room for the reply. A conversation
    sized just above (max - reserve) but under max must trigger the floor and land headroom >= 0."""
    from localharness.agent.context import ContextManager, TokenCounter, RESPONSE_RESERVE_TOKENS
    tc = TokenCounter()
    filler = " ".join(f"f{i}" for i in range(5_000))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "OLD " + filler},   # big older exchange (droppable whole)
        _make_assistant_with_tool_call("tc-1"),
        _make_tool_result("tc-1", "ok-old"),
        {"role": "user", "content": "NEW task"},
        _make_assistant_with_tool_call("tc-2"),
        _make_tool_result("tc-2", "ok-new"),
    ]
    total = tc.count_messages(messages)
    # Place `total` inside (max - reserve, max]: old floor (keyed to max) would NOT fire, shipping a
    # request with < reserve free; the fix (keyed to max - reserve) must fire and leave headroom >= 0.
    max_ctx = total + RESPONSE_RESERVE_TOKENS // 2
    assert max_ctx > RESPONSE_RESERVE_TOKENS, "budget must exceed the reserve so it actually applies"
    cm = ContextManager(max_context_tokens=max_ctx, token_counter=tc)
    out, budget = await cm.build_messages(messages)
    assert budget.headroom >= 0, "floor must leave >= reserve tokens free for the model's reply"
    non_system = [m for m in out if m.get("role") != "system"]
    assert non_system and non_system[0]["role"] == "user"  # FIX 1 structure preserved by the floor


# --- #30: rebind must be exception-safe (a failed re-probe must not brick the session) --- #


def test_rebind_failure_restores_prior_binding_and_count_survives(monkeypatch):
    """#30: a failed re-probe must leave the counter in the PRIOR, consistent binding — never
    _mode=live + _tokenize_url=None, which makes count() raise on EVERY later call (the brick
    reported to the user as a successful swap). rebind still RAISES (the known-runtime fail-loud
    contract) but the state is restored BEFORE it raises, so the session stays usable."""
    from localharness.agent.context import TokenCounter

    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 5)
    monkeypatch.setattr(TokenCounter, "_remote_count_messages", lambda self, m, t: 7)
    tc = TokenCounter(base_url="http://localhost:8000/v1", model="model-a", provider_type="vllm")
    assert tc._mode == "vllm" and tc._model == "model-a"
    assert tc._messages_exact
    prev_url = tc._tokenize_url
    tc.count("prime the cache")  # counts cached under model-a's tokenizer

    # Re-probe now fails for the new model on a KNOWN runtime → rebind must raise…
    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: None)
    with pytest.raises(RuntimeError):
        tc.rebind(base_url="http://localhost:8000/v1", model="model-b", provider_type="vllm")

    # …but leave the PRIOR binding intact — consistent, not a half-set brick.
    assert tc._model == "model-a"
    assert tc._tokenize_url == prev_url
    assert tc._mode == "vllm"
    assert tc._messages_exact  # message-level capability restored with the rest

    # The #30 symptom was count() raising on EVERY later turn. The restored exact binding answers.
    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 5)
    assert tc.count("a later turn") == 5


# --- #31: served-window probe (id-matched, never raises) feeds the /model budget refit --- #


def test_probe_served_window_vllm_id_matched(monkeypatch):
    from localharness.agent import context as ctxmod

    class _Resp:
        def json(self):
            return {"data": [
                {"id": "other", "max_model_len": 8192},
                {"id": "model-b", "max_model_len": 32768},
            ]}

    import httpx
    monkeypatch.setattr(httpx, "get", lambda url, timeout=0: _Resp())
    assert ctxmod.probe_served_window("http://localhost:8000/v1", "model-b", "vllm") == 32768
    # An id the endpoint doesn't serve → None (caller discloses instead of refitting a guess).
    assert ctxmod.probe_served_window("http://localhost:8000/v1", "absent", "vllm") is None


def test_probe_served_window_llamacpp_props(monkeypatch):
    from localharness.agent import context as ctxmod
    import httpx

    class _Resp:
        def json(self):
            return {"default_generation_settings": {"n_ctx": 16384}}

    monkeypatch.setattr(httpx, "get", lambda url, timeout=0: _Resp())
    assert ctxmod.probe_served_window("http://localhost:8080/v1", "m", "llamacpp") == 16384


def test_probe_served_window_never_raises(monkeypatch):
    """A probe error must degrade to None (→ the caller discloses), never crash the swap."""
    from localharness.agent import context as ctxmod
    import httpx

    def _boom(url, timeout=0):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert ctxmod.probe_served_window("http://localhost:8000/v1", "model-b", "vllm") is None


def test_probe_served_window_ollama_api_ps(monkeypatch):
    """Ollama's SERVED window is the LOADED model's num_ctx from GET /api/ps — NOT the model
    ceiling from /api/show (which over-reports and would cause silent prompt truncation). Loaded
    → that served window; not-loaded (lazy-load) / older builds → None (disclose, never the max)."""
    from localharness.agent import context as ctxmod
    import httpx

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    got: dict = {}

    def _get(url, timeout=0):
        got["url"] = url
        return _Resp({"models": [{"model": "qwen2.5:0.5b", "name": "qwen2.5:0.5b",
                                  "context_length": 4096}]})

    monkeypatch.setattr(httpx, "get", _get)
    assert ctxmod.probe_served_window("http://localhost:11434/v1", "qwen2.5:0.5b", "ollama") == 4096
    assert got["url"] == "http://localhost:11434/api/ps"   # native API, /v1 stripped

    # model not loaded (empty /api/ps) → None — disclose, never fall back to the model ceiling.
    monkeypatch.setattr(httpx, "get", lambda url, timeout=0: _Resp({"models": []}))
    assert ctxmod.probe_served_window("http://localhost:11434/v1", "qwen2.5:0.5b", "ollama") is None


def test_probe_served_window_lmstudio_loaded_context(monkeypatch):
    """LM Studio's /api/v0/models reports the LOADED window (loaded_context_length) id-matched.
    max_context_length is IGNORED — over-reporting would let the start guard pass a too-big cfg."""
    from localharness.agent import context as ctxmod
    import httpx

    class _Resp:
        def json(self):
            return {"object": "list", "data": [
                {"id": "other", "state": "loaded", "loaded_context_length": 4096},
                {"id": "target", "state": "loaded",
                 "loaded_context_length": 8192, "max_context_length": 32768},
            ]}

    got: dict = {}

    def _get(url, timeout=0):
        got["url"] = url
        return _Resp()

    monkeypatch.setattr(httpx, "get", _get)
    assert ctxmod.probe_served_window("http://localhost:1234/v1", "target", "lmstudio") == 8192
    assert got["url"] == "http://localhost:1234/api/v0/models"  # native REST, /v1 stripped

    # A model present but NOT loaded (no loaded_context_length) → None, even if max is known.
    class _RespNotLoaded:
        def json(self):
            return {"data": [{"id": "target", "state": "not-loaded", "max_context_length": 32768}]}

    monkeypatch.setattr(httpx, "get", lambda url, timeout=0: _RespNotLoaded())
    assert ctxmod.probe_served_window("http://localhost:1234/v1", "target", "lmstudio") is None
# --- provider_type is a PRIORITY HINT, not an exclusive lock ---------------------------
# A stored provider_type that no longer matches the running server (user swapped vLLM ->
# llama.cpp behind the same port) used to HARD-FAIL start: the hint selected exactly one
# /tokenize shape and, when it didn't answer, raised. An ABSENT hint self-healed by trying
# both. Explicit config was therefore strictly worse than no config. The hint now orders
# the probe instead of narrowing it.


def _patch_urlopen_by_shape(monkeypatch, *, answers: set[str]):
    """Serve /tokenize ONLY for the shapes in `answers`, mirroring the real contracts:
    vLLM {model,prompt}->{count}; llama.cpp {content}->{tokens:[...]}. A foreign shape
    400s, which is what a real server does when handed the other runtime's payload."""
    import json as _json
    import urllib.error
    import urllib.request

    class _Resp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._payload

    def fake_urlopen(req, timeout=0):
        body = _json.loads(req.data.decode("utf-8"))
        if "content" in body and "llamacpp" in answers:
            return _Resp(b'{"tokens": [1, 2, 3]}')
        if "prompt" in body and "vllm" in answers:
            return _Resp(b'{"count": 3}')
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_stale_vllm_hint_falls_through_to_llamacpp(monkeypatch):
    """Config says vllm, server is actually llama.cpp -> lock onto llamacpp, do NOT raise."""
    from localharness.agent.context import TokenCounter
    _patch_urlopen_by_shape(monkeypatch, answers={"llamacpp"})
    tc = TokenCounter(base_url="http://localhost:8000/v1", model="m", provider_type="vllm")
    assert tc._mode == "llamacpp"
    assert tc.count("abc") == 3
    assert not tc.approximate


def test_stale_llamacpp_hint_falls_through_to_vllm(monkeypatch):
    """Symmetric: config says llamacpp, server is actually vLLM -> lock onto vllm."""
    from localharness.agent.context import TokenCounter
    _patch_urlopen_by_shape(monkeypatch, answers={"vllm"})
    tc = TokenCounter(base_url="http://localhost:8000/v1", model="m", provider_type="llamacpp")
    assert tc._mode == "vllm"
    assert tc.count("abc") == 3


def test_hint_is_tried_first_when_both_shapes_answer(monkeypatch):
    """The hint must ORDER the probe: with a server that would answer either shape, the
    configured one wins. This is what keeps the common (correct-config) path at zero
    extra round-trips."""
    from localharness.agent.context import TokenCounter
    _patch_urlopen_by_shape(monkeypatch, answers={"vllm", "llamacpp"})
    assert TokenCounter(
        base_url="http://localhost:8000/v1", model="m", provider_type="llamacpp"
    )._mode == "llamacpp"
    assert TokenCounter(
        base_url="http://localhost:8000/v1", model="m", provider_type="vllm"
    )._mode == "vllm"


def test_no_shape_answers_still_hard_fails(monkeypatch):
    """The safety net stays: a runtime that SHOULD serve /tokenize but answers on neither
    shape is still a hard error — silently degrading to approximate counting is what let
    context overflows through (#8)."""
    import pytest as _pytest
    from localharness.agent.context import TokenCounter
    _patch_urlopen_by_shape(monkeypatch, answers=set())
    with _pytest.raises(RuntimeError, match="exact token counting unavailable"):
        TokenCounter(base_url="http://localhost:8000/v1", model="m", provider_type="vllm")
