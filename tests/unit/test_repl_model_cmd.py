"""REPL /model — list, hot-swap, managed restart, persistence."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import (
    HarnessConfig,
    ManagedServerConfig,
    OrgConfig,
    ProviderConfig,
)
from localharness.config.overlay import (
    atomic_write_overlay,
    load_overlay,
)


class FakeChannel:
    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, text, metadata=None):
        self.messages.append(text)


class FakeLLM:
    def __init__(self, model="model-a"):
        self.config = SimpleNamespace(base_url="http://localhost:8081/v1", model=model)

    async def detect_capabilities(self):
        return SimpleNamespace(tool_call_mode="native")


def _repl(tmp_path, harness, live):
    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM())
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(),
        agent_loop=agent,
        channel=channel,
        bus=SimpleNamespace(),
        config_dir=tmp_path,
        harness_config=harness,
    )

    async def _live_models(base_url):
        return list(live), True  # (#38) shared (ids, reachable) contract

    repl._live_models = _live_models
    return repl, channel, agent


def _harness(server=None, audit_log_path=None):
    return HarnessConfig(
        provider=ProviderConfig(
            provider_type="vllm",
            base_url="http://localhost:8081/v1",
            default_model="model-a",
            available_models=["model-a"],
        ),
        org=OrgConfig(default_model="model-a", audit_log_path=audit_log_path),
        server=server,
    )


@pytest.mark.asyncio
async def test_model_list_shows_serving_and_downloaded(tmp_path, monkeypatch):
    from localharness.provider import server as managed_server

    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["model-a", "cached-b"])
    srv = ManagedServerConfig(binary="/x/vllm", model="model-a")
    repl, channel, _ = _repl(tmp_path, _harness(srv), live=["model-a"])

    handled = await repl._handle_slash("/model")
    assert handled is True
    out = channel.messages[-1]
    assert "model-a" in out and "(serving)" in out and "[active]" in out
    assert "cached-b" in out and "downloaded" in out


@pytest.mark.asyncio
async def test_model_hotswap_updates_client_and_persists(tmp_path):
    repl, channel, agent = _repl(tmp_path, _harness(), live=["model-a", "model-b"])

    await repl._handle_slash("/model model-b")
    assert agent._llm.config.model == "model-b"
    # Persistence now goes to the atomic USER OVERLAY, not a config.yaml rewrite (issue #22).
    assert not (tmp_path / "config.yaml").exists()
    # #35: the overlay lands under the REPL's config_dir (tmp_path), not LOCALHARNESS_HOME.
    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay["provider"]["default_model"] == "model-b"
    assert overlay["org"]["default_model"] == "model-b"
    assert "Switched to model-b" in channel.messages[-1]


@pytest.mark.asyncio
async def test_model_swap_by_number(tmp_path):
    repl, _, agent = _repl(tmp_path, _harness(), live=["model-a", "model-b"])
    await repl._handle_slash("/model 2")
    assert agent._llm.config.model == "model-b"


@pytest.mark.asyncio
async def test_model_unknown_rejected(tmp_path):
    repl, channel, agent = _repl(tmp_path, _harness(), live=["model-a"])
    await repl._handle_slash("/model nope")
    assert agent._llm.config.model == "model-a"
    assert "Unknown model" in channel.messages[-1]


@pytest.mark.asyncio
async def test_model_managed_restart_path(tmp_path, monkeypatch):
    from localharness.provider import server as managed_server

    calls: list[str] = []
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    monkeypatch.setattr(managed_server, "stop_server", lambda cfg, launch="binary": calls.append("stop") or True)
    monkeypatch.setattr(managed_server, "start_server", lambda cfg, cmd: calls.append("start") or 1234)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        calls.append("wait")
        return ["cached-b"]

    monkeypatch.setattr(managed_server, "wait_ready", fake_wait_ready)

    srv = ManagedServerConfig(binary="/x/vllm", model="model-a")
    harness = _harness(srv)
    repl, channel, agent = _repl(tmp_path, harness, live=["model-a"])

    await repl._handle_slash("/model cached-b")
    assert calls == ["stop", "start", "wait"]
    assert agent._llm.config.model == "cached-b"
    assert harness.server.model == "cached-b"
    assert not (tmp_path / "config.yaml").exists()
    overlay = load_overlay(tmp_path / "overrides.yaml")  # #35: under config_dir, not LOCALHARNESS_HOME
    assert overlay["provider"]["default_model"] == "cached-b"


@pytest.mark.asyncio
async def test_model_unavailable_without_harness_config(tmp_path):
    repl, channel, _ = _repl(tmp_path, None, live=[])
    await repl._handle_slash("/model")
    assert "unavailable" in channel.messages[-1]


# --- Gap #25: TokenCounter must rebind to the new model on a mid-session swap --- #


@pytest.mark.asyncio
async def test_model_hotswap_rebinds_token_counter(tmp_path, monkeypatch):
    """A hot-swap must rebind the shared TokenCounter so mid-session counting uses
    the new served model's tokenizer, not the construction-time one (issue #25)."""
    from localharness.agent.context import TokenCounter

    # Offline: the /tokenize probe always answers, so exact mode locks with no live call.
    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    tc = TokenCounter(base_url="http://localhost:8081/v1", model="model-a", provider_type="vllm")
    assert tc._model == "model-a"
    tc.count("prime the cache")  # populate the content-hash cache under model-a
    assert tc._cache  # non-empty

    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM(), _ctx=SimpleNamespace(_token_counter=tc))
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(),
        agent_loop=agent,
        channel=channel,
        bus=SimpleNamespace(),
        config_dir=tmp_path,
        harness_config=_harness(),
    )

    async def _live_models(base_url):
        return ["model-a", "model-b"], True

    repl._live_models = _live_models

    await repl._handle_slash("/model model-b")
    assert agent._llm.config.model == "model-b"
    assert tc._model == "model-b"  # rebound to the new model
    assert tc._cache == {}  # stale per-tokenizer counts cleared


@pytest.mark.asyncio
async def test_model_managed_restart_rebinds_token_counter(tmp_path, monkeypatch):
    """The managed-restart path must also rebind the counter to the served model (#25)."""
    from localharness.agent.context import TokenCounter
    from localharness.provider import server as managed_server

    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 3)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    monkeypatch.setattr(managed_server, "stop_server", lambda cfg, launch="binary": True)
    monkeypatch.setattr(managed_server, "start_server", lambda cfg, cmd: 1234)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        return ["cached-b"]

    monkeypatch.setattr(managed_server, "wait_ready", fake_wait_ready)

    tc = TokenCounter(base_url="http://localhost:8081/v1", model="model-a", provider_type="vllm")
    srv = ManagedServerConfig(binary="/x/vllm", model="model-a")
    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM(), _ctx=SimpleNamespace(_token_counter=tc))
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(),
        agent_loop=agent,
        channel=channel,
        bus=SimpleNamespace(),
        config_dir=tmp_path,
        harness_config=_harness(srv),
    )

    async def _live_models(base_url):
        return ["model-a"], True

    repl._live_models = _live_models

    await repl._handle_slash("/model cached-b")
    assert agent._llm.config.model == "cached-b"
    assert tc._model == "cached-b"


# --- Gap #22: /model persistence migrates to the atomic, audited user-overlay path --- #


@pytest.mark.asyncio
async def test_model_swap_persists_via_atomic_overlay_with_audit(tmp_path, components_home):
    audit = components_home / "audit.jsonl"
    harness = _harness(audit_log_path=str(audit))
    repl, _, agent = _repl(tmp_path, harness, live=["model-a", "model-b"])

    await repl._handle_slash("/model model-b")

    # Overlay is the write target (atomic); config.yaml is never rewritten. #35: it lands under
    # the REPL's config_dir (tmp_path), not LOCALHARNESS_HOME (components_home).
    assert not (tmp_path / "config.yaml").exists()
    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay["provider"]["default_model"] == "model-b"
    assert overlay["org"]["default_model"] == "model-b"
    # available_models UNION-merges rather than clobbering (overlay deep_merge replaces lists).
    assert set(overlay["provider"]["available_models"]) == {"model-a", "model-b"}

    # ComponentMutated audit event emitted for the provider default.
    events = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()]
    muts = [e for e in events if e.get("event_type") == "ComponentMutated"]
    assert any(
        e["path"] == "provider.default_model" and e["after_value"] == "model-b"
        for e in muts
    )


@pytest.mark.asyncio
async def test_model_swap_preserves_unrelated_overlay_keys(tmp_path, components_home):
    """A model switch must not clobber pre-existing overlay keys — notably the agent-scope
    slice (where the tag_grouping_enabled kill lever lives) and unrelated harness keys."""
    # #35: the REPL persists under its config_dir (tmp_path); pre-seed the overlay THERE.
    overlay_path = tmp_path / "overrides.yaml"
    atomic_write_overlay(
        overlay_path,
        {
            "agent": {"stuck_detector": {"window_size": 9}},
            "org": {"log_level": "debug"},
        },
    )
    repl, _, _ = _repl(tmp_path, _harness(), live=["model-a", "model-b"])

    await repl._handle_slash("/model model-b")

    overlay = load_overlay(overlay_path)
    assert overlay["agent"]["stuck_detector"]["window_size"] == 9  # untouched
    assert overlay["org"]["log_level"] == "debug"  # unrelated harness key survives
    assert overlay["provider"]["default_model"] == "model-b"
    assert overlay["org"]["default_model"] == "model-b"


# --- #38: the REPL /model list distinguishes malformed from unreachable (shared probe) --- #


@pytest.mark.asyncio
async def test_model_list_malformed_response_repl_path(tmp_path, monkeypatch):
    """#38: the REPL /model list must render a reached-but-malformed body as its own message,
    NOT a bare 'no models' — it now delegates to model_ops.list_live_models (the diverged
    duplicate is gone), so both callers share ONE failure taxonomy."""
    import json as _json

    import httpx

    class _HtmlResp:
        def json(self):
            raise _json.JSONDecodeError("Expecting value", "<html></html>", 0)

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _HtmlResp())
    # No fake _live_models — exercise the REAL delegation to model_ops.list_live_models.
    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM())
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(),
        agent_loop=agent,
        channel=channel,
        bus=SimpleNamespace(),
        config_dir=tmp_path,
        harness_config=_harness(),
    )

    await repl._handle_slash("/model")

    joined = "\n".join(channel.messages).lower()
    assert "wasn't understood" in joined or "openai-compatible" in joined


# --- #37: an audit-emit failure is not a persist failure --- #


@pytest.mark.asyncio
async def test_model_swap_audit_failure_not_reported_as_persist_failure(tmp_path, monkeypatch):
    """#37: when the audit emit raises AFTER the durable overlay write, the REPL must still
    report the swap as succeeded (with a secondary audit warning), NOT 'persisting failed'."""
    from localharness.cli import model_ops

    harness = _harness(audit_log_path=str(tmp_path / "audit.jsonl"))
    repl, channel, agent = _repl(tmp_path, harness, live=["model-a", "model-b"])

    class _BoomBus:
        def __init__(self, *a, **k):
            pass

        async def publish(self, *a, **k):
            raise RuntimeError("audit disk full")

    monkeypatch.setattr(model_ops, "EventBus", _BoomBus)

    await repl._handle_slash("/model model-b")

    assert agent._llm.config.model == "model-b"
    joined = "\n".join(channel.messages)
    assert "Switched to model-b" in joined
    assert "persisting the new default failed" not in joined
    # The overlay was still written durably.
    assert load_overlay(tmp_path / "overrides.yaml")["provider"]["default_model"] == "model-b"
    # A secondary, honestly-labeled audit warning is surfaced.
    assert "audit" in joined.lower()


# --- Per-agent pin trap: a persisted switch never reaches a model-pinned agent --- #


@pytest.mark.asyncio
async def test_model_swap_warns_on_pinned_agent(tmp_path):
    """A persisted switch silently never reaches an agent whose yaml pins a concrete model
    (start_cmd resolves the per-agent pin first). Warn, naming the agent + its pin."""
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "pinned.yaml").write_text(
        "name: pinned-agent\nrole: x\nmodel: some-pinned-model\n", encoding="utf-8"
    )
    (agents / "inheritor.yaml").write_text(
        "name: inheritor\nrole: x\nmodel: inherit\n", encoding="utf-8"
    )
    repl, channel, _ = _repl(tmp_path, _harness(), live=["model-a", "model-b"])

    await repl._handle_slash("/model model-b")

    joined = "\n".join(channel.messages)
    assert "won't reach these agents" in joined
    assert "pinned-agent" in joined and "some-pinned-model" in joined
    assert "inheritor" not in joined  # a plain inheriting agent is NOT named


@pytest.mark.asyncio
async def test_model_swap_no_pin_warning_when_none_pinned(tmp_path):
    """No pin → no warning (only agents with a concrete non-inherit model trip it)."""
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "inheritor.yaml").write_text(
        "name: inheritor\nrole: x\nmodel: inherit\n", encoding="utf-8"
    )
    repl, channel, agent = _repl(tmp_path, _harness(), live=["model-a", "model-b"])

    await repl._handle_slash("/model model-b")

    joined = "\n".join(channel.messages)
    assert "won't reach these agents" not in joined
    assert "Switched to model-b" in channel.messages[-1]


# --- #30/#31/#32: swap must refit the window + disclose a failed counter rebind, off-loop --- #


def _repl_with_ctx(tmp_path, tc, max_ctx=131_072, live=("model-a", "model-b")):
    channel = FakeChannel()
    ctx = SimpleNamespace(_token_counter=tc, max_context_tokens=max_ctx)
    agent = SimpleNamespace(_llm=FakeLLM(), _ctx=ctx)
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(), agent_loop=agent, channel=channel,
        bus=SimpleNamespace(), config_dir=tmp_path, harness_config=_harness(),
    )

    async def _live_models(base_url):
        return list(live), True  # (#38) shared (ids, reachable) contract

    repl._live_models = _live_models
    return repl, channel, agent, ctx


@pytest.mark.asyncio
async def test_model_hotswap_rebind_failure_discloses_and_stays_usable(tmp_path, monkeypatch):
    """#30: when the counter rebind FAILS on a swap, the user MUST be told via a CHANNEL
    message (not just a log line), the counter is left in a consistent PRIOR binding, and the
    next count() does not raise-every-turn (the shipped brick reported as a successful swap)."""
    from localharness.agent.context import TokenCounter

    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    tc = TokenCounter(base_url="http://localhost:8081/v1", model="model-a", provider_type="vllm")
    tc.count("prime")
    repl, channel, agent, ctx = _repl_with_ctx(tmp_path, tc)

    # Isolate the rebind failure: no window info in this test.
    monkeypatch.setattr("localharness.agent.context.probe_served_window", lambda *a, **k: None)
    # The re-probe fails for the new model on a KNOWN runtime → rebind raises internally.
    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: None)

    await repl._handle_slash("/model model-b")

    assert agent._llm.config.model == "model-b"  # the generation swap still completed
    joined = "\n".join(channel.messages)
    assert "model-b" in joined
    # Disclosure landed in the CHANNEL (assert on messages, not logs) and is actionable.
    low = joined.lower()
    assert "count" in low and "/model" in joined
    # Counter restored to the prior, consistent binding.
    assert tc._model == "model-a"
    # NEXT count() does not raise-every-turn — the prior exact binding answers again.
    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    assert tc.count("a later turn") == 7


@pytest.mark.asyncio
async def test_model_hotswap_refits_context_window_budget(tmp_path, monkeypatch):
    """#31: a hot-swap must refit ctx.max_context_tokens to the new served window and disclose
    the refit, so a 128K->32K swap can't leave a stale budget that 400s mid-session."""
    from localharness.agent.context import TokenCounter
    from localharness.cli.init_cmd import _fit_context_tokens

    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    tc = TokenCounter(base_url="http://localhost:8081/v1", model="model-a", provider_type="vllm")
    repl, channel, agent, ctx = _repl_with_ctx(tmp_path, tc, max_ctx=131_072)

    monkeypatch.setattr("localharness.agent.context.probe_served_window", lambda *a, **k: 32_768)
    await repl._handle_slash("/model model-b")

    assert ctx.max_context_tokens == _fit_context_tokens(32_768)  # refit to fit the 32K window
    assert "budget" in "\n".join(channel.messages).lower()


@pytest.mark.asyncio
async def test_model_hotswap_window_probe_failure_discloses(tmp_path, monkeypatch):
    """#31: when the served window can't be read, the budget stays put AND the user is told
    (at least parity with the managed 're-run init' hint) — never a silent stale budget."""
    from localharness.agent.context import TokenCounter

    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    tc = TokenCounter(base_url="http://localhost:8081/v1", model="model-a", provider_type="vllm")
    repl, channel, agent, ctx = _repl_with_ctx(tmp_path, tc, max_ctx=131_072)

    monkeypatch.setattr("localharness.agent.context.probe_served_window", lambda *a, **k: None)
    await repl._handle_slash("/model model-b")

    assert ctx.max_context_tokens == 131_072  # unchanged — not silently trusting a stale budget
    low = "\n".join(channel.messages).lower()
    assert "budget" in low or "window" in low or "init" in low


@pytest.mark.asyncio
async def test_refresh_token_counter_runs_off_loop(tmp_path, monkeypatch):
    """#32: the /model-triggered rebind + window probe are blocking (urllib/httpx, up to ~20s
    for two probe shapes). They must run OFF the event loop via asyncio.to_thread, or a slow
    /tokenize freezes the Discord adapter + idle consolidation that share the loop."""
    import asyncio
    from localharness.agent.context import TokenCounter

    assert asyncio.iscoroutinefunction(OrchestratorREPL._refresh_token_counter)

    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    tc = TokenCounter(base_url="http://localhost:8081/v1", model="model-a", provider_type="vllm")
    repl, channel, agent, ctx = _repl_with_ctx(tmp_path, tc)
    monkeypatch.setattr("localharness.agent.context.probe_served_window", lambda *a, **k: 32_768)

    dispatched = []
    real_to_thread = asyncio.to_thread

    async def spy(fn, *a, **k):
        dispatched.append(fn)
        return await real_to_thread(fn, *a, **k)

    monkeypatch.setattr(asyncio, "to_thread", spy)
    await repl._refresh_token_counter("model-b")
    # Both blocking calls (window probe + counter rebind) went through a worker thread.
    assert len(dispatched) >= 2
