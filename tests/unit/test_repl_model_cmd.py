"""REPL /model — list, hot-swap, managed restart, persistence."""
from __future__ import annotations

import asyncio
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


_REBIND_UNSET = object()  # mirrors LLMClient._REBIND_UNSET — see rebind_endpoint below


class FakeLLM:
    def __init__(self, model="model-a", base_url="http://localhost:8081/v1", mode="native",
                 provider_type="vllm"):
        self.config = SimpleNamespace(base_url=base_url, model=model, provider_type=provider_type)
        self._mode = mode
        # (base_url, api_key, extra_headers, provider_type) per cross-endpoint rebind — the
        # provider_type is recorded, not swallowed: it is the speed ledger's key.
        self.rebinds: list = []

    async def detect_capabilities(self):
        return SimpleNamespace(tool_call_mode=self._mode)

    def rebind_endpoint(self, base_url, *, api_key=None, extra_headers=None,
                        provider_type=_REBIND_UNSET):
        # Mirrors LLMClient.rebind_endpoint's observable effect: re-point at the new server. The
        # sentinel default is load-bearing — OMITTED means "leave provider_type unchanged" while
        # an explicit None means "set it to unknown runtime". Collapsing the two (defaulting to
        # None) would let a call site that DROPS provider_type still look correct here, while in
        # production the client keeps the old endpoint's type and files speed samples under the
        # wrong ledger key.
        self.rebinds.append((base_url, api_key, extra_headers, provider_type))
        self.config.base_url = base_url
        if provider_type is not _REBIND_UNSET:
            self.config.provider_type = provider_type  # ledger key follows the target endpoint


class FakeBoxChannel(FakeChannel):
    """FakeChannel + the box-mode surface `_run_with_box` drives: start/stop, the control-queue
    hand-off and the frame hooks. Input arrives the way the real prompt_toolkit box delivers it —
    through the queue the REPL hands us, with Ctrl+C going through the on_interrupt callback."""

    def __init__(self):
        super().__init__()
        self.ctrl_q = None
        self.on_interrupt = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def start_input_box(self, ctrl_queue, on_interrupt) -> None:
        self.ctrl_q, self.on_interrupt = ctrl_queue, on_interrupt

    async def stop_input_box(self) -> None:
        pass

    async def box_echo_prompt(self, text, annotation="") -> None:
        pass

    def box_set_queued(self, n: int) -> None:
        pass

    def box_notify_working(self, on: bool) -> None:
        pass


async def _until(pred, timeout: float = 2.0) -> None:
    """Poll until `pred()` holds, with a HARD bound — a wedged coordinator fails this test
    instead of hanging the suite."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        assert loop.time() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


def _repl(tmp_path, harness, live, channel=None):
    channel = channel if channel is not None else FakeChannel()
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


def _harness(server=None, audit_log_path=None, extra_endpoints=None):
    return HarnessConfig(
        provider=ProviderConfig(
            provider_type="vllm",
            base_url="http://localhost:8081/v1",
            default_model="model-a",
            available_models=["model-a"],
        ),
        org=OrgConfig(default_model="model-a", audit_log_path=audit_log_path),
        server=server,
        extra_endpoints=extra_endpoints or [],
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
    # #129: the overlay write outlives the session — the confirmation must SAY it persisted.
    assert "saved as default — persists across restarts" in channel.messages[-1]


@pytest.mark.asyncio
async def test_model_hotswap_stays_silent_when_persistence_fails(tmp_path, monkeypatch):
    """#129: the 'saved as default' claim is earned, not decorative — a failed overlay write
    must NOT be announced as persisted (the session still switched)."""
    from localharness.cli import model_ops

    repl, channel, agent = _repl(tmp_path, _harness(), live=["model-a", "model-b"])

    async def _boom(*a, **k):
        raise RuntimeError("overlay is read-only")
    monkeypatch.setattr(model_ops, "persist_default_model", _boom)

    await repl._handle_slash("/model model-b")
    assert agent._llm.config.model == "model-b"          # the live swap still happened
    joined = "\n".join(channel.messages)
    assert "persisting the new default failed" in joined
    assert "saved as default" not in joined


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
async def test_model_list_arg_lists_instead_of_erroring(tmp_path):
    """/help describes /model as "List available models" — typing "/model list" is the
    natural reading and must behave like bare /model (new-user papercut caught in the
    full-session dogfood), never 'Unknown model \\'list\\''."""
    repl, channel, agent = _repl(tmp_path, _harness(), live=["model-a", "model-b"])
    await repl._handle_slash("/model list")
    assert agent._llm.config.model == "model-a"  # no swap happened
    assert "Unknown model" not in channel.messages[-1]
    assert "Models:" in channel.messages[-1] and "model-b" in channel.messages[-1]


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
async def test_ctrl_c_during_a_slow_model_swap_never_exits_the_session(tmp_path):
    """A /model swap blocks the single box coordinator for as long as it runs (a managed restart:
    minutes). Ctrl+C presses land in the control queue meanwhile — invisible, since the loop
    cannot answer them — and drain the INSTANT the swap finishes: two of them armed and then
    exited the session the user had just spent minutes waiting for. A press from that blocked
    window is stale and must not count toward the arm-then-exit ladder; a fresh one still does."""
    channel = FakeBoxChannel()
    repl, _, agent = _repl(tmp_path, _harness(), live=["model-a", "model-b"], channel=channel)
    entered, gate = asyncio.Event(), asyncio.Event()

    async def slow_live_models(base_url):  # stands in for the swap's blocking probe/restart
        entered.set()
        await gate.wait()
        return ["model-a", "model-b"], True

    repl._live_models = slow_live_models
    loop_task = asyncio.ensure_future(repl._run_with_box())
    try:
        await _until(lambda: channel.ctrl_q is not None)
        channel.ctrl_q.put_nowait(("submit", "/model model-b"))
        await asyncio.wait_for(entered.wait(), timeout=2.0)  # the swap now owns the coordinator

        channel.on_interrupt()  # impatient user: nothing is happening on screen…
        channel.on_interrupt()
        assert repl._sigint_armed is False, "the blocked loop cannot have answered either press"

        gate.set()
        await _until(lambda: any("Switched to model-b" in m for m in channel.messages))
        await _until(lambda: channel.ctrl_q.empty())  # the backlog drains here
        assert not loop_task.done(), "a Ctrl+C from the wait must not end the session"
        assert repl._sigint_armed is False
        assert not any("Ctrl+C again" in m for m in channel.messages)
        assert agent._llm.config.model == "model-b"

        # …and the ladder itself still works for presses made NOW: arm, then exit.
        channel.on_interrupt()
        await _until(lambda: repl._sigint_armed)
        assert any("Ctrl+C again" in m for m in channel.messages)
        channel.on_interrupt()
        await asyncio.wait_for(loop_task, timeout=2.0)
    finally:
        loop_task.cancel()


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


def _repl_with_ctx(tmp_path, tc, max_ctx=131_072, live=("model-a", "model-b"), overrides=None):
    channel = FakeChannel()
    ctx = SimpleNamespace(_token_counter=tc, max_context_tokens=max_ctx)
    agent = SimpleNamespace(_llm=FakeLLM(), _ctx=ctx)
    if overrides is not None:
        # #132: the per-model pin map lives on the running agent's ContextConfig.
        from localharness.config.models import ContextConfig
        agent._config = SimpleNamespace(
            context=ContextConfig(max_context_tokens=max_ctx, model_context_overrides=overrides)
        )
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


# --- 0.10.0 model tree: cross-endpoint /model switch (switch to a peer server) --- #


def _xrepl(tmp_path, harness, models_by_url):
    """A REPL whose /model discovery is endpoint-aware: `models_by_url` maps a base_url to either
    a list of served model ids (reachable) OR the string 'unreachable'/'malformed' to exercise the
    graceful-skip paths. No real HTTP — the shared list_live_models probe is stubbed per endpoint."""
    from localharness.cli import model_ops

    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM())
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(), agent_loop=agent, channel=channel,
        bus=SimpleNamespace(), config_dir=tmp_path, harness_config=harness,
    )
    probed: list[str] = []

    async def _live_models(base_url):
        probed.append(base_url)
        v = models_by_url.get(base_url)
        if v == "unreachable":
            return [], False
        if v == "malformed":
            raise model_ops.MalformedModelListError(f"{base_url} not an OpenAI model list")
        return list(v or []), True

    repl._live_models = _live_models
    return repl, channel, agent, probed


def _peer(name="ollama-local", base_url="http://localhost:11434/v1", ptype="ollama"):
    from localharness.config.models import EndpointRef
    return EndpointRef(name=name, base_url=base_url, provider_type=ptype, api_key="none")


@pytest.mark.asyncio
async def test_model_cross_endpoint_switch_rebinds_refits_persists(tmp_path):
    """A `/model <name>` whose name is served ONLY on a peer endpoint must: rebind the client at
    the peer base_url, set config.model, refit the counter with the PEER's provider_type, and
    persist the peer endpoint (additive active_endpoint), NOT the primary default-model path."""
    peer = _peer()
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent, probed = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:11434/v1": ["gpt-oss:20b"]},
    )

    refresh_calls: list = []

    async def _refresh(model, *, base_url=None, provider_type=None):
        refresh_calls.append((model, base_url, provider_type))
        return ""

    repl._refresh_token_counter = _refresh

    persist_calls: list = []

    async def _persist_active(endpoint, model):
        persist_calls.append((endpoint.base_url, endpoint.provider_type, model))

    repl._persist_active_endpoint = _persist_active

    await repl._handle_slash("/model gpt-oss:20b")

    # rebound at the peer, model + base_url now point at the peer server
    assert agent._llm.rebinds and agent._llm.rebinds[0][0] == "http://localhost:11434/v1"
    assert agent._llm.config.model == "gpt-oss:20b"
    assert agent._llm.config.base_url == "http://localhost:11434/v1"
    # the CLIENT's provider_type moved too (passed explicitly, not left at the old vLLM) — this
    # is the speed ledger's key, so a stale one files this peer's samples under `vllm:…`
    assert agent._llm.rebinds[0][3] == "ollama"
    assert agent._llm.config.provider_type == "ollama"
    # counter refit used the PEER's provider_type (ollama → labeled approximate), not the old vLLM
    assert refresh_calls == [("gpt-oss:20b", "http://localhost:11434/v1", "ollama")]
    # persisted the peer endpoint (active_endpoint), not provider.default_model
    assert persist_calls == [("http://localhost:11434/v1", "ollama", "gpt-oss:20b")]
    assert "Switched to gpt-oss:20b on ollama-local" in channel.messages[-1]


@pytest.mark.asyncio
async def test_model_same_endpoint_switch_does_not_rebind_regression(tmp_path):
    """REGRESSION guard: with a peer configured, switching to a model on the CURRENT endpoint
    still takes the hot-swap path — NO rebind_endpoint, persisted via the primary default path."""
    peer = _peer()
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent, probed = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a", "model-b"],
         "http://localhost:11434/v1": ["gpt-oss:20b"]},
    )

    await repl._handle_slash("/model model-b")

    assert agent._llm.rebinds == []  # never rebound — same endpoint
    assert agent._llm.config.model == "model-b"
    assert agent._llm.config.base_url == "http://localhost:8081/v1"  # unchanged
    # landed on the primary → today's default-model overlay path
    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay["provider"]["default_model"] == "model-b"
    assert "active_endpoint" not in overlay  # no peer record written for a primary switch


@pytest.mark.asyncio
async def test_model_cross_endpoint_persist_writes_active_endpoint_overlay(tmp_path):
    """End-to-end persistence (no spy): a peer switch writes an ADDITIVE, atomic active_endpoint
    overlay and never mutates provider.default_model / server.model — a real, loadable record."""
    peer = _peer(base_url="http://localhost:11434/v1", ptype="ollama")
    srv = ManagedServerConfig(binary="/x/vllm", model="model-a")
    harness = _harness(server=srv, extra_endpoints=[peer])
    repl, channel, agent, probed = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:11434/v1": ["gpt-oss:20b"]},
    )

    await repl._handle_slash("/model gpt-oss:20b")

    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay["active_endpoint"]["base_url"] == "http://localhost:11434/v1"
    assert overlay["active_endpoint"]["provider_type"] == "ollama"
    assert overlay["active_endpoint"]["model"] == "gpt-oss:20b"
    # primary/server identity untouched (reusing default-model persistence would corrupt these)
    assert "provider" not in overlay or overlay.get("provider", {}).get("default_model") != "gpt-oss:20b"
    assert "server" not in overlay or overlay.get("server", {}).get("model") != "gpt-oss:20b"
    # the written overlay must still LOAD as a valid HarnessConfig (extra='forbid' would reject a
    # stray key) — proving the field is load-bearing now, before start-side resume is wired.
    from localharness.config.models import HarnessConfig
    from localharness.config.overlay import deep_merge
    HarnessConfig.model_validate(deep_merge(harness.model_dump(mode="python"), overlay))


# --- Phase C2: cold-peer cross-framework heavy-swap (GPU-lock) --- #


def _cold_peer(name="llamacpp-local", base_url="http://127.0.0.1:8080/v1", served="qwen-gguf"):
    """A peer that is NOT running but carries a lifecycle launch spec the harness can bring up (a
    llama.cpp llama-server). gpu=True → it participates in the GPU-lock."""
    from localharness.config.models import EndpointRef
    return EndpointRef(
        name=name, base_url=base_url, provider_type="llamacpp", gpu=True,
        lifecycle=ManagedServerConfig(
            runtime="llamacpp", launch="binary", binary="/x/llama-server",
            model="/x/model.gguf", port=8080, gpu=True,
            extra_args=["-c", "32768", "--parallel", "1", "-ngl", "99", "--jinja", "-a", served],
        ),
    )


def _heavy_primary(model="model-a"):
    """A harness-managed GPU vLLM (docker) — the incumbent heavy the GPU-lock must stop before a
    cold peer can launch. gpu defaults True."""
    return ManagedServerConfig(launch="docker", docker_image="vllm:latest", model=model)


def test_cold_lifecycle_targets_surfaces_cold_launchable_peer(tmp_path):
    """A down peer WITH a lifecycle block is offered (by its configured served-name); an attach-only
    peer (lifecycle=None) is not; a lifecycle peer already probed live is left to the live path."""
    cold = _cold_peer(served="qwen-gguf")
    attach = _peer(name="ollama", base_url="http://localhost:11434/v1")  # lifecycle=None
    harness = _harness(server=_heavy_primary(), extra_endpoints=[cold, attach])
    repl, _, _, _ = _xrepl(tmp_path, harness, {})
    assert repl._cold_lifecycle_targets(["model-a"], {}) == {"qwen-gguf": cold}
    # a lifecycle peer that IS up (present as a peer_target value) is NOT re-offered as cold
    assert repl._cold_lifecycle_targets(["model-a"], {"qwen-gguf": cold}) == {}


@pytest.mark.asyncio
async def test_model_cold_peer_heavy_swap_stops_incumbent_launches_rebinds(tmp_path, monkeypatch):
    """/model <cold-peer> → GPU-lock: verified-stop the incumbent heavy (docker vLLM), LAUNCH the
    cold llama.cpp peer ourselves (start→wait), rebind the client at the peer, and move _active_heavy.
    Ordering: stop-incumbent → start-peer → wait-peer."""
    from localharness.provider import server as managed_server
    from localharness.provider import lifecycle

    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)  # no real settle in tests
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])
    calls: list[str] = []
    monkeypatch.setattr(managed_server, "stop_server",
                        lambda cfg, launch="binary": calls.append(f"stop:{launch}") or True)
    monkeypatch.setattr(managed_server, "start_server",
                        lambda cfg, cmd: calls.append("start") or 4321)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        calls.append("wait")
        return ["qwen-gguf"]

    monkeypatch.setattr(managed_server, "wait_ready", fake_wait_ready)

    cold = _cold_peer(served="qwen-gguf")
    harness = _harness(server=_heavy_primary(), extra_endpoints=[cold])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],       # primary (current) serving model-a
         "http://127.0.0.1:8080/v1": "unreachable"},    # the peer is COLD
    )

    async def _refresh(model, *, base_url=None, provider_type=None):
        return ""
    repl._refresh_token_counter = _refresh
    persisted: list = []

    async def _persist_active(endpoint, model):
        persisted.append((endpoint.base_url, model))
    repl._persist_active_endpoint = _persist_active

    assert repl._active_heavy is not None and repl._active_heavy[0] is harness.server  # seeded

    await repl._handle_slash("/model qwen-gguf")

    assert calls == ["stop:docker", "start", "wait"]          # incumbent stopped BEFORE peer launch
    assert agent._llm.config.model == "qwen-gguf"
    assert agent._llm.config.base_url == "http://127.0.0.1:8080/v1"
    assert agent._llm.rebinds and agent._llm.rebinds[0][0] == "http://127.0.0.1:8080/v1"
    assert agent._llm.rebinds[0][3] == "llamacpp"          # ledger key follows the launched peer
    assert agent._llm.config.provider_type == "llamacpp"
    assert repl._active_heavy == (cold.lifecycle, "http://127.0.0.1:8080/v1")  # GPU occupant moved
    assert persisted == [("http://127.0.0.1:8080/v1", "qwen-gguf")]
    assert "Switched to qwen-gguf on llamacpp-local" in channel.messages[-1]


@pytest.mark.asyncio
async def test_model_cold_peer_launch_failure_restores_incumbent(tmp_path, monkeypatch):
    """If launching the cold peer FAILS after the incumbent was stopped, the harness re-activates the
    incumbent (best-effort) so the box is never left with nothing serving; the client stays put."""
    from localharness.provider import server as managed_server
    from localharness.provider import lifecycle

    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])
    events: list[str] = []
    monkeypatch.setattr(managed_server, "stop_server",
                        lambda cfg, launch="binary": events.append(f"stop:{launch}") or True)
    monkeypatch.setattr(managed_server, "start_server",
                        lambda cfg, cmd: events.append("start") or 4321)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        # the PEER (8080) never comes up; the RESTORE of the primary (8081) succeeds
        if base_url.startswith("http://127.0.0.1:8080"):
            events.append("peer-wait-FAIL")
            raise TimeoutError("llama-server never became ready")
        events.append("restore-wait-ok")
        return ["model-a"]

    monkeypatch.setattr(managed_server, "wait_ready", fake_wait_ready)

    cold = _cold_peer()
    harness = _harness(server=_heavy_primary(), extra_endpoints=[cold])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"], "http://127.0.0.1:8080/v1": "unreachable"},
    )

    async def _refresh(model, *, base_url=None, provider_type=None):
        return ""
    repl._refresh_token_counter = _refresh

    await repl._handle_slash("/model qwen-gguf")

    # orphan-safe restore: incumbent stopped → peer launched → peer FAILED → peer torn down FIRST
    # (stop:binary — else it orphans on the shared pidfile) → incumbent relaunched + ready.
    assert events == ["stop:docker", "start", "peer-wait-FAIL", "stop:binary", "start", "restore-wait-ok"]
    assert agent._llm.config.model == "model-a"          # client never moved to the failed peer
    assert agent._llm.rebinds == []
    assert repl._active_heavy == (harness.server, "http://localhost:8081/v1")  # restored occupant
    assert "Restored" in channel.messages[-1]


@pytest.mark.asyncio
async def test_model_swap_back_to_primary_stops_cold_peer_first(tmp_path, monkeypatch):
    """The b2 case: while ON a cold-launched llama.cpp peer (the real GPU occupant), /model <primary
    registry model> must stop the PEER before restarting the primary vLLM. Proves the GPU-lock
    generalizes to the managed-restart branch — the primary is DOWN, the peer holds the accelerator."""
    from localharness.provider import server as managed_server
    from localharness.provider import lifecycle

    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    stops: list[str] = []
    monkeypatch.setattr(managed_server, "stop_server",
                        lambda cfg, launch="binary": stops.append(launch) or True)
    monkeypatch.setattr(managed_server, "start_server", lambda cfg, cmd: 1234)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        return ["cached-b"]

    monkeypatch.setattr(managed_server, "wait_ready", fake_wait_ready)

    cold = _cold_peer()
    harness = _harness(server=_heavy_primary(), extra_endpoints=[cold])
    repl, channel, agent, _ = _xrepl(tmp_path, harness, {"http://localhost:8081/v1": ["model-a"]})
    # simulate we're currently ON the cold-launched peer: it's the GPU occupant + the bound endpoint
    repl._active_heavy = (cold.lifecycle, cold.base_url)
    agent._llm.config.base_url = cold.base_url
    agent._llm.config.model = "qwen-gguf"

    await repl._handle_slash("/model cached-b")

    assert stops == ["binary", "docker"]  # PEER stopped (GPU-lock) THEN primary (restart), in order
    assert agent._llm.config.model == "cached-b"
    assert repl._active_heavy == (harness.server, "http://localhost:8081/v1")


@pytest.mark.asyncio
async def test_model_swap_back_free_fail_reports_no_change(tmp_path, monkeypatch):
    """New Finding B: if freeing the current heavy peer FAILS during a swap-back, the primary is NOT
    (re)started and the user is told honestly — no misleading 'managed server is down' claim."""
    from localharness.provider import server as managed_server
    from localharness.provider import lifecycle

    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    started: list = []

    def _boom_stop(cfg, launch="binary"):
        if launch == "binary":                       # the peer we're trying to free won't die
            raise RuntimeError("llama-server won't die")
        return True

    monkeypatch.setattr(managed_server, "stop_server", _boom_stop)
    monkeypatch.setattr(managed_server, "start_server", lambda cfg, cmd: started.append(cmd) or 1)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        return ["cached-b"]

    monkeypatch.setattr(managed_server, "wait_ready", fake_wait_ready)

    cold = _cold_peer()
    harness = _harness(server=_heavy_primary(), extra_endpoints=[cold])
    repl, channel, agent, _ = _xrepl(tmp_path, harness, {"http://localhost:8081/v1": ["model-a"]})
    repl._active_heavy = (cold.lifecycle, cold.base_url)   # currently on the cold peer
    agent._llm.config.base_url = cold.base_url
    agent._llm.config.model = "qwen-gguf"

    await repl._handle_slash("/model cached-b")

    assert started == []                               # the primary was NEVER (re)started
    assert agent._llm.config.model == "qwen-gguf"      # session model unchanged
    assert "Could not free the GPU" in channel.messages[-1]


def test_cold_alias_colliding_with_downloaded_is_not_offered(tmp_path):
    """Finding 6: a cold peer alias equal to a downloaded checkpoint name is dropped from cold_target
    (passed local_choices = live + downloaded), so it can't shadow the local name or duplicate a menu
    number into the wrong (cross-framework launch) branch."""
    cold = _cold_peer(served="collide")
    harness = _harness(server=_heavy_primary(), extra_endpoints=[cold])
    repl, _, _, _ = _xrepl(tmp_path, harness, {})
    assert repl._cold_lifecycle_targets(["model-a", "collide"], {}) == {}   # collides → dropped
    assert repl._cold_lifecycle_targets(["model-a"], {}) == {"collide": cold}  # no collision → offered


@pytest.mark.asyncio
async def test_backward_compat_no_extra_endpoints_probes_only_primary(tmp_path):
    """Backward compat: with extra_endpoints=[], discovery probes ONLY the primary — zero extra
    I/O, resolution unchanged (single-endpoint users see no behavior change)."""
    harness = _harness()  # no peers
    repl, channel, agent, probed = _xrepl(
        tmp_path, harness, {"http://localhost:8081/v1": ["model-a", "model-b"]},
    )

    await repl._handle_slash("/model")  # list
    assert probed == ["http://localhost:8081/v1"]  # only the primary endpoint was probed


@pytest.mark.asyncio
async def test_unreachable_peer_skipped_with_note_never_raises(tmp_path):
    """An unreachable peer is SKIPPED with a note (never raises), and a peer-only name a dead peer
    would have served resolves as Unknown, not a crash."""
    peer = _peer()
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent, probed = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:11434/v1": "unreachable"},
    )

    await repl._handle_slash("/model")  # must not raise
    joined = "\n".join(channel.messages)
    assert "ollama-local" in joined and "unreachable" in joined

    await repl._handle_slash("/model gpt-oss:20b")  # peer-only name, peer is dead
    assert "Unknown model" in channel.messages[-1]
    assert agent._llm.rebinds == []  # never attempted a rebind to a dead peer


@pytest.mark.asyncio
async def test_malformed_peer_skipped_with_note(tmp_path):
    """A peer that responds but not with an OpenAI model list is skipped with its OWN note (the
    #38 taxonomy), never a bare 'no models' and never a raise."""
    peer = _peer(name="lmstudio-local", base_url="http://localhost:1234/v1", ptype="lmstudio")
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent, probed = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:1234/v1": "malformed"},
    )

    await repl._handle_slash("/model")
    joined = "\n".join(channel.messages)
    assert "lmstudio-local" in joined and "not an OpenAI-compatible model list" in joined


# --- FIX-PASS: critic FIX-FIRST findings (#1, #2, #6, #7) --- #


def test_list_live_models_swallows_invalid_url():
    """#1: httpx.InvalidURL (a typo'd / non-numeric-port base_url) is NOT an httpx.RequestError, so
    the shared probe must catch it explicitly and report unreachable ([], False) — never propagate
    and crash the /model call."""
    from localharness.cli import model_ops
    assert model_ops.list_live_models("http://localhost:notaport/v1") == ([], False)


@pytest.mark.asyncio
async def test_malformed_peer_base_url_never_crashes_session(tmp_path, monkeypatch):
    """#1 end-to-end: a typo'd peer base_url (non-numeric port) makes the REAL probe raise
    httpx.InvalidURL. `/model` (bare list) AND `/model <primary-live>` must BOTH succeed, the bad
    peer is skipped with a note, and NOTHING propagates to crash the REPL."""
    import httpx
    real_get = httpx.get

    def fake_get(url, *a, **k):
        if "notaport" in url:
            return real_get(url, *a, **k)  # let httpx raise the genuine InvalidURL

        class _R:
            def json(self):
                return {"data": [{"id": "model-a"}, {"id": "model-b"}]}

        return _R()

    monkeypatch.setattr(httpx, "get", fake_get)

    peer = _peer(name="typo-peer", base_url="http://localhost:notaport/v1", ptype="ollama")
    harness = _harness(extra_endpoints=[peer])
    # No fake _live_models — exercise the REAL model_ops.list_live_models (the InvalidURL site).
    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM())
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(), agent_loop=agent, channel=channel,
        bus=SimpleNamespace(), config_dir=tmp_path, harness_config=harness,
    )

    await repl._handle_slash("/model")  # bare list — must NOT raise
    assert "typo-peer" in "\n".join(channel.messages)  # skipped WITH a note, not a crash

    await repl._handle_slash("/model model-b")  # a primary-served name — must still switch cleanly
    assert agent._llm.config.model == "model-b"
    assert agent._llm.rebinds == []  # local hot-swap, never touched the dead peer


@pytest.mark.asyncio
async def test_second_same_peer_hop_refits_with_peer_provider_type(tmp_path, monkeypatch):
    """#2: after primary→peerA-modelX (cross-endpoint), a SECOND hop peerA-modelX→peerA-modelY takes
    the same-endpoint hot-swap branch. Its counter refit must resolve the PEER's provider_type
    (ollama) from the current base_url, NOT the stale primary vllm — else the rebind hard-raises and
    a clean switch is mislabeled 'couldn't rebind'. Persistence must still record the peer endpoint."""
    from localharness.config.overlay import load_overlay

    peer = _peer(name="ollama-local", base_url="http://localhost:11434/v1", ptype="ollama")
    harness = _harness(extra_endpoints=[peer])

    rebind_calls: list = []

    class _RecordingTC:
        def rebind(self, base_url, model, ptype):
            rebind_calls.append((base_url, model, ptype))

    ctx = SimpleNamespace(_token_counter=_RecordingTC(), max_context_tokens=131_072)
    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM(), _ctx=ctx)
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(), agent_loop=agent, channel=channel,
        bus=SimpleNamespace(), config_dir=tmp_path, harness_config=harness,
    )

    async def _live_models(base_url):
        # after the first switch the client sits on peerA, which lists BOTH peer models as `live`.
        if base_url == "http://localhost:11434/v1":
            return ["modelX", "modelY"], True
        return ["model-a"], True  # primary

    repl._live_models = _live_models
    monkeypatch.setattr("localharness.agent.context.probe_served_window", lambda *a, **k: None)

    await repl._handle_slash("/model modelX")  # cross-endpoint (peer-only name) → sits on peerA
    assert agent._llm.config.base_url == "http://localhost:11434/v1"
    assert rebind_calls[-1] == ("http://localhost:11434/v1", "modelX", "ollama")  # explicit ptype

    await repl._handle_slash("/model modelY")  # same-peer hot-swap → ptype must RESOLVE to ollama
    assert agent._llm.config.model == "modelY"
    assert rebind_calls[-1] == ("http://localhost:11434/v1", "modelY", "ollama")  # NOT stale vllm
    assert "could not rebind" not in "\n".join(channel.messages).lower()  # no spurious failure note

    # persistence on the same-peer hop still records the PEER endpoint (resolved by base_url).
    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay["active_endpoint"]["model"] == "modelY"
    assert overlay["active_endpoint"]["provider_type"] == "ollama"


@pytest.mark.asyncio
async def test_persist_landed_unknown_base_url_persists_nothing(tmp_path):
    """#6: an unknown base_url (neither the primary NOR a configured peer) must persist NOTHING —
    never fall back to persist_default_model, which would rewrite provider/server.model and make the
    next `start` serve the peer model on the managed GPU."""
    peer = _peer(name="ollama-local", base_url="http://localhost:11434/v1", ptype="ollama")
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent = _repl(tmp_path, harness, live=["model-a"])

    called = {"default": 0, "active": 0}

    async def _default(model):
        called["default"] += 1

    async def _active(endpoint, model):
        called["active"] += 1

    repl._persist_default_model = _default
    repl._persist_active_endpoint = _active

    await repl._persist_landed("http://localhost:9999/v1", "ghost-model")  # unknown endpoint

    assert called == {"default": 0, "active": 0}  # persisted NOTHING (no dangerous default fallback)
    assert not (tmp_path / "overrides.yaml").exists()  # nothing written


@pytest.mark.asyncio
async def test_peer_vs_peer_collision_emits_note(tmp_path):
    """#7: the same model name served on TWO peers — the first-configured peer wins (kept), and the
    HIDDEN duplicate on the later peer is NOTED so it's discoverable, not silently dropped."""
    peer_a = _peer(name="peer-a", base_url="http://localhost:11434/v1", ptype="ollama")
    peer_b = _peer(name="peer-b", base_url="http://localhost:1234/v1", ptype="lmstudio")
    harness = _harness(extra_endpoints=[peer_a, peer_b])
    repl, channel, agent, probed = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:11434/v1": ["shared-model"],
         "http://localhost:1234/v1": ["shared-model"]},
    )

    await repl._handle_slash("/model")
    joined = "\n".join(channel.messages)
    assert "shared-model" in joined  # first-configured (peer-a) copy is listed
    assert "peer-b" in joined and "hidden by" in joined and "peer-a" in joined  # duplicate flagged


# --- Step 2: /model rendered as a grouped-by-endpoint TREE (0.10.0 model tree) --- #


@pytest.mark.asyncio
async def test_model_tree_groups_primary_and_peer_sections(tmp_path):
    """Bare /model with a peer configured renders a grouped tree: a primary vLLM header carrying
    its live model (marked active), then an Ollama peer header with the peer-only model indented
    UNDER it (not under the vLLM header)."""
    peer = _peer()  # ollama-local @ 11434
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:11434/v1": ["gpt-oss:20b"]},
    )

    await repl._handle_slash("/model")
    out = channel.messages[-1]
    # primary provider section names its framework + host; its live model is marked active
    assert "▸ vLLM · localhost:8081" in out
    assert "model-a" in out and "[active]" in out and "●" in out
    # peer section names the peer framework + host; the peer-only model is grouped beneath it
    assert "▸ Ollama · localhost:11434" in out
    assert "gpt-oss:20b" in out
    # the peer model sits under the Ollama header, the live model under the vLLM header
    assert out.index("▸ vLLM") < out.index("model-a") < out.index("▸ Ollama") < out.index("gpt-oss:20b")
    # the one-Enter trailer + numbering hint is preserved
    assert "scroll the menu and press Enter" in out


@pytest.mark.asyncio
async def test_model_tree_continuous_numbering_resolves_peer_model(tmp_path):
    """The tree numbers models in one continuous sequence across groups; the number shown for a
    peer model resolves THAT model via /model <number> (cross-endpoint switch), proving the
    displayed numbers match the resolver's choices."""
    peer = _peer()
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:11434/v1": ["gpt-oss:20b"]},
    )

    await repl._handle_slash("/model")
    assert "2. gpt-oss:20b" in channel.messages[-1]  # live model-a is 1, peer model is 2

    await repl._handle_slash("/model 2")
    assert agent._llm.config.model == "gpt-oss:20b"
    assert agent._llm.config.base_url == "http://localhost:11434/v1"  # rebound to the peer endpoint


@pytest.mark.asyncio
async def test_model_tree_feeds_peer_models_into_picker_cache(tmp_path):
    """After a bare /model the picker cache (Tab menu) also holds the peer's models, each tagged
    with the peer framework + host — so the menu can switch across endpoints, not just list them."""
    peer = _peer()
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:11434/v1": ["gpt-oss:20b"]},
    )

    await repl._handle_slash("/model")
    assert ("model-a", "serving now") in repl._model_cache
    assert ("gpt-oss:20b", "Ollama · localhost:11434") in repl._model_cache


@pytest.mark.asyncio
async def test_model_tree_marks_cold_peer_plainly(tmp_path):
    """A configured-but-unreachable peer is marked plainly beneath the tree (a note naming the
    endpoint + 'unreachable'); the reachable primary section still renders its model normally."""
    peer = _peer(name="ollama-local")
    harness = _harness(extra_endpoints=[peer])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness,
        {"http://localhost:8081/v1": ["model-a"],
         "http://localhost:11434/v1": "unreachable"},
    )

    await repl._handle_slash("/model")
    out = channel.messages[-1]
    assert "▸ vLLM · localhost:8081" in out and "model-a" in out
    assert "ollama-local" in out and "unreachable" in out  # cold peer marked plainly


@pytest.mark.asyncio
async def test_model_flat_listing_unchanged_without_peers(tmp_path):
    """Backward compat: with no extra_endpoints the listing stays the flat single-endpoint style
    (no ▸ endpoint headers) — existing single-endpoint output/behavior is unchanged."""
    repl, channel, _ = _repl(tmp_path, _harness(), live=["model-a", "model-b"])

    await repl._handle_slash("/model")
    out = channel.messages[-1]
    assert "▸" not in out  # no grouped-tree headers
    assert out.startswith("Models:")
    assert "1. model-a" in out and "2. model-b" in out


async def test_swap_tradeoff_note_from_ledger(tmp_path):
    """Heavy-swap announcements name the verified tradeoff (ledger medians only): both sides,
    target-only, current-only, or silence — never an estimate."""
    from localharness.provider.speed_stats import record_tps

    repl, channel, agent = _repl(tmp_path, _harness(), ["model-a"])
    path = tmp_path / "speed_stats.json"  # config_dir=tmp_path roots the ledger here
    record_tps(path, "vllm", "cur", 31.0)
    record_tps(path, "ollama", "tgt", 12.0)
    assert repl._swap_tradeoff_note("vllm", "cur", "ollama", "tgt") \
        == " Speed: 31.0 t/s measured → 12.0 t/s measured."
    assert repl._swap_tradeoff_note("vllm", "cur", "ollama", "nope") \
        == " Current model runs 31.0 t/s measured; target unmeasured."
    assert repl._swap_tradeoff_note("vllm", "nope", "ollama", "tgt") \
        == " Target: 12.0 t/s measured."
    assert repl._swap_tradeoff_note(None, "cur", None, "tgt") == ""


async def test_swap_tradeoff_note_is_colorized_by_band(tmp_path):
    """Both callers send this line with colorize=True, and terminal._RATE_NOTE only bands
    'N t/s measured'. Two of the three shapes put the number before a bare 't/s', so the one
    line warning the user they are about to spend minutes loading a SLOWER model rendered with
    no band at all. Every rate a shape names must colorize, in its own speed band."""
    from localharness.channels.terminal import _colorize_rate_notes
    from localharness.provider.speed_stats import record_tps

    repl, _channel, _agent = _repl(tmp_path, _harness(), ["model-a"])
    path = tmp_path / "speed_stats.json"
    record_tps(path, "vllm", "cur", 31.0)   # green band (>30)
    record_tps(path, "ollama", "tgt", 12.0)  # red band (<20)

    both = _colorize_rate_notes(repl._swap_tradeoff_note("vllm", "cur", "ollama", "tgt"))
    assert both.spans, "no styled spans — the note never colorizes"
    assert [str(s.style) for s in both.spans] == ["green", "bold red"], "both rates band"

    tgt_only = _colorize_rate_notes(repl._swap_tradeoff_note("vllm", "nope", "ollama", "tgt"))
    assert [str(s.style) for s in tgt_only.spans] == ["bold red"]
    cur_only = _colorize_rate_notes(repl._swap_tradeoff_note("vllm", "cur", "ollama", "nope"))
    assert [str(s.style) for s in cur_only.spans] == ["green"]


@pytest.mark.asyncio
async def test_model_hotswap_refit_honors_a_per_model_context_pin(tmp_path, monkeypatch):
    """#132: a per-model pin is CONFIGURATION and outranks the served-window probe. Without it
    the swap refits to the probe's 32K window; with it the pinned budget wins and is disclosed."""
    from localharness.agent.context import TokenCounter

    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    tc = TokenCounter(base_url="http://localhost:8081/v1", model="model-a", provider_type="vllm")
    repl, channel, agent, ctx = _repl_with_ctx(
        tmp_path, tc, max_ctx=131_072, overrides={"model-b": 24_000}
    )

    def _probe_must_not_decide(*a, **k):
        raise AssertionError("the pin must short-circuit the served-window probe")
    monkeypatch.setattr("localharness.agent.context.probe_served_window", _probe_must_not_decide)

    await repl._handle_slash("/model model-b")

    assert ctx.max_context_tokens == 24_000
    assert "pinned to 24,000 tokens for model-b" in "\n".join(channel.messages)


@pytest.mark.asyncio
async def test_model_hotswap_refit_unchanged_when_no_pin_matches(tmp_path, monkeypatch):
    """#132: a map that does not name THIS model changes nothing — the probe still decides
    (exact-name match only; no globbing, no accidental cross-model pinning)."""
    from localharness.agent.context import TokenCounter
    from localharness.cli.init_cmd import _fit_context_tokens

    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    tc = TokenCounter(base_url="http://localhost:8081/v1", model="model-a", provider_type="vllm")
    repl, channel, agent, ctx = _repl_with_ctx(
        tmp_path, tc, max_ctx=131_072, overrides={"some-other-model": 24_000}
    )

    monkeypatch.setattr("localharness.agent.context.probe_served_window", lambda *a, **k: 32_768)
    await repl._handle_slash("/model model-b")

    assert ctx.max_context_tokens == _fit_context_tokens(32_768)
    assert "pinned to" not in "\n".join(channel.messages)
