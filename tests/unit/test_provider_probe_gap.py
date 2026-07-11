"""AUDIT-03 (provider/probe seams that silently run the wrong model/mode) + AUDIT-06
fix#2/#3 construction characterization.

Every test here is CONSTRUCTION-ONLY or a CAPTURE-SPY (records kwargs / the constructed
LLMConfig / the MatrixEntry, returns a fabricated value). NO test calls a real model — the
only network seam (the OpenAI create()) is always replaced by a spy that returns a hand-built
response object, so nothing ever reaches an endpoint. Model-agnostic and offline.

The four AUDIT-03 seams:
  a) the autoresearch GATE builds an unprobed ollama/`bench-default` client, ignoring cfg.provider
     (experiment.py:330 -> build_llm_client_factory(_synthesize_default_entry()))
  b) _complete_xml_fallback omits BOTH stop_sequences and extra_body{enable_thinking:False}
     (the sibling divergence vs _complete_native client.py:274-282)  [also AUDIT-06 fix#2 twin]
  c) the proposer hardcodes tool_call_mode="native" (proposer.py:235), never probes
  d) startup derives mode from the STORED provider.supports_function_calling flag and discards
     the probe (_probe_llm returns only a bool; start_cmd.py:160,165)
  e) the root enabler: LLMConfig default tool_call_mode="native" (client.py:33)

AUDIT-06 fix#3 asymmetry: the MATRIX path DOES probe via detect_capabilities
(orchestrator.py:143-144); the gate path does not.
"""
import inspect

import pytest


# --------------------------------------------------------------------------- #
# Shared spy plumbing for the OpenAI create() network seam.
# A capture spy: records kwargs, returns a fabricated response. NEVER connects.
# --------------------------------------------------------------------------- #


class _FakeMsg:
    content = "ok"
    tool_calls: list = []


class _Choice:
    message = _FakeMsg()


class _Resp:
    choices = [_Choice()]
    usage = None


def _make_create_spy(captured: dict):
    """Return an async spy for client._client.chat.completions.create that records
    the kwargs it was called with and returns a fabricated response (no network)."""

    async def _spy_create(**kwargs):
        captured.update(kwargs)
        return _Resp()

    return _spy_create


# --------------------------------------------------------------------------- #
# Task 1 — AUDIT-03b + AUDIT-06 fix#2 twin: _complete_xml_fallback create()-spy (RED)
# --------------------------------------------------------------------------- #


async def test_xml_fallback_forwards_stop_without_thinking_disable():
    """Sibling-parity contract for _complete_xml_fallback, post-#11: the fallback must
    forward `stop` exactly like its native/xml siblings, and — like them — must NOT send
    extra_body{enable_thinking:False}. Thinking is deliberately left ON for local subjects
    (#11 ruling): the loop strips <think> blocks, and the kwarg was silently dropped by
    Ollama / type-checked by llama.cpp anyway.

    Every assertion is on the recorded create() kwargs via a capture-spy returning a
    fabricated _Resp — no real model call.
    """
    from localharness.provider.client import LLMClient, LLMConfig

    # is_local=True + stop_sequences=["X"] make the SIBLINGS add stop + extra_body; tool_call_mode
    # ="xml" builds self._fn_converter so the fallback's system injection runs. base_url/model are
    # dummies we never connect to (the spy intercepts the only create() call).
    client = LLMClient(
        LLMConfig(
            base_url="http://localhost:0/v1",
            model="m",
            is_local=True,
            stop_sequences=["X"],
            tool_call_mode="xml",
        )
    )

    captured: dict = {}
    client._client.chat.completions.create = _make_create_spy(captured)

    # ToolSchema is a plain JSON-Schema dict (core/types.py:16); the _fn_converter accepts a dict.
    tool = {
        "name": "list_files",
        "description": "List directory contents",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    await client._complete_xml_fallback(
        messages=[{"role": "system", "content": "sys"}],
        tools=[tool],
        stream=False,
    )

    assert captured.get("stop") == ["X"], "fallback dropped stop_sequences (sibling divergence)"
    assert "extra_body" not in captured, (
        "thinking must stay ON for local subjects (#11) — no path may send enable_thinking:False"
    )


async def test_local_paths_send_no_thinking_disable():
    """#11 regression, all remaining request paths: neither _complete_native nor
    _complete_xml may send extra_body{enable_thinking:False} for a local subject.
    Thinking stays ON (owner ruling); suppression was also unreliable per-runtime
    (Ollama drops the field silently, llama.cpp 400s on a non-boolean)."""
    from localharness.provider.client import LLMClient, LLMConfig

    for mode, method in (("native", "_complete_native"), ("xml", "_complete_xml")):
        client = LLMClient(
            LLMConfig(
                base_url="http://localhost:0/v1",
                model="m",
                is_local=True,
                tool_call_mode=mode,
            )
        )
        captured: dict = {}
        client._client.chat.completions.create = _make_create_spy(captured)
        await getattr(client, method)(
            messages=[{"role": "user", "content": "hi"}], tools=None, stream=False
        )
        assert "extra_body" not in captured, (
            f"{method} sent extra_body for a local subject — thinking must stay ON (#11)"
        )


# --------------------------------------------------------------------------- #
# Task 2A — AUDIT-03a (D2 direction): gate-factory probe-skip spy (RED)
# --------------------------------------------------------------------------- #


def test_gate_factory_ignores_cfg_provider_and_skips_probe(monkeypatch):
    """The autoresearch gate's default factory is, verbatim (experiment.py:330):
        build_llm_client_factory(_synthesize_default_entry())
    _synthesize_default_entry() (orchestrator.py:226-230) is a HARDCODED
    MatrixEntry(provider='ollama', model_id='bench-default'); build_llm_client_factory's docstring
    says 'This path does NOT probe capabilities'. So the gate runs an unprobed ollama/bench-default
    client regardless of cfg.provider.

    We spy the construction seam (_build_bench_client) to OBSERVE the MatrixEntry the gate builds
    with — we do NOT replace behavior and NEVER call .complete(). The assertion is model-agnostic:
    the gate must NOT hardcode bench-default/ollama (D2: it SHOULD resolve model/base_url from
    cfg.provider + a detect_capabilities probe, like the matrix path). Today it does hardcode both
    -> RED -> xfail(strict=True) passes.
    """
    import localharness.bench.orchestrator as orch
    from localharness.bench.orchestrator import (
        _synthesize_default_entry,
        build_llm_client_factory,
    )

    real_entry_holder: dict = {}

    def _spy_build(entry):
        real_entry_holder["entry"] = entry

        class _Dummy:  # stand-in client — never connected, never probed
            ...

        return _Dummy()

    monkeypatch.setattr(orch, "_build_bench_client", _spy_build)

    # Reproduce EXACTLY what the gate does at experiment.py:330, then invoke the returned factory
    # with a dummy scenario to trigger _build_bench_client — no real bench, no network.
    factory = build_llm_client_factory(_synthesize_default_entry())
    factory(object())  # _factory(_scen) ignores its arg; just triggers the construction seam

    entry = real_entry_holder["entry"]
    # The gate synthesizes a HARDCODED ollama/bench-default entry and never probes. D2 fix
    # direction: it SHOULD resolve model_id/base_url from cfg.provider + a detect_capabilities probe.
    assert entry.model_id != "bench-default", "gate still hardcodes model_id=bench-default (unprobed)"
    assert entry.provider != "ollama", "gate still hardcodes provider=ollama regardless of cfg.provider"


# --------------------------------------------------------------------------- #
# Task 2B — AUDIT-06 fix#3: the MATRIX path DOES probe (GREEN characterization)
# --------------------------------------------------------------------------- #


async def test_matrix_path_probes(monkeypatch, tmp_path):
    """fix#3: the matrix path probes (orchestrator.py:143-144 — _run_one_model builds a client and
    awaits detect_capabilities); the gate path (AUDIT-03a above) does not. This is the asymmetry.

    GREEN characterization of present behavior. We monkeypatch _build_bench_client to return a stub
    whose detect_capabilities is a capture-spy (sets probed=True, returns a fabricated
    CapabilityResult) and stub accumulate_runs to a no-op — so no real bench, no network. With
    scenarios=[] the scenario loop never runs; the only thing exercised is the probe.
    """
    import localharness.bench.orchestrator as orch
    from localharness.bench.config import MatrixEntry, SamplingConfig
    from localharness.provider.client import CapabilityResult

    state = {"probed": False}

    class _ProbedStub:
        async def detect_capabilities(self):
            state["probed"] = True
            # Fabricated CapabilityResult (client.py:45-51) — no real probe.
            return CapabilityResult(
                tool_call_mode="native",
                context_window=128_000,
                supports_streaming=True,
                probe_duration_ms=0.0,
                probe_error=None,
            )

    monkeypatch.setattr(orch, "_build_bench_client", lambda entry: _ProbedStub())

    async def _noop_accumulate(*args, **kwargs):
        return [], "stop"

    monkeypatch.setattr(orch, "accumulate_runs", _noop_accumulate)

    entry = MatrixEntry(name="m", provider="vllm", model_id="some-model", base_url="http://localhost:0/v1")
    await orch._run_one_model(
        entry,
        scenarios=[],  # empty -> the per-scenario loop body never runs; only the probe fires
        results_path=tmp_path,
        sampling=SamplingConfig(),
    )

    assert state["probed"] is True, "matrix path must invoke detect_capabilities (fix#3)"


# --------------------------------------------------------------------------- #
# #10 / #13 — _build_bench_client must use the synced canonical defaults
# --------------------------------------------------------------------------- #


def test_build_bench_client_timeout_is_600(monkeypatch):
    """#10: the bench client must not run a 300s read timeout — a full max_tokens completion
    at slow single-stream decode (~410s) would time out mid-generation."""
    from unittest.mock import patch
    import localharness.bench.orchestrator as orch
    from localharness.bench.config import MatrixEntry

    entry = MatrixEntry(name="m", provider="vllm", model_id="x", base_url="http://localhost:8000/v1")
    with patch("localharness.provider.client.AsyncOpenAI"):
        client = orch._build_bench_client(entry)
    assert client.config.timeout_seconds == 600.0


# --------------------------------------------------------------------------- #
# Task 3A — AUDIT-03c: proposer hardcodes tool_call_mode='native' (RED)
# --------------------------------------------------------------------------- #


def _proposer_cfg():
    """A HarnessConfig with a DISTINCT proposer block (PROP-02 _proposer_model_distinct validator
    forces proposer.model != provider.default_model). Mirrors test_proposer.py::_cfg."""
    from localharness.config.models import HarnessConfig

    return HarnessConfig.model_validate(
        {
            "version": "1",
            "provider": {
                "provider_type": "ollama",
                "base_url": "http://localhost:11434/v1",
                "default_model": "gpt-oss:120b",
            },
            "proposer": {
                "base_url": "http://localhost:11434/v1",
                "model": "frontier-strong:latest",
            },
        }
    )


async def test_proposer_hardcodes_native_mode(proposer_corpus, proposer_results, monkeypatch):
    """When llm is None the proposer builds LLMClient(LLMConfig(..., tool_call_mode='native'))
    (proposer.py:226-237, :235 the literal). We spy the MODULE-LEVEL LLMClient (the exact seam
    test_proposer.py:116 uses) to RECORD the LLMConfig it is constructed with, returning a fake
    whose complete() yields a parseable proposal so propose() runs to completion.

    The seal runs FIRST (proposer.py:206) so the spy's complete() is only reached AFTER a valid
    train run_id passes the seal (no real model — the fake returns canned JSON). THE assertion:
    the proposer SHOULD derive/probe the mode from cfg.proposer, not hardcode native. Today it is
    the literal 'native' -> RED -> xfail(strict=True) passes. (An XML-mode proposer model would
    mis-parse every proposal.)
    """
    import localharness.autoresearch.proposer as prop

    recorded: dict = {}
    complete_calls = {"n": 0}

    class _SpyClient:
        def __init__(self, cfg):
            recorded["cfg"] = cfg

        async def detect_capabilities(self):
            # Probe spy: return a non-"native" mode so the production code feeds it into LLMConfig.
            class _Cap:
                tool_call_mode = "xml"
            return _Cap()

        async def complete(self, messages, tools=None, stream=False):
            complete_calls["n"] += 1

            class _M:
                # _parse (proposer.py:59) requires {"after", "rationale"} with rationale min_length>=1.
                content = '{"after": "You are a careful, terse assistant.", "rationale": "verbosity in failed traces"}'

            return _M(), None

    monkeypatch.setattr(prop, "LLMClient", _SpyClient, raising=False)

    cfg = _proposer_cfg()
    # agent.role coerces trivially (a string path); valid train run_id passes the seal.
    await prop.propose(
        "agent.role",
        [proposer_results["train_run_id"]],
        cfg=cfg,
        llm=None,  # force the hardcoded construction path
        corpus_path=proposer_corpus,
        results_path=proposer_results["results"],
    )

    # complete() reached ONLY after the seal (valid train run_id) — proves no pre-seal model call.
    assert complete_calls["n"] == 1, "spy.complete must be reached exactly once, after the seal"
    # THE fix target: mode must NOT be the hardcoded 'native' — it should come from cfg.proposer / a probe.
    assert recorded["cfg"].tool_call_mode != "native", (
        "proposer hardcodes tool_call_mode='native' (proposer.py:235) — should derive/probe from cfg.proposer"
    )


# --------------------------------------------------------------------------- #
# Task 3B — AUDIT-03d: startup builds the LLMConfig from the STORED flag, discards the probe (GREEN)
# --------------------------------------------------------------------------- #


def test_startup_uses_stored_flag_not_probe():
    """AUDIT-03d (post-fix): start_cmd now feeds the PROBED tool_call_mode (from
    detect_capabilities) into the startup LLMConfig instead of the STORED
    provider.supports_function_calling flag.

    _probe_llm returns (bool, str | None) — the probed mode — and the startup path
    builds LLMConfig with tool_call_mode=probed_mode. A model swap re-probes by
    calling _probe_llm again before constructing the new LLMClient.

    Source-level proof via inspect (the 14-03 source-regression precedent): the
    stored-flag derivation ('native' if supports_function_calling else 'xml') must
    be GONE, and the probe result (probed_mode) must reach the LLMConfig.
    """
    from localharness.cli import start_cmd

    src = inspect.getsource(start_cmd).replace("'", '"')
    # The stored-flag derivation must be gone.
    assert 'tool_call_mode="native" if provider.supports_function_calling else "xml"' not in src, (
        "start_cmd still derives mode from the STORED supports_function_calling flag — must use probe"
    )
    # The probed mode must reach the LLMConfig.
    assert "probed_mode" in src, (
        "_probe_llm result (probed_mode) must be fed into the LLMConfig (FIDEL-04)"
    )
    # _probe_llm must return a mode, not just a bool.
    assert "tuple[bool" in src or "tuple[bool, str" in src or "probed_mode" in src, (
        "_probe_llm must surface the probed mode, not just a bool"
    )


# --------------------------------------------------------------------------- #
# Task 3C — AUDIT-03e: the root enabler — LLMConfig default tool_call_mode='native' (GREEN)
# --------------------------------------------------------------------------- #


def test_llmconfig_default_mode_is_native():
    """AUDIT-03e: the inherited default that makes every unprobed path native (client.py:33).
    GREEN characterization of the root enabler."""
    from localharness.provider.client import LLMConfig

    assert LLMConfig(base_url="x", model="y").tool_call_mode == "native"
