"""Unit tests for all 12 Pydantic config models."""
import pytest
from pydantic import ValidationError


def test_agent_config_valid():
    from localharness.config.models import AgentConfig
    cfg = AgentConfig(name="test-agent", role="Test role")
    assert cfg.name == "test-agent"
    assert cfg.role == "Test role"


def test_agent_config_camelcase_raises():
    from localharness.config.models import AgentConfig
    with pytest.raises(ValidationError):
        AgentConfig(name="BadName", role="x")


def test_agent_config_empty_model_raises():
    from localharness.config.models import AgentConfig
    with pytest.raises(ValidationError):
        AgentConfig(name="test", role="x", model="")


def test_schedule_config_valid_cron():
    from localharness.config.models import ScheduleConfig
    cfg = ScheduleConfig(cron="30 5 * * 1-5")
    assert cfg.cron == "30 5 * * 1-5"


def test_schedule_config_six_fields_raises():
    from localharness.config.models import ScheduleConfig
    with pytest.raises(ValidationError):
        ScheduleConfig(cron="30 5 * * * *")


def test_schedule_config_invalid_timezone_raises():
    from localharness.config.models import ScheduleConfig
    with pytest.raises(ValidationError):
        ScheduleConfig(timezone="NotAPlace")


def test_mcp_server_config_stdio_missing_command_raises():
    from localharness.config.models import MCPServerConfig
    with pytest.raises(ValidationError):
        MCPServerConfig(name="x", transport="stdio")


def test_mcp_server_config_http_missing_url_raises():
    from localharness.config.models import MCPServerConfig
    with pytest.raises(ValidationError):
        MCPServerConfig(name="x", transport="streamable_http")


def test_mcp_server_config_stdio_with_command_valid():
    from localharness.config.models import MCPServerConfig
    cfg = MCPServerConfig(name="x", transport="stdio", command="node")
    assert cfg.command == "node"


def test_permission_config_deny_patterns_default_count():
    from localharness.config.models import PermissionConfig
    cfg = PermissionConfig()
    # issue #15 grew the list from 7 to 24: destructive service/process-op globs + the fixed
    # sudo pattern + the embedded rm -rf form.
    assert len(cfg.deny_patterns) == 24


def test_permission_config_invalid_pattern_raises():
    from localharness.config.models import PermissionConfig
    with pytest.raises(ValidationError):
        PermissionConfig(deny_patterns=["invalid pattern!"])


def test_budget_config_max_actions_zero_raises():
    from localharness.config.models import BudgetConfig
    with pytest.raises(ValidationError):
        BudgetConfig(max_actions=0)


def test_tool_config_inherit_string_normalizes():
    from localharness.config.models import ToolConfig
    cfg = ToolConfig(inherit="division")
    assert cfg.inherit == ["division"]


# test_agent_config_memory_defaults_filled removed in phase 38 (dead config, zero readers —
# v013 Risk #1): the fields now stay None, and MemoryStore derives its real paths from base_dir.


def test_harness_config_with_provider_validates():
    from localharness.config.models import HarnessConfig, ProviderConfig
    cfg = HarnessConfig(
        provider=ProviderConfig(
            provider_type="ollama",
            base_url="http://localhost:11434/v1",
            default_model="qwen2.5:72b",
        )
    )
    assert cfg.provider.provider_type == "ollama"


def test_all_models_extra_forbid():
    from localharness.config.models import AgentConfig
    with pytest.raises(ValidationError):
        AgentConfig(name="a", role="x", bogus=1)


def test_division_config_valid():
    from localharness.config.models import DivisionConfig
    cfg = DivisionConfig(name="financial")
    assert cfg.name == "financial"


def test_org_config_valid():
    from localharness.config.models import OrgConfig
    cfg = OrgConfig(name="default", default_model="qwen2.5:72b")
    assert cfg.name == "default"


def test_memory_config_defaults():
    from localharness.config.models import MemoryConfig
    cfg = MemoryConfig()
    assert cfg.max_notes_chars == 16_000
    assert cfg.inject_into_context is True


def test_trace_ambient_injection_default_on():
    """Owner reversal 2026-07-17: the every-turn ambient shelf IS an activation event and is
    traced (source='injection') by default. The kill-switch defaults ON; setting it False
    restores the pre-reversal behavior (no injection-trace rows, today's exact bytes)."""
    from localharness.config.models import MemoryConfig
    cfg = MemoryConfig()
    assert cfg.trace_ambient_injection is True


def test_trace_injection_weight_default_and_bounds():
    """Consumer-side discount for injection-source co-fire (retrieval-source is implicitly 1.0).
    Default 0.3 — a materialized-view weight in [0,1]; the raw trace log keeps full fidelity, the
    discount lives ONLY in derived computations (discovery's co-fire strength)."""
    import pydantic

    from localharness.config.models import MemoryConsolidationConfig

    cfg = MemoryConsolidationConfig()
    assert cfg.trace_injection_weight == 0.3
    # bounded [0.0, 1.0]: 0.0 = ignore injection co-fire entirely, 1.0 = no discount.
    for bad in (-0.1, 1.1):
        with pytest.raises(pydantic.ValidationError):
            MemoryConsolidationConfig(trace_injection_weight=bad)


def test_context_config_defaults():
    from localharness.config.models import ContextConfig
    from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS
    cfg = ContextConfig()
    # Single source of truth: the schema default now tracks the served reference window
    # (131_072). At runtime `start` derives the EFFECTIVE budget from the probed
    # max_model_len minus the output reservation; this config value is only an explicit
    # cap/override. The old 61_440 default silently capped a 131K-window agent at <half.
    assert cfg.max_context_tokens == DEFAULT_MAX_CONTEXT_TOKENS
    assert cfg.compaction_threshold_pct == 80.0


def test_context_config_resolve_budget_pin_outranks_scalar():
    """#137: ONE resolution of pin-vs-scalar, so `start` and `doctor` can never report
    different budgets for the same session. Exact name only — no globbing, no prefix match."""
    from localharness.config.models import ContextConfig
    cfg = ContextConfig(
        max_context_tokens=61_440, model_context_overrides={"qwen3.8-27b": 40_000}
    )
    assert cfg.resolve_budget("qwen3.8-27b") == (40_000, True)
    assert cfg.resolve_budget("qwen3.8-27b-awq") == (61_440, False)  # no prefix globbing
    assert cfg.resolve_budget("other-model") == (61_440, False)
    assert cfg.resolve_budget(None) == (61_440, False)
    # Empty map (the overwhelmingly common case) = the scalar, unchanged.
    assert ContextConfig(max_context_tokens=8_000).resolve_budget("m") == (8_000, False)


# --- Phase 14-02 Task 1: StuckDetectorConfig / RecoveryInjectionConfig / OrgConfig.hooks ---

def test_stuck_detector_config_defaults_mirror_loop_hardcode():
    from localharness.config.models import AgentConfig
    cfg = AgentConfig(name="t", role="t")
    assert cfg.stuck_detector.window_size == 5
    assert cfg.stuck_detector.recovery_threshold == 2
    assert cfg.stuck_detector.escalation_threshold == 3


def test_stuck_detector_override():
    from localharness.config.models import AgentConfig
    cfg = AgentConfig(name="t", role="t", stuck_detector={"window_size": 7})
    assert cfg.stuck_detector.window_size == 7


def test_max_subagent_depth_default_and_bounds():
    import pytest as _pytest
    from pydantic import ValidationError
    from localharness.config.models import AgentConfig
    assert AgentConfig(name="t", role="t").max_subagent_depth == 2  # nesting on by default
    assert AgentConfig(name="t", role="t", max_subagent_depth=1).max_subagent_depth == 1  # kill-switch
    for bad in (0, 5):
        with _pytest.raises(ValidationError):
            AgentConfig(name="t", role="t", max_subagent_depth=bad)


def test_recovery_injection_default_matches_loop_string():
    from localharness.config.models import AgentConfig
    cfg = AgentConfig(name="t", role="t")
    expected = (
        "You have attempted the same tool call multiple times with identical arguments "
        "and received the same result. That approach is not working. "
        "Consider a fundamentally different strategy: try different arguments, "
        "use a different tool, or conclude that the information is not available this way."
    )
    assert cfg.recovery_injection.message == expected


def test_recovery_injection_override():
    from localharness.config.models import AgentConfig
    cfg = AgentConfig(name="t", role="t", recovery_injection={"message": "custom"})
    assert cfg.recovery_injection.message == "custom"


def test_org_hooks_default_empty():
    from localharness.config.models import OrgConfig
    cfg = OrgConfig()
    assert cfg.hooks == {}


def test_org_hooks_accept_freeform_dict():
    from localharness.config.models import OrgConfig
    cfg = OrgConfig(hooks={"my_hook": {"enabled": True}})
    assert cfg.hooks["my_hook"]["enabled"] is True


def test_stuck_detector_zero_window_raises():
    from localharness.config.models import AgentConfig
    with pytest.raises(ValidationError):
        AgentConfig(name="t", role="t", stuck_detector={"window_size": 0})


def test_stuck_detector_extra_forbid():
    from localharness.config.models import AgentConfig
    with pytest.raises(ValidationError):
        AgentConfig(name="t", role="t", stuck_detector={"unknownField": 5})


# -----------------------------------------------------------------------------
# PROP-02 — ProposerConfig (Phase 16 Wave 0 RED stubs)
# -----------------------------------------------------------------------------

from localharness.config.models import ProposerConfig  # noqa: F401


def _harness_dict(**overrides) -> dict:
    """Minimal valid HarnessConfig dict; overrides merge at the top level."""
    data = {
        "version": "1",
        "provider": {
            "provider_type": "ollama",
            "base_url": "http://localhost:11434/v1",
            "default_model": "gpt-oss:120b",
        },
    }
    data.update(overrides)
    return data


def test_proposer_model_must_differ():
    """PROP-02: proposer.model == provider.default_model → ValidationError (distinct-model rule)."""
    from localharness.config.models import HarnessConfig

    bad = _harness_dict(
        proposer={
            "base_url": "http://localhost:11434/v1",
            "model": "gpt-oss:120b",  # same as provider.default_model
        }
    )
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(bad)


def test_proposer_config_optional():
    """PROP-02: a HarnessConfig with NO proposer block validates (proposer is opt-in)."""
    from localharness.config.models import HarnessConfig

    cfg = HarnessConfig.model_validate(_harness_dict())
    assert getattr(cfg, "proposer", None) is None


# -----------------------------------------------------------------------------
# Phase 34 COLL — PredictiveGateConfig + TriggerLexiconConfig under MemoryConfig
# Collect-only default-on (write_gate_enabled precedent); lexicon is registry-tunable
# data, not code (COLL-02); extra keys rejected; axes auto-enumerate with zero
# catalogue edits (the walk_model_fields list[str]-as-leaf precedent).
# -----------------------------------------------------------------------------


def test_predictive_gate_defaults():
    """COLL default-on with the owner-steered scoring weights."""
    from localharness.config.models import MemoryConfig
    cfg = MemoryConfig()
    assert cfg.predictive_gate.enabled is True
    assert cfg.predictive_gate.min_prior_n == 5
    assert cfg.predictive_gate.latency_weight == 0.5
    assert cfg.predictive_gate.size_weight == 0.25
    assert cfg.predictive_gate.reask_threshold == 0.8


def test_predictive_gate_lexicon_defaults():
    """COLL-02 recall-first trigger lists — the owner's own specimens land per family."""
    from localharness.config.models import MemoryConfig
    lex = MemoryConfig().predictive_gate.lexicon
    assert "nah" in lex.negation
    assert "i meant" in lex.correction_phrase
    assert "exactly" in lex.confirmation
    assert "hold on" in lex.interruption
    assert "frustrating" in lex.frustration


def test_predictive_gate_extra_forbid():
    """extra='forbid' — an unknown predictive_gate key is rejected, not silently kept."""
    from localharness.config.models import MemoryConfig
    with pytest.raises(ValidationError):
        MemoryConfig(predictive_gate={"nope": 1})


def test_predictive_gate_registry_discovery():
    """The new axes auto-enumerate via walk_model_fields (zero catalogue edits) — both a
    scalar leaf and a lexicon list[str] leaf appear on AgentConfig's walked paths."""
    from localharness.config.models import AgentConfig
    from localharness.registry.paths import walk_model_fields
    paths = {p for p, _ann in walk_model_fields(AgentConfig)}
    assert "memory.predictive_gate.enabled" in paths
    assert "memory.predictive_gate.lexicon.negation" in paths


def test_mining_novelty_fold_threshold_default_is_sweep_winner():
    """sweep-20260711d: NV-HI (0.70) was the sole HOLDS row (ARI 0.748 vs 0.542 baseline,
    recall 1.000, 32 vs 60 atoms); the old default 0.5 was the worst tested novelty value.
    See .planning/runs/sweep-20260711d/SYNTHESIS.md (untracked)."""
    from localharness.config.models import MemoryConsolidationConfig

    assert MemoryConsolidationConfig().mining_novelty_fold_threshold == 0.70


# --- #1: EndpointRef.base_url rejects a genuinely-malformed URL at load (clean config error) --- #


@pytest.mark.parametrize("url", [
    "http://localhost:11434/v1",
    "https://api.example.com/v1",
    "http://127.0.0.1:8000",        # #1 lenient: /v1 suffix NOT required (servers differ)
    "http://localhost:notaport/v1",  # #1 lenient: a bad port is a runtime skip, not a load error
])
def test_endpoint_ref_accepts_wellformed_urls(url):
    from localharness.config.models import EndpointRef
    assert EndpointRef(name="peer", base_url=url).base_url == url


@pytest.mark.parametrize("url", ["not-a-url", "ftp://x/y", "http://", "", "localhost:8000", "://nope"])
def test_endpoint_ref_rejects_malformed_base_url(url):
    """A base_url that is not a well-formed http(s)://host URL is rejected at load — a clean config
    error instead of a silent runtime skip on every /model."""
    from localharness.config.models import EndpointRef
    with pytest.raises(ValidationError):
        EndpointRef(name="peer", base_url=url)


# --- C2.1: EndpointRef gains gpu + lifecycle (Phase C2 cross-framework heavy-swap launch spec) --- #


def test_endpoint_ref_gpu_defaults_false_lifecycle_none():
    """0.10.0 attach-only default preserved: a peer is CPU-light (coexists) and not launchable
    unless explicitly declared. The GPU-lock only acts on gpu=True peers; a lifecycle block only
    exists for peers the harness can bring up itself."""
    from localharness.config.models import EndpointRef
    ep = EndpointRef(name="peer", base_url="http://localhost:11434/v1")
    assert ep.gpu is False
    assert ep.lifecycle is None


def test_endpoint_ref_accepts_gpu_lifecycle_launch_spec():
    """A cold GPU-heavy peer carries a full ManagedServerConfig launch spec so the harness can
    start it on the freed accelerator (Phase C2). Here: a llama.cpp peer."""
    from localharness.config.models import EndpointRef, ManagedServerConfig
    ep = EndpointRef(
        name="llamacpp-local",
        base_url="http://127.0.0.1:8080/v1",
        provider_type="llamacpp",
        gpu=True,
        lifecycle=ManagedServerConfig(
            runtime="llamacpp",
            launch="binary",
            binary="/home/x/llama.cpp/build/bin/llama-server",
            model="/home/x/models/q.gguf",
            port=8080,
            extra_args=["-c", "32768", "--parallel", "1", "-ngl", "99", "--jinja", "-a", "qwen3.6-35b-a3b"],
            gpu=True,
        ),
    )
    assert ep.gpu is True
    assert ep.lifecycle is not None
    assert ep.lifecycle.runtime == "llamacpp"
    assert ep.lifecycle.binary.endswith("llama-server")


@pytest.mark.parametrize("ep_gpu, life_gpu", [(False, False), (False, True), (True, False)])
def test_endpoint_ref_rejects_non_heavy_launchable_peer(ep_gpu, life_gpu):
    """A LAUNCHABLE peer (lifecycle set) must be gpu=True on BOTH the endpoint and its lifecycle spec.
    Every non-heavy combination is rejected: the harness tracks one launched server at a time via a
    single per-config pidfile (and the GPU-lock keeps one launched heavy up), so a coexisting CPU
    launchable peer — which the shared pidfile could not track without orphaning/mis-stopping a
    process — is not yet supported. Attaching to a running CPU peer stays fine (lifecycle=None)."""
    from localharness.config.models import EndpointRef, ManagedServerConfig
    with pytest.raises(ValidationError):
        EndpointRef(
            name="bad", base_url="http://127.0.0.1:8080/v1", provider_type="llamacpp", gpu=ep_gpu,
            lifecycle=ManagedServerConfig(
                runtime="llamacpp", launch="binary", binary="/x/llama-server",
                model="/x/q.gguf", gpu=life_gpu,
            ),
        )


# --- D1: ManagedServerConfig runtime="ollama" (Phase D DaemonStrategy) --- #


def test_managed_server_ollama_runtime_needs_no_binary():
    """runtime='ollama' — the harness spawns `ollama serve` (a PATH command, no checkpoint/image),
    so the launch-target validator requires neither binary nor docker_image; `model` is the ollama
    tag, `binary` defaults to None (resolved to PATH `ollama` by the command builder)."""
    from localharness.config.models import ManagedServerConfig
    srv = ManagedServerConfig(runtime="ollama", model="qwen2.5:7b", port=11434, gpu=False)
    assert srv.runtime == "ollama"
    assert srv.binary is None and srv.docker_image is None
    assert srv.model == "qwen2.5:7b" and srv.port == 11434


def test_managed_server_lmstudio_runtime_needs_no_binary():
    """runtime='lmstudio' (Phase D5 LmsStrategy) — the harness drives LM Studio's `lms` CLI (a
    command, not a checkpoint/image), so the launch-target validator requires neither binary nor
    docker_image; `model` is the LM Studio model key. `binary` is normally the `lms` path but is not
    REQUIRED (LmsStrategy falls back to a bare `lms` on PATH)."""
    from localharness.config.models import ManagedServerConfig
    srv = ManagedServerConfig(runtime="lmstudio", model="qwen2.5-0.5b-instruct", port=1234, gpu=False)
    assert srv.runtime == "lmstudio"
    assert srv.binary is None and srv.docker_image is None
    assert srv.model == "qwen2.5-0.5b-instruct" and srv.port == 1234


def test_managed_server_lmstudio_rejects_gpu_in_extra_args():
    """runtime=lmstudio: `--gpu` is derived from `gpu` (the GPU-lock signal); a manual --gpu in
    extra_args (which CLI last-wins would silently honor) is rejected at load to prevent GPU-lock
    desync → a two-heavy freeze on a later swap."""
    from localharness.config.models import ManagedServerConfig
    with pytest.raises(ValidationError):
        ManagedServerConfig(runtime="lmstudio", model="m", extra_args=["--gpu", "max"])
    with pytest.raises(ValidationError):
        ManagedServerConfig(runtime="lmstudio", model="m", extra_args=["--gpu=off"])
    ok = ManagedServerConfig(runtime="lmstudio", model="m", extra_args=["-c", "4096"])
    assert ok.extra_args == ["-c", "4096"]


@pytest.mark.parametrize("kw", [
    dict(runtime="llamacpp", model="/x/m.gguf"),               # llamacpp needs binary
    dict(runtime="vllm", launch="binary", model="m"),          # binary launch needs binary
    dict(runtime="vllm", launch="docker", model="m"),          # docker launch needs docker_image
])
def test_managed_server_non_ollama_launch_validators_unchanged(kw):
    """Regression: the ollama exemption is scoped — vLLM/llama.cpp still require their launch target."""
    from localharness.config.models import ManagedServerConfig
    with pytest.raises(ValidationError):
        ManagedServerConfig(**kw)
