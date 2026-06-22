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
    assert len(cfg.deny_patterns) == 7


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


def test_agent_config_memory_defaults_filled():
    from localharness.config.models import AgentConfig
    cfg = AgentConfig(name="my-agent", role="Test role")
    assert cfg.memory.sqlite_path is not None
    assert "my-agent" in cfg.memory.sqlite_path


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
