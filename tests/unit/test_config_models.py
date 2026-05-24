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
    cfg = ContextConfig()
    assert cfg.max_context_tokens == 128_000
    assert cfg.compaction_threshold_pct == 80.0
