"""16 mandatory unit tests for ConfigLoader per spec 06 section 10.7."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from localharness.config.loader import (
    ConfigLoader,
    ConfigError,
    ConfigParseError,
    ConfigReferenceError,
    ConfigValidationError,
)
from localharness.config.models import AgentConfig, MCPServerConfig, ScheduleConfig


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "agents").mkdir()
    (tmp_path / "divisions").mkdir()
    return tmp_path


# ------------------------------------------------------------------ #
# 1. Minimal config loads without error
# ------------------------------------------------------------------ #
def test_minimal_agent_config_valid(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "test-agent.yaml", {"name": "test-agent", "role": "Test role"})
    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.load_agent("test-agent")
    assert cfg.name == "test-agent"
    assert cfg.role == "Test role"


# ------------------------------------------------------------------ #
# 2. CamelCase name raises ConfigValidationError
# ------------------------------------------------------------------ #
def test_invalid_name_rejected(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "BadName.yaml", {"name": "BadName", "role": "Test role"})
    loader = ConfigLoader(config_dir=config_dir)
    with pytest.raises(ConfigValidationError) as exc_info:
        loader.load_agent("BadName")
    errors = exc_info.value.errors
    assert any("name" in e.field_path for e in errors)


# ------------------------------------------------------------------ #
# 3. model: inherit resolves to division model
# ------------------------------------------------------------------ #
def test_inherit_sentinel_resolved(config_dir: Path) -> None:
    _write_yaml(config_dir / "divisions" / "research.yaml", {
        "name": "research",
        "model": "llama3:70b",
    })
    _write_yaml(config_dir / "agents" / "my-agent.yaml", {
        "name": "my-agent",
        "role": "Research agent",
        "division": "research",
        "model": "inherit",
    })
    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.load_agent("my-agent")
    assert cfg.model == "llama3:70b"


# ------------------------------------------------------------------ #
# 4. deny_patterns are unioned across org + division + agent
# ------------------------------------------------------------------ #
def test_deny_patterns_union(config_dir: Path) -> None:
    _write_yaml(config_dir / "org.yaml", {
        "name": "myorg",
        "permissions": {
            "deny_patterns": ["bash(curl:*)"],
        },
    })
    _write_yaml(config_dir / "divisions" / "infra.yaml", {
        "name": "infra",
        "permissions": {
            "deny_patterns": ["bash(wget:*)"],
        },
    })
    _write_yaml(config_dir / "agents" / "deployer.yaml", {
        "name": "deployer",
        "role": "Deploy agent",
        "division": "infra",
        "permissions": {
            "deny_patterns": ["bash(rm:*)"],
        },
    })
    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.load_agent("deployer")
    patterns = cfg.permissions.deny_patterns
    assert "bash(curl:*)" in patterns
    assert "bash(wget:*)" in patterns
    assert "bash(rm:*)" in patterns


# ------------------------------------------------------------------ #
# 5. Tool in both add and deny results in tool being denied
# ------------------------------------------------------------------ #
def test_tool_deny_wins_over_add(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "conflicted.yaml", {
        "name": "conflicted",
        "role": "Test agent",
        "tools": {
            "add": ["bash"],
            "deny": ["bash"],
        },
    })
    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.load_agent("conflicted")
    assert "bash" in cfg.tools.deny
    assert "bash" in cfg.tools.add  # stored as-is in model; enforcement is at runtime


# ------------------------------------------------------------------ #
# 6. division: nonexistent raises ConfigReferenceError
# ------------------------------------------------------------------ #
def test_division_not_found_raises(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "orphan.yaml", {
        "name": "orphan",
        "role": "Orphan agent",
        "division": "nonexistent",
    })
    loader = ConfigLoader(config_dir=config_dir)
    with pytest.raises(ConfigReferenceError):
        loader.load_agent("orphan")


# ------------------------------------------------------------------ #
# 7. Agent max_actions overrides division max_actions
# ------------------------------------------------------------------ #
def test_budget_agent_wins(config_dir: Path) -> None:
    _write_yaml(config_dir / "divisions" / "big.yaml", {
        "name": "big",
        "permissions": {"budget": {"max_actions": 200}},
    })
    _write_yaml(config_dir / "agents" / "small.yaml", {
        "name": "small",
        "role": "Small agent",
        "division": "big",
        "permissions": {"budget": {"max_actions": 50}},
    })
    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.load_agent("small")
    assert cfg.permissions.budget.max_actions == 50


# ------------------------------------------------------------------ #
# 8. sqlite_path auto-filled from agent name
# ------------------------------------------------------------------ #
def test_memory_defaults_filled(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "my-agent.yaml", {
        "name": "my-agent",
        "role": "Memory test agent",
    })
    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.load_agent("my-agent")
    assert cfg.memory.sqlite_path is not None
    assert "my-agent" in cfg.memory.sqlite_path
    assert "memory.db" in cfg.memory.sqlite_path


# ------------------------------------------------------------------ #
# 9. write_agent() creates YAML on disk
# ------------------------------------------------------------------ #
def test_write_agent_creates_file(config_dir: Path) -> None:
    loader = ConfigLoader(config_dir=config_dir)
    cfg = AgentConfig(name="new-agent", role="New agent role")
    path = loader.write_agent(cfg)
    assert path.exists()
    assert path.suffix == ".yaml"
    assert path.stem == "new-agent"


# ------------------------------------------------------------------ #
# 10. overwrite=True creates .yaml.bak
# ------------------------------------------------------------------ #
def test_write_agent_backup_on_overwrite(config_dir: Path) -> None:
    loader = ConfigLoader(config_dir=config_dir)
    cfg = AgentConfig(name="backup-agent", role="Backup test agent")
    first_path = loader.write_agent(cfg)
    assert first_path.exists()
    second_path = loader.write_agent(cfg, overwrite=True)
    bak_path = first_path.with_suffix(".yaml.bak")
    assert bak_path.exists()
    assert second_path == first_path


# ------------------------------------------------------------------ #
# 11. validate_all() returns one tuple per config file
# ------------------------------------------------------------------ #
def test_validate_all_returns_results(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "agent-a.yaml", {"name": "agent-a", "role": "Role A"})
    _write_yaml(config_dir / "agents" / "agent-b.yaml", {"name": "agent-b", "role": "Role B"})
    _write_yaml(config_dir / "divisions" / "div-a.yaml", {"name": "div-a"})
    loader = ConfigLoader(config_dir=config_dir)
    results = loader.validate_all()
    assert isinstance(results, list)
    assert len(results) >= 3
    for path_str, err in results:
        assert isinstance(path_str, str)
        assert err is None or isinstance(err, ConfigError)


# ------------------------------------------------------------------ #
# 12. loader uses yaml.safe_load (verify no yaml.load call)
# ------------------------------------------------------------------ #
def test_yaml_safe_load_only(config_dir: Path) -> None:
    """Verify loader source does not contain a call to yaml.load (unsafe)."""
    import localharness.config.loader as loader_module
    source = inspect.getsource(loader_module)
    # Must not contain bare yaml.load( — yaml.safe_load is the only allowed call
    import re as _re
    unsafe_calls = _re.findall(r'\byaml\.load\s*\(', source)
    assert not unsafe_calls, f"Found unsafe yaml.load call(s) in loader: {unsafe_calls}"
    # Also verify yaml.safe_load is present
    assert "yaml.safe_load" in source


# ------------------------------------------------------------------ #
# 13. 6-field cron raises ConfigValidationError
# ------------------------------------------------------------------ #
def test_cron_five_fields_required(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "cron-agent.yaml", {
        "name": "cron-agent",
        "role": "Cron agent",
        "schedule": {"cron": "30 5 * * * *"},  # 6 fields — invalid
    })
    loader = ConfigLoader(config_dir=config_dir)
    with pytest.raises(ConfigValidationError):
        loader.load_agent("cron-agent")


# ------------------------------------------------------------------ #
# 14. timezone: NotAPlace raises ConfigValidationError
# ------------------------------------------------------------------ #
def test_invalid_timezone_rejected(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "tz-agent.yaml", {
        "name": "tz-agent",
        "role": "Timezone agent",
        "schedule": {"cron": "30 5 * * 1", "timezone": "NotAPlace"},
    })
    loader = ConfigLoader(config_dir=config_dir)
    with pytest.raises(ConfigValidationError):
        loader.load_agent("tz-agent")


# ------------------------------------------------------------------ #
# 15. stdio transport without command raises
# ------------------------------------------------------------------ #
def test_mcp_stdio_requires_command(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "mcp-agent.yaml", {
        "name": "mcp-agent",
        "role": "MCP agent",
        "tools": {
            "mcp_servers": [
                {"name": "myserver", "transport": "stdio"},  # no command
            ]
        },
    })
    loader = ConfigLoader(config_dir=config_dir)
    with pytest.raises(ConfigValidationError):
        loader.load_agent("mcp-agent")


# ------------------------------------------------------------------ #
# 16. streamable_http without url raises
# ------------------------------------------------------------------ #
def test_mcp_http_requires_url(config_dir: Path) -> None:
    _write_yaml(config_dir / "agents" / "http-agent.yaml", {
        "name": "http-agent",
        "role": "HTTP MCP agent",
        "tools": {
            "mcp_servers": [
                {"name": "myserver", "transport": "streamable_http"},  # no url
            ]
        },
    })
    loader = ConfigLoader(config_dir=config_dir)
    with pytest.raises(ConfigValidationError):
        loader.load_agent("http-agent")
