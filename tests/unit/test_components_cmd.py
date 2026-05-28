"""Phase 14 Wave 0 scaffolding for `localharness components` CLI.

Covers requirements REG-01, REG-02, REG-03, REG-04 (CLI surface).
Every test is xfail-marked until Phase 14-04 lands the actual command.
Tests use the `components_home` fixture from tests/conftest.py for
LOCALHARNESS_HOME isolation.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_list_includes_all_six_surfaces(components_home):
    """REG-01 / REG-04: `components list` must surface all 6 mutable categories."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    result = runner.invoke(app, ["components", "list"])
    assert result.exit_code == 0
    # Implementer must assert presence of: agent.system_prompt / tools.*.description /
    # org.context.compaction_threshold_pct / agent.stuck_detector /
    # agent.recovery_injection / hooks.*.config
    raise NotImplementedError("Stub for 14-04")


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_list_json_structure(components_home):
    """REG-01: `components list --json` rows have keys {path, type, current_value, layer}."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    result = runner.invoke(app, ["components", "list", "--json"])
    assert result.exit_code == 0
    raise NotImplementedError("Stub for 14-04")


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_get_resolves_cascade(components_home):
    """REG-02: setting an overlay value makes `get` return layer=user."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    result = runner.invoke(app, ["components", "get", "org.context.compaction_threshold_pct"])
    assert result.exit_code == 0
    raise NotImplementedError("Stub for 14-04")


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_get_unknown_path_exits_2(components_home):
    """REG-02: unknown dot-path exits non-zero with helpful message."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    result = runner.invoke(app, ["components", "get", "nonsense.path"])
    assert result.exit_code == 2
    raise NotImplementedError("Stub for 14-04")


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_set_round_trip(components_home):
    """REG-03: set X then get X returns X."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    set_result = runner.invoke(app, ["components", "set", "org.context.compaction_threshold_pct", "0.85"])
    assert set_result.exit_code == 0
    get_result = runner.invoke(app, ["components", "get", "org.context.compaction_threshold_pct"])
    assert get_result.exit_code == 0
    raise NotImplementedError("Stub for 14-04")


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_set_emits_component_mutated(components_home):
    """REG-03: `set` writes a ComponentMutated line to audit.jsonl."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    result = runner.invoke(app, ["components", "set", "org.context.compaction_threshold_pct", "0.85"])
    assert result.exit_code == 0
    audit_path = components_home / "audit.jsonl"
    assert audit_path.exists()
    raise NotImplementedError("Stub for 14-04")


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_set_validation_failure_does_not_write(components_home):
    """REG-03: invalid value exits non-zero AND overlay file unchanged."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    overlay_path = components_home / "overrides.yaml"
    before_exists = overlay_path.exists()
    before_contents = overlay_path.read_text() if before_exists else None
    result = runner.invoke(app, ["components", "set", "org.context.compaction_threshold_pct", "not-a-number"])
    assert result.exit_code != 0
    after_exists = overlay_path.exists()
    after_contents = overlay_path.read_text() if after_exists else None
    assert before_exists == after_exists
    assert before_contents == after_contents
    raise NotImplementedError("Stub for 14-04")


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_set_refuses_multi_path(components_home):
    """REG-03: multi-path set syntax is rejected at the CLI."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    # Either comma-separated or repeated args must fail
    result = runner.invoke(app, ["components", "set", "a.b,c.d", "1"])
    assert result.exit_code != 0
    raise NotImplementedError("Stub for 14-04")


@pytest.mark.xfail(reason="Phase 14-04 components_cmd.py not yet implemented", strict=False)
def test_audit_log_append_only(components_home):
    """Invariant: 3 sequential `set` calls produce exactly 3 audit lines (append, never truncate)."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-04")
    runner = CliRunner()
    for value in ("0.70", "0.80", "0.90"):
        result = runner.invoke(app, ["components", "set", "org.context.compaction_threshold_pct", value])
        assert result.exit_code == 0
    audit_path = components_home / "audit.jsonl"
    assert audit_path.exists()
    lines = audit_path.read_text().splitlines()
    assert len(lines) == 3
    raise NotImplementedError("Stub for 14-04")
