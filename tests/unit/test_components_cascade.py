"""Phase 14 Wave 0 scaffolding for the cascade resolution invariant.

Covers INV-cascade (defaults < project < user [< experiment]).
Every test is xfail-marked until Phase 14-03 wires the cascade resolver.
"""
from __future__ import annotations

import pytest
import yaml


@pytest.mark.xfail(reason="Phase 14-03 cascade resolution not yet implemented", strict=True)
def test_user_overlay_wins_over_project(components_home):
    """User overrides.yaml wins over project config.yaml on the same path."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-03")
    # Project layer (already seeded by fixture) — override the threshold
    project_yaml = {
        "version": "1",
        "provider": {"provider_type": "ollama", "base_url": "http://x", "default_model": "m"},
        "org": {"context": {"compaction_threshold_pct": 0.70}},
    }
    (components_home / "config.yaml").write_text(yaml.safe_dump(project_yaml))
    # User layer
    user_overlay = {"org": {"context": {"compaction_threshold_pct": 0.85}}}
    (components_home / "overrides.yaml").write_text(yaml.safe_dump(user_overlay))

    runner = CliRunner()
    result = runner.invoke(app, ["components", "get", "org.context.compaction_threshold_pct"])
    assert result.exit_code == 0
    # Implementer asserts "0.85" in result.output and "user" appears as layer
    raise NotImplementedError("Stub for 14-03")


@pytest.mark.xfail(reason="Phase 14-03 cascade resolution not yet implemented", strict=True)
def test_project_wins_over_default(components_home):
    """Without a user overlay, project YAML value wins over Pydantic default."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-03")
    project_yaml = {
        "version": "1",
        "provider": {"provider_type": "ollama", "base_url": "http://x", "default_model": "m"},
        "org": {"context": {"compaction_threshold_pct": 0.65}},
    }
    (components_home / "config.yaml").write_text(yaml.safe_dump(project_yaml))

    runner = CliRunner()
    result = runner.invoke(app, ["components", "get", "org.context.compaction_threshold_pct"])
    assert result.exit_code == 0
    raise NotImplementedError("Stub for 14-03")


@pytest.mark.xfail(reason="Phase 14-03 cascade resolution not yet implemented", strict=True)
def test_unknown_overlay_key_rejected_at_set_time(components_home):
    """extra=forbid: `set` with a typo'd dot-path exits non-zero AND does not write overlay."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-03")
    overlay_path = components_home / "overrides.yaml"
    before_exists = overlay_path.exists()
    before_contents = overlay_path.read_text() if before_exists else None

    runner = CliRunner()
    result = runner.invoke(app, ["components", "set", "agent.stuck_detector.windowSize", "9"])
    assert result.exit_code != 0
    after_exists = overlay_path.exists()
    after_contents = overlay_path.read_text() if after_exists else None
    assert before_exists == after_exists
    assert before_contents == after_contents
    raise NotImplementedError("Stub for 14-03")


@pytest.mark.xfail(reason="Phase 17 experiment overlay layer not yet implemented", strict=True)
def test_experiment_overlay_wins_over_user(components_home):
    """Phase 17 forward-compat: experiment overlay wins over user overlay."""
    try:
        from typer.testing import CliRunner
        from localharness.cli.app import app
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 17 experiment runner")
    user_overlay = {"org": {"context": {"compaction_threshold_pct": 0.80}}}
    (components_home / "overrides.yaml").write_text(yaml.safe_dump(user_overlay))
    experiment_dir = components_home / "experiments" / "exp-001"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_overlay = {"org": {"context": {"compaction_threshold_pct": 0.90}}}
    (experiment_dir / "overrides.yaml").write_text(yaml.safe_dump(experiment_overlay))

    runner = CliRunner()
    result = runner.invoke(app, ["components", "get", "org.context.compaction_threshold_pct"])
    assert result.exit_code == 0
    raise NotImplementedError("Stub for Phase 17")
