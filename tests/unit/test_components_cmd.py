"""REG-01..04 CLI surface tests. Phase 14-04.

Uses the `components_home` fixture from tests/conftest.py to isolate
LOCALHARNESS_HOME so config.yaml, overrides.yaml, and audit.jsonl all
live in a tmp dir for each test.
"""
from __future__ import annotations

import json
import re as _re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.registry import SURFACE_FAMILIES

runner = CliRunner()


# ------------------------------------------------------------------ #
# Module-shape contract (Task 1 RED — now GREEN)
# ------------------------------------------------------------------ #


def test_components_subapp_registered_in_main_app():
    """Task 2 / 14-04: cli/app.py must register components_app under name='components'."""
    top_help = runner.invoke(app, ["--help"])
    assert top_help.exit_code == 0, top_help.output
    assert "components" in top_help.output

    sub_help = runner.invoke(app, ["components", "--help"])
    assert sub_help.exit_code == 0, sub_help.output
    for cmd in ("list", "get", "set"):
        assert cmd in sub_help.output


def test_components_cmd_module_exports_typer_subapp():
    """Task 1 / 14-04: components_cmd.py exports a `components_app` Typer subapp."""
    import typer

    from localharness.cli.components_cmd import (  # noqa: F401
        components_app,
        components_get,
        components_list,
        components_set,
    )

    assert isinstance(components_app, typer.Typer)
    assert components_app.info.name == "components"
    registered_names = {cmd.name for cmd in components_app.registered_commands}
    assert {"list", "get", "set"}.issubset(registered_names)


# ------------------------------------------------------------------ #
# Test helpers
# ------------------------------------------------------------------ #


def _write_project_yaml(home: Path, **overrides) -> None:
    """Write a minimal HarnessConfig YAML to home/config.yaml.

    overrides keys are merged at the top level. Pass org={...} to set
    org.audit_log_path or context.
    """
    data = {
        "version": "1",
        "provider": {
            "provider_type": "ollama",
            "base_url": "http://localhost:11434/v1",
            "default_model": "test-model",
        },
    }
    data.update(overrides)
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _read_audit(home: Path) -> list[dict]:
    """Read audit.jsonl lines."""
    audit = home / "audit.jsonl"
    if not audit.exists():
        return []
    return [
        json.loads(line)
        for line in audit.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_overlay_bytes(home: Path) -> bytes:
    overlay = home / "overrides.yaml"
    return overlay.read_bytes() if overlay.exists() else b""


# ------------------------------------------------------------------ #
# REG-01: list
# ------------------------------------------------------------------ #


def test_list_includes_all_six_surfaces(components_home):
    """REG-01 / REG-04: `components list` must surface all 6 mutable categories."""
    _write_project_yaml(
        components_home,
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    result = runner.invoke(app, ["components", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    paths = [row["path"] for row in payload]
    for family_name, patterns in SURFACE_FAMILIES.items():
        matched = any(_re.search(p, path) for path in paths for p in patterns)
        assert matched, (
            f"No path matched surface family {family_name!r}; "
            f"patterns={patterns}; sample paths={paths[:10]}..."
        )


def test_list_json_structure(components_home):
    """REG-01: `components list --json` rows have keys {path, type, current_value, layer}."""
    _write_project_yaml(
        components_home,
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    result = runner.invoke(app, ["components", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, list) and len(payload) > 0
    required_keys = {"path", "type", "current_value", "layer"}
    for row in payload:
        assert required_keys.issubset(row.keys()), f"Missing keys in {row}"


# ------------------------------------------------------------------ #
# REG-02: get
# ------------------------------------------------------------------ #


def test_get_resolves_cascade(components_home):
    """REG-02: user overlay value beats project value; `get` reports layer=user."""
    _write_project_yaml(
        components_home,
        org={
            "context": {"compaction_threshold_pct": 70.0},
            "audit_log_path": str(components_home / "audit.jsonl"),
        },
    )
    # User overlay bumps to 85.0
    (components_home / "overrides.yaml").write_text(
        yaml.safe_dump({"org": {"context": {"compaction_threshold_pct": 85.0}}}),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["components", "get", "org.context.compaction_threshold_pct", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["value"] == 85.0
    assert payload["layer"] == "user"


def test_get_unknown_path_exits_2(components_home):
    """REG-02: unknown dot-path exits non-zero with helpful message."""
    _write_project_yaml(
        components_home,
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    result = runner.invoke(app, ["components", "get", "nonsense.path"])
    assert result.exit_code == 2, result.output


# ------------------------------------------------------------------ #
# REG-03: set
# ------------------------------------------------------------------ #


def test_set_round_trip(components_home):
    """REG-03: set X then get X returns X with layer=user."""
    _write_project_yaml(
        components_home,
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    set_result = runner.invoke(
        app,
        ["components", "set", "org.context.compaction_threshold_pct", "85.0"],
    )
    assert set_result.exit_code == 0, set_result.output
    get_result = runner.invoke(
        app,
        ["components", "get", "org.context.compaction_threshold_pct", "--json"],
    )
    assert get_result.exit_code == 0, get_result.output
    payload = json.loads(get_result.stdout)
    assert payload["value"] == 85.0
    assert payload["layer"] == "user"


def test_set_emits_component_mutated(components_home):
    """REG-03: `set` writes a ComponentMutated line to audit.jsonl."""
    _write_project_yaml(
        components_home,
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    result = runner.invoke(
        app,
        ["components", "set", "org.context.compaction_threshold_pct", "85.0"],
    )
    assert result.exit_code == 0, result.output
    events = _read_audit(components_home)
    mutations = [e for e in events if e.get("event_type") == "ComponentMutated"]
    assert len(mutations) == 1
    assert mutations[0]["path"] == "org.context.compaction_threshold_pct"
    assert mutations[0]["after_value"] == 85.0
    assert mutations[0]["layer"] == "user"
    assert mutations[0]["actor"] == "cli"


def test_set_validation_failure_does_not_write(components_home):
    """REG-03: invalid value exits non-zero AND overlay file unchanged.

    compaction_threshold_pct has le=99.0; 150.0 must fail Pydantic validation.
    """
    _write_project_yaml(
        components_home,
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    before_bytes = _read_overlay_bytes(components_home)
    result = runner.invoke(
        app,
        ["components", "set", "org.context.compaction_threshold_pct", "150.0"],
    )
    assert result.exit_code != 0, result.output
    after_bytes = _read_overlay_bytes(components_home)
    assert before_bytes == after_bytes, "Overlay file changed despite validation failure"


def test_set_refuses_multi_path(components_home):
    """REG-03: multi-path set syntax is rejected at the CLI."""
    _write_project_yaml(
        components_home,
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    result = runner.invoke(app, ["components", "set", "a.b,c.d", "5"])
    assert result.exit_code == 2, result.output
    assert "atomic" in result.output.lower() or "one path" in result.output.lower()


def test_audit_log_append_only(components_home):
    """Invariant: 3 sequential `set` calls produce exactly 3 ComponentMutated lines."""
    _write_project_yaml(
        components_home,
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    for val in ("85.0", "82.0", "78.0"):
        r = runner.invoke(
            app,
            ["components", "set", "org.context.compaction_threshold_pct", val],
        )
        assert r.exit_code == 0, r.output
    events = _read_audit(components_home)
    mutations = [e for e in events if e.get("event_type") == "ComponentMutated"]
    assert len(mutations) == 3, f"Expected 3 ComponentMutated lines, got {len(mutations)}"
