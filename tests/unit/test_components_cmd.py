"""REG-01..04 CLI surface tests. Phase 14-04.

Uses the `components_home` fixture from tests/conftest.py to isolate
LOCALHARNESS_HOME so config.yaml, overrides.yaml, and audit.jsonl all
live in a tmp dir for each test.
"""
from __future__ import annotations

import json
import re as _re
from pathlib import Path

import pytest
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
    # Band renamed in 43-03: `user` never said WHICH global file, and the word `project` meant the
    # global config.yaml. Asserted as the literal string on purpose — this is the word `components
    # get` prints, so the pin belongs on the vocabulary, not on the constant. NOTE the deliberate
    # asymmetry with the ComponentMutated audit event below (`layer == "user"`): that is a
    # different, older vocabulary written into a persisted log, and 43-03 does NOT rename it.
    assert payload["layer"] == "global-overrides"


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
    assert payload["layer"] == "global-overrides"


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


# ------------------------------------------------------------------ #
# PROP-02: proposer.* path enumeration (Phase 16 Wave 0 RED stub)
# ------------------------------------------------------------------ #

from localharness.config.models import ProposerConfig  # noqa: F401


@pytest.mark.plugin("autoresearch")
def test_proposer_paths_enumerated(components_home):
    """PROP-02: with a proposer block attached, `components list` surfaces `proposer.*` paths.

    Proves the registry walker auto-addresses ProposerConfig with zero extra registry code.
    """
    _write_project_yaml(
        components_home,
        proposer={
            "base_url": "http://localhost:11434/v1",
            "model": "frontier-strong:latest",
        },
        org={"audit_log_path": str(components_home / "audit.jsonl")},
    )
    result = runner.invoke(app, ["components", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    paths = [row["path"] for row in payload]
    assert any(p.startswith("proposer.") for p in paths), f"no proposer.* paths in {paths[:15]}..."


# ------------------------------------------------------------------ #
# Issue #22: `components set agent.*` must reach agent-scoped axes.
#
# Agent axes live in AgentConfig, NOT HarnessConfig. The set path validated the
# merged doc against HarnessConfig (extra="forbid", no `agent` key), so EVERY
# `set agent.<path> ...` died with "Extra inputs are not permitted" before any
# write — the documented safety lever (models.py: `components set
# agent.memory.consolidation.enabled <true|false>`) was structurally
# dead. The fix routes agent.* validation through AgentConfig and writes the same
# user-overlay `agent:` section that `get`/`load_agent` read back.
# ------------------------------------------------------------------ #

_AGENT_AXIS = "agent.memory.consolidation.enabled"


def _write_agent_yaml(home: Path, name: str, body: dict) -> None:
    """Seed home/agents/{name}.yaml so ConfigLoader().load_agent(name) can compose it."""
    agents_dir = home / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{name}.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")


def test_set_agent_axis_round_trip(components_home):
    """#22 (1): `set agent.<axis> false` succeeds and `get` reads it back false @ layer=user.

    RED before the fix: the set validated the merged doc against HarnessConfig and exited 2
    ("Extra inputs are not permitted" on `agent`).
    """
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    set_result = runner.invoke(app, ["components", "set", _AGENT_AXIS, "false"])
    assert set_result.exit_code == 0, set_result.output
    get_result = runner.invoke(app, ["components", "get", _AGENT_AXIS, "--json"])
    assert get_result.exit_code == 0, get_result.output
    payload = json.loads(get_result.stdout)
    assert payload["value"] is False
    assert payload["layer"] == "global-overrides"


def test_set_agent_axis_writes_agent_section_to_overlay(components_home):
    """#22 (1): the value lands under the overlay's `agent:` section — the layer get/load_agent read."""
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    r = runner.invoke(app, ["components", "set", _AGENT_AXIS, "false"])
    assert r.exit_code == 0, r.output
    overlay = yaml.safe_load((components_home / "overrides.yaml").read_text(encoding="utf-8"))
    assert overlay["agent"]["memory"]["consolidation"]["enabled"] is False


def test_loaded_agent_without_override_reflects_set_value(components_home):
    """#22 (2): a freshly loaded agent that does NOT set the axis reflects the overlay value."""
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    _write_agent_yaml(components_home, "plain", {"name": "plain", "role": "a plain agent"})
    r = runner.invoke(app, ["components", "set", _AGENT_AXIS, "false"])
    assert r.exit_code == 0, r.output
    from localharness.config.loader import ConfigLoader

    agent = ConfigLoader().load_agent("plain")
    assert agent.memory.consolidation.enabled is False


def test_per_agent_yaml_wins_over_set_value(components_home):
    """#22 (3): a per-agent yaml override BEATS the overlay default (the overlay is low-priority)."""
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    _write_agent_yaml(
        components_home,
        "opinionated",
        {"name": "opinionated", "role": "sets its own axis",
         "memory": {"consolidation": {"enabled": True}}},
    )
    r = runner.invoke(app, ["components", "set", _AGENT_AXIS, "false"])
    assert r.exit_code == 0, r.output
    from localharness.config.loader import ConfigLoader

    agent = ConfigLoader().load_agent("opinionated")
    assert agent.memory.consolidation.enabled is True


def test_set_org_axis_unchanged_by_fix(components_home):
    """#22 (4): the org.* set path is byte-identical — validates via HarnessConfig, no `agent` key grows."""
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    r = runner.invoke(app, ["components", "set", "org.context.compaction_threshold_pct", "85.0"])
    assert r.exit_code == 0, r.output
    overlay = yaml.safe_load((components_home / "overrides.yaml").read_text(encoding="utf-8"))
    assert overlay["org"]["context"]["compaction_threshold_pct"] == 85.0
    assert "agent" not in overlay


def test_set_org_axis_after_agent_axis_still_validates(components_home):
    """#22 (4, interaction): once the overlay carries an `agent:` section, a LATER org.* set must
    still succeed — the harness-path validation excludes `agent:` (not a HarnessConfig field).

    Caught by the live CLI drive: a naive fix that only special-cased the agent branch left the
    org branch merging the whole overlay (incl. `agent:`) into HarnessConfig -> 'Extra inputs'.
    """
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    r1 = runner.invoke(app, ["components", "set", _AGENT_AXIS, "false"])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["components", "set", "org.context.compaction_threshold_pct", "85.0"])
    assert r2.exit_code == 0, r2.output
    overlay = yaml.safe_load((components_home / "overrides.yaml").read_text(encoding="utf-8"))
    assert overlay["agent"]["memory"]["consolidation"]["enabled"] is False
    assert overlay["org"]["context"]["compaction_threshold_pct"] == 85.0
    # both axes still read back at layer=user
    for dotpath, expected in ((_AGENT_AXIS, False), ("org.context.compaction_threshold_pct", 85.0)):
        g = runner.invoke(app, ["components", "get", dotpath, "--json"])
        assert g.exit_code == 0, g.output
        payload = json.loads(g.stdout)
        assert payload["value"] == expected and payload["layer"] == "global-overrides", (dotpath, payload)


def test_set_invalid_agent_key_fails_clearly(components_home):
    """#22 (5): a genuinely invalid agent axis is refused (non-zero exit, no overlay write)."""
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    before = _read_overlay_bytes(components_home)
    r = runner.invoke(app, ["components", "set", "agent.memory.consolidation.does_not_exist", "5"])
    assert r.exit_code == 2, r.output
    assert _read_overlay_bytes(components_home) == before


def test_set_agent_axis_rejects_wrong_type(components_home):
    """#22 (6): the value validates through the REAL field type — a bool axis rejects 'banana'."""
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    before = _read_overlay_bytes(components_home)
    r = runner.invoke(app, ["components", "set", _AGENT_AXIS, "banana"])
    assert r.exit_code == 2, r.output
    assert _read_overlay_bytes(components_home) == before


def test_list_missing_config_points_at_init(tmp_path, monkeypatch):
    """#119: `components list` with no config on disk must end with the same next step
    `doctor` gives — 'Failed to load config: ...' alone leaves a new user with no way out.
    The exit code stays 2."""
    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(empty_home))

    r = runner.invoke(app, ["components", "list"])
    assert r.exit_code == 2, r.output
    flat = " ".join((r.output + (r.stderr or "")).split())
    assert "Failed to load config" in flat
    assert "Run 'localharness init' to create it." in flat


def test_set_agent_effort_is_refused_and_says_where_effort_lives(components_home):
    """D-08: `agent.effort` is not addressable through the registry, because the `agent:` overlay
    does not carry effort; the refusal names agents/<name>.yaml and /effort, and writes nothing."""
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    before = _read_overlay_bytes(components_home)
    result = runner.invoke(app, ["components", "set", "agent.effort", "low"])
    assert result.exit_code == 2, result.output
    assert "agents/<name>.yaml" in result.output
    assert "/effort" in result.output
    assert _read_overlay_bytes(components_home) == before


def test_get_agent_effort_says_the_same(components_home):
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    result = runner.invoke(app, ["components", "get", "agent.effort"])
    assert result.exit_code == 2, result.output
    assert "agents/<name>.yaml" in result.output


# ------------------------------------------------------------------ #
# write_value / audit_value: the checked write under `components set`
# ------------------------------------------------------------------ #

import asyncio  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

from localharness.cli.components_cmd import (  # noqa: E402
    Refusal,
    Written,
    audit_value,
    write_value,
)
from localharness.config.loader import ConfigLoader  # noqa: E402
from localharness.registry.catalogue import PluginRows, plugin_catalogue_rows  # noqa: E402
from localharness.registry.provenance import layered_catalogue  # noqa: E402

_PCT = "org.context.compaction_threshold_pct"


def _write_deps(home: Path, extra_plugins=()):
    _write_project_yaml(home, org={"audit_log_path": str(home / "audit.jsonl")})
    loader = ConfigLoader(config_dir=home)
    from localharness.plugins.resolve import resolve

    plugins = (*plugin_catalogue_rows(resolve(loader)), *extra_plugins)
    catalogue, _ = layered_catalogue(home, None, loader=loader, plugins=plugins)
    return loader, catalogue, plugins


def test_write_value_writes_the_machine_overlay_and_returns_written(components_home):
    loader, catalogue, plugins = _write_deps(components_home)
    written = write_value(loader, catalogue, plugins, _PCT, "85.0")
    assert isinstance(written, Written)
    assert (written.path, written.after) == (_PCT, 85.0)
    assert written.before == catalogue[_PCT].current_value
    assert written.overlay_path == loader.user_overlay_path
    assert written.audit_path == components_home / "audit.jsonl"
    assert yaml.safe_load(written.overlay_path.read_text(encoding="utf-8"))["org"]["context"][
        "compaction_threshold_pct"] == 85.0


def test_written_repr_shows_neither_value(components_home):
    loader, catalogue, plugins = _write_deps(components_home)
    written = write_value(loader, catalogue, plugins, _PCT, "85.5")
    assert "85.5" not in repr(written)
    assert repr(written.before) not in repr(written)


def test_written_is_frozen(components_home):
    loader, catalogue, plugins = _write_deps(components_home)
    written = write_value(loader, catalogue, plugins, _PCT, "85.0")
    with pytest.raises(Exception):
        written.path = "other"  # type: ignore[misc]


def test_write_value_audit_path_is_none_when_the_org_leaves_it_unset(components_home):
    loader, catalogue, plugins = _write_deps(components_home)
    (components_home / "config.yaml").write_text(
        yaml.safe_dump({"version": "1", "provider": {
            "provider_type": "ollama", "base_url": "http://localhost:11434/v1",
            "default_model": "test-model"}, "org": {"audit_log_path": ""}}), encoding="utf-8")
    loader = ConfigLoader(config_dir=components_home)
    catalogue, _ = layered_catalogue(components_home, None, loader=loader, plugins=plugins)
    assert write_value(loader, catalogue, plugins, _PCT, "85.0").audit_path is None


@pytest.mark.parametrize("path, raw, text", [
    ("no.such.path", "1", "Unknown path: 'no.such.path'"),
    ("agent.effort", "high", "agent.effort is not settable here"),
    (_PCT, "abc", "Cannot coerce 'abc' to float"),
    (_PCT, "150.0", "Validation failed for org.context.compaction_threshold_pct=150.0"),
])
def test_write_value_refuses_and_leaves_the_overlay_unchanged(components_home, path, raw, text):
    loader, catalogue, plugins = _write_deps(components_home)
    before = _read_overlay_bytes(components_home)
    with pytest.raises(Refusal) as info:
        write_value(loader, catalogue, plugins, path, raw)
    assert text in str(info.value)
    assert _read_overlay_bytes(components_home) == before


def test_write_value_refuses_a_value_the_plugins_own_model_refuses(components_home):
    class FakeConfig(BaseModel):
        n: int = Field(0, le=3)

    rows = PluginRows("fakeplug", True, False, FakeConfig, FakeConfig())
    loader, catalogue, plugins = _write_deps(components_home, (rows,))
    before = _read_overlay_bytes(components_home)
    with pytest.raises(Refusal) as info:
        write_value(loader, catalogue, plugins, "fakeplug.n", "9")
    assert "Validation failed for fakeplug.n=9" in str(info.value)
    assert "less than or equal to 3" in str(info.value)
    assert _read_overlay_bytes(components_home) == before


def test_write_value_refuses_a_failed_overlay_write(components_home, monkeypatch):
    loader, catalogue, plugins = _write_deps(components_home)

    def boom(path, overlay):
        raise OSError("disk full")

    monkeypatch.setattr("localharness.cli.components_cmd.atomic_write_overlay", boom)
    with pytest.raises(Refusal) as info:
        write_value(loader, catalogue, plugins, _PCT, "85.0")
    assert str(info.value) == f"Failed to write overlay {loader.user_overlay_path}: disk full"


def test_write_value_never_writes_a_workspace_overlay(components_home, tmp_path):
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    (ws / "overrides.yaml").write_text("org: {}\n", encoding="utf-8")
    ws_bytes = (ws / "overrides.yaml").read_bytes()
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    loader = ConfigLoader(config_dir=components_home, local_config_dir=ws)
    from localharness.plugins.resolve import resolve

    plugins = plugin_catalogue_rows(resolve(loader))
    catalogue, _ = layered_catalogue(components_home, ws, loader=loader, plugins=plugins)
    written = write_value(loader, catalogue, plugins, _PCT, "85.0")
    assert written.overlay_path == loader.user_overlay_path
    assert (ws / "overrides.yaml").read_bytes() == ws_bytes


def test_a_refusal_scrubs_its_secrets():
    assert "hunter2" not in str(Refusal("Cannot coerce 'hunter2' to SecretStr: bad", ["hunter2"]))


def test_audit_value_appends_one_component_mutated_entry(components_home):
    loader, catalogue, plugins = _write_deps(components_home)
    written = write_value(loader, catalogue, plugins, _PCT, "85.0")

    async def go():
        await audit_value(written)

    asyncio.run(go())
    mutations = [e for e in _read_audit(components_home) if e.get("event_type") == "ComponentMutated"]
    assert len(mutations) == 1
    assert (mutations[0]["path"], mutations[0]["after_value"]) == (_PCT, 85.0)
    assert mutations[0]["before_value"] == written.before
    assert (mutations[0]["layer"], mutations[0]["actor"]) == ("user", "cli")


def test_set_run_through_write_value_keeps_its_receipt(components_home):
    _write_project_yaml(components_home, org={"audit_log_path": str(components_home / "audit.jsonl")})
    result = runner.invoke(app, ["components", "set", _PCT, "85.0", "--json"])
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert (receipt["path"], receipt["after"], receipt["layer"]) == (_PCT, 85.0, "user")
