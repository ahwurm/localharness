"""The `agent.memory.*` settings surface, captured from the pre-plugin wiring (Phase 47 Wave 0).
These tests MUST pass unedited after memory becomes a plugin — they are the compatibility
invariant. Do not edit an assertion to make the conversion pass; a red here is a regression.

The path set is read from what `components list --json` actually renders (plugin rows included),
never from `MemoryConfig` directly — after the cut the rows come from the memory plugin's
AgentConfigModel, and the same literal must still hold.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()

AGENT_MEMORY_PATHS = frozenset({
    "agent.memory.sqlite_path",
    "agent.memory.history_path",
    "agent.memory.notes_path",
    "agent.memory.max_notes_chars",
    "agent.memory.shared_read",
    "agent.memory.recall_scope",
    "agent.memory.inject_into_context",
    "agent.memory.index_mode",
    "agent.memory.max_session_history_entries",
    "agent.memory.trace_ambient_injection",
    "agent.memory.embedding_model",
    "agent.memory.consolidation.enabled",
    "agent.memory.consolidation.idle_minutes",
    "agent.memory.consolidation.staleness_hours",
    "agent.memory.consolidation.iteration_cap",
    "agent.memory.archival.enabled",
})

CONFIG = (
    "version: '1'\n"
    "provider:\n"
    "  provider_type: ollama\n"
    "  base_url: http://localhost:11434/v1\n"
    "  default_model: test-model\n"
    "  api_key: none\n"
)


def _config_dir(tmp_path: Path, monkeypatch, extra: str = "") -> Path:
    d = tmp_path / "lh"
    d.mkdir()
    (d / "config.yaml").write_text(CONFIG + extra, encoding="utf-8")
    monkeypatch.setenv("LOCALHARNESS_DIR", str(d))
    return d


def _agent(d: Path, body: str) -> None:
    (d / "agents").mkdir(exist_ok=True)
    (d / "agents" / "probe.yaml").write_text(
        "name: probe\nrole: Probe\nmodel: inherit\n" + body, encoding="utf-8"
    )


def test_agent_memory_paths_are_exactly_the_sixteen(tmp_path, monkeypatch):
    _config_dir(tmp_path, monkeypatch)
    r = runner.invoke(app, ["components", "list", "--json"])
    assert r.exit_code == 0, r.output
    paths = {row["path"] for row in json.loads(r.stdout)}
    assert {p for p in paths if p.startswith("agent.memory.")} == AGENT_MEMORY_PATHS


def test_invalid_agent_memory_value_fails_at_agent_load(tmp_path, monkeypatch):
    from localharness.config.loader import ConfigLoader, ConfigValidationError

    d = _config_dir(tmp_path, monkeypatch)
    _agent(d, "memory:\n  max_notes_chars: -5\n")  # max_notes_chars is on line 5
    with pytest.raises(ConfigValidationError) as ei:
        ConfigLoader(config_dir=d).load_agent("probe")
    [err] = ei.value.errors
    assert err.field_path == "memory.max_notes_chars"
    assert err.yaml_line == 5
    assert err.value == -5


def test_both_memory_keys_are_accepted(tmp_path, monkeypatch):
    from localharness.config.loader import ConfigLoader

    d = _config_dir(tmp_path, monkeypatch, "org:\n  memory_enabled: false\n")
    _agent(d, "memory:\n  recall_scope: both\n")
    loader = ConfigLoader(config_dir=d)
    assert loader.load_harness().org.memory_enabled is False
    assert loader.load_agent("probe").memory.recall_scope == "both"


def test_components_set_recall_scope_both_is_accepted(tmp_path, monkeypatch):
    _config_dir(tmp_path, monkeypatch)
    ok = runner.invoke(app, ["components", "set", "agent.memory.recall_scope", "both"])
    assert ok.exit_code == 0, ok.output
    bad = runner.invoke(app, ["components", "set", "agent.memory.recall_scope", "nonsense"])
    assert bad.exit_code != 0, bad.output
